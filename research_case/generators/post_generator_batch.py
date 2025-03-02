import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class BatchProcessor:
    """Handles batch processing of OpenAI API requests with state management"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        
    def prepare_stimulus_batch(
        self,
        personas: Dict,
        original_posts: Dict,
        posts_per_persona: int
    ) -> List[Dict]:
        """
        Creates batch requests for stimulus generation.
        
        Args:
            personas: Dictionary of user personas
            original_posts: Dictionary of original posts by user
            posts_per_persona: Number of posts to generate per persona
            
        Returns:
            List of batch requests
        """
        batch_requests = []
        
        for user_id, persona in personas.items():
            user_posts = original_posts.get(user_id, [])
            if not user_posts:
                logger.warning(f"No original posts found for user {user_id}")
                continue
                
            for i in range(min(posts_per_persona, len(user_posts))):
                original_post = user_posts[i].get('full_text', '')
                
                stimulus_prompt = f"""Describe the topic of the tweet with enough detail so that it can be reused to create a similar tweet. 
                It is important that you extract the topic of the tweet and phrase it as neutrally as possible.
                Tweet: "{original_post}"
                """
                
                request = {
                    "custom_id": f"{user_id}_stim_{i}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o",
                        "messages": [{"role": "user", "content": stimulus_prompt}],
                        "temperature": 0.2,
                        "response_format": {"type": "text"}
                    }
                }
                
                batch_requests.append(request)
                
        return batch_requests
        
    def process_batch(self, batch_requests: List[Dict], batch_path: Path) -> Dict:
        """
        Process a batch of requests through the OpenAI API.
        
        Args:
            batch_requests: List of request objects
            batch_path: Path to save the batch JSONL file
            
        Returns:
            Batch completion status
        """
        # Create batch file
        with open(batch_path, 'w') as f:
            for request in batch_requests:
                json.dump(request, f)
                f.write('\n')
                
        # Upload and submit batch
        input_file_id = self.llm_client.client.files.create(
            file=open(batch_path, "rb"),
            purpose="batch"
        ).id
        
        # Submit batch
        batch = self.llm_client.client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        
        batch_id = batch.id
        logger.info(f"Submitted batch {batch_id}")
        
        # Wait for completion
        while True:
            status = self.llm_client.client.batches.retrieve(batch_id)
            if status.status == "completed":
                return status
            elif status.status in ["failed", "cancelled"]:
                raise Exception(f"Batch failed with status: {status.status}")
            
            logger.info(
                f"Batch status: {status.status}, "
                f"completed: {status.request_counts.completed}/{status.request_counts.total}"
            )
            time.sleep(300)  # Check every 5 minutes

    def create_initial_structure(
        self,
        batch_results: Dict,
        personas: Dict,
        original_posts: Dict,
        batch_id: str
    ) -> Dict:
        """
        Create initial output structure with stimuli but no generated posts.
        
        Args:
            batch_results: Results from stimulus generation batch
            personas: Dictionary of user personas
            original_posts: Dictionary of original posts
            batch_id: ID of the stimulus batch
            
        Returns:
            Initial structure dictionary
        """
        # Parse batch results
        results_by_id = {}
        output_content = self.llm_client.client.files.retrieve_content(batch_results.output_file_id)
        for line in output_content.splitlines():
            result = json.loads(line)
            results_by_id[result['custom_id']] = result['response']['choices'][0]['message']['content']
            
        # Create structure
        generated_posts = []
        total_posts = 0
        
        for user_id, persona in personas.items():
            user_posts = original_posts.get(user_id, [])
            if not user_posts:
                continue
                
            for i, post in enumerate(user_posts):
                stim_id = f"{user_id}_stim_{i}"
                if stim_id not in results_by_id:
                    continue
                    
                record = {
                    "user_id": user_id,
                    "generation_id": f"{user_id}_gen_{i}",
                    "original_post_id": post.get('tweet_id', ''),
                    "original_text": post.get('full_text', ''),
                    "original_timestamp": post.get('created_at', ''),
                    "stimulus": results_by_id[stim_id],
                    "generated_text": None,  # placeholder
                    "generation_timestamp": None,  # placeholder
                }
                
                # Add persona fields
                record.update({
                    f"persona_{k}": v for k, v in persona.items()
                })
                
                generated_posts.append(record)
                total_posts += 1
                
        structure = {
            "metadata": {
                "generation_timestamp": datetime.now(timezone.utc).isoformat(),
                "num_users": len(personas),
                "posts_per_persona": len(next(iter(generated_posts))),
                "total_posts_generated": total_posts,
                "stimulus_batch_id": batch_id
            },
            "generated_posts": generated_posts
        }
        
        return structure
        
    def prepare_post_generation(self, saved_structure: Dict) -> List[Dict]:
        """
        Prepare post generation requests from saved structure.
        
        Args:
            saved_structure: Previously saved structure with stimuli
            
        Returns:
            List of batch requests for post generation
        """
        batch_requests = []
        
        for post in saved_structure['generated_posts']:
            # Skip any already generated posts
            if post['generated_text'] is not None:
                continue
                
            # Extract persona fields
            persona = {
                k.replace('persona_', ''): v 
                for k, v in post.items() 
                if k.startswith('persona_')
            }
            
            # Create prompt
            persona_section = "\n".join(
                f"{field.replace('_', ' ').title()}: {value}" 
                for field, value in persona.items()
            )
            
            generation_prompt = f"""You are a social media user with the following characteristics:
            {persona_section}
            
            Context: You are writing a social media post in response to the following stimulus:
            {post['stimulus']}
            
            Task: Write ONE social media post that this persona would create.
            
            Return a JSON object with this structure:
            {{"post_text": "Your generated post here"}}
            """
            
            request = {
                "custom_id": post['generation_id'],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": generation_prompt}],
                    "temperature": 0.7,
                    "response_format": {"type": "json_object"}
                }
            }
            
            batch_requests.append(request)
            
        return batch_requests
        
    def update_with_generated_posts(
        self,
        saved_structure: Dict,
        batch_results: Dict,
        batch_id: str
    ) -> Dict:
        """
        Update structure with generated posts from batch results.
        
        Args:
            saved_structure: Previously saved structure with stimuli
            batch_results: Results from post generation batch
            batch_id: ID of the post generation batch
            
        Returns:
            Updated structure
        """
        # Parse batch results
        results_by_id = {}
        output_content = self.llm_client.client.files.retrieve_content(batch_results.output_file_id)
        for line in output_content.splitlines():
            result = json.loads(line)
            results_by_id[result['custom_id']] = result['response']
            
        # Update posts
        timestamp = datetime.now(timezone.utc).isoformat()
        for post in saved_structure['generated_posts']:
            if post['generation_id'] in results_by_id:
                post['generated_text'] = results_by_id[post['generation_id']]
                post['generation_timestamp'] = timestamp
                
        # Update metadata
        saved_structure['metadata']['post_batch_id'] = batch_id
        saved_structure['metadata']['generation_timestamp'] = timestamp
        
        return saved_structure