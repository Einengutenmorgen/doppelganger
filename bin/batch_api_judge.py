#!/usr/bin/env python3
"""Evaluate generated social media posts using OpenAI's Batch API."""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any
import tempfile

# For OpenAI API
from openai import OpenAI

# Setup argument parser
def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate generated social media posts with Batch API.')
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to the JSON file containing generated posts'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o-mini',
        help='OpenAI model to use'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set the logging level'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='OpenAI API key (or set OPENAI_API_KEY environment variable)'
    )
    parser.add_argument(
        '--poll-interval',
        type=int,
        default=10,
        help='Seconds to wait between status checks for the batch'
    )
    return parser.parse_args()

def create_evaluation_prompt(original_post: str,
                           generated_post: str,
                           persona: Dict[str, str],
                           stimulus: str) -> List[Dict[str, Any]]:
    """Create the evaluation prompt for the LLM."""
    prompt_content = f"""You are an expert evaluator assessing the quality of an AI-generated social media post. Your task is to compare the generated post with the original and provide a structured assessment based on the following criteria:

1. **Authenticity (1-10):**  
   - Does the generated post reflect the unique tone, voice, and personality of the original author?  
   - Does it maintain their typical word choices, phrasing, and emotional nuances?  
   - Justify the score with a brief explanation.  

2. **Style Consistency (1-10):**  
   - How well does the generated post match the writing style of the original?  
   - Does it retain the sentence structure, rhythm, and distinctive linguistic patterns (idiolect/sociolect)?  
   - Provide an explanation for the score.  

3. **Intent Matching (True/False):**  
   - Does the generated post preserve the key message, emotional impact, and overall intent of the original post?  
   - If False, briefly explain the discrepancies.  

**Original Post:**  
{original_post}

**Generated Post:**  
{generated_post}

**Persona Information:**
{json.dumps(persona, indent=2)}

**Stimulus:**
{stimulus}

Return the evaluation in the following structured JSON format:
{{
    "authenticity": {{
        "score": <integer 1-10>,
        "explanation": "<brief explanation>"
    }},
    "style_consistency": {{
        "score": <integer 1-10>,
        "explanation": "<brief explanation>"
    }},
    "matching_intent": <true/false>,
    "overall_feedback": "<brief assessment summarizing the strengths and weaknesses of the generated post>"
}}"""

    return [
        {"role": "system", "content": prompt_content}
    ]

def parse_analysis(response: str) -> Dict:
    """Parse the LLM response into a structured analysis dictionary."""
    try:
        # Parse JSON response
        analysis = json.loads(response)
        
        # Required fields and their expected types
        required_fields = {
            "authenticity": dict,
            "style_consistency": dict,
            "matching_intent": bool,
            "overall_feedback": str
        }
        
        # Required subfields for dictionary fields
        required_subfields = {
            "authenticity": {"score": int, "explanation": str},
            "style_consistency": {"score": int, "explanation": str}
        }
        
        # Check required fields and their types
        for field, expected_type in required_fields.items():
            if field not in analysis:
                raise KeyError(f"Missing required field: {field}")
            
            if not isinstance(analysis[field], expected_type):
                raise TypeError(f"Field {field} must be of type {expected_type.__name__}")
            
            # Special handling for dictionary fields
            if expected_type == dict and field in required_subfields:
                for subfield, subfield_type in required_subfields[field].items():
                    if subfield not in analysis[field]:
                        raise KeyError(f"Missing required subfield {subfield} in {field}")
                    
                    if not isinstance(analysis[field][subfield], subfield_type):
                        raise TypeError(f"Subfield {subfield} in {field} must be of type {subfield_type.__name__}")
                    
                    # Check for non-empty strings
                    if subfield_type == str and not str(analysis[field][subfield]).strip():
                        raise ValueError(f"Subfield {subfield} in {field} cannot be empty")
            
            # Check non-empty string for overall_feedback
            elif expected_type == str and not str(analysis[field]).strip():
                raise ValueError(f"Field {field} cannot be empty")
        
        return analysis
        
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse LLM response as JSON: {e}")
        raise
    except (KeyError, TypeError, ValueError) as e:
        logging.error(f"Invalid analysis structure: {e}")
        raise

def get_default_evaluation() -> Dict:
    """Return default evaluation if LLM call fails."""
    return {
        "authenticity": {
            "score": 0,
            "explanation": "Evaluation failed"
        },
        "style_consistency": {
            "score": 0,
            "explanation": "Evaluation failed"
        },
        "matching_intent": False,
        "overall_feedback": "LLM evaluation failed"
    }

def save_results(results: dict, output_path: str):
    """Save evaluation results to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    logging.info(f"Evaluation results saved to {output_path}")

def create_batch_input_file(posts: List[Dict], model: str) -> str:
    """Create a JSONL file for batch processing."""
    # Create a temporary file
    fd, temp_path = tempfile.mkstemp(suffix='.jsonl')
    try:
        with os.fdopen(fd, 'w') as f:
            for post in posts:
                try:
                    # Extract persona data
                    persona = {k.replace('persona_', ''): v for k, v in post.items() if k.startswith('persona_')}
                    
                    # Create messages
                    messages = create_evaluation_prompt(
                        original_post=post['original_text'],
                        generated_post=post['generated_text'],
                        persona=persona,
                        stimulus=post['stimulus']
                    )
                    
                    # Create the batch request entry
                    batch_entry = {
                        "custom_id": post['generation_id'],
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": model,
                            "messages": messages,
                            "temperature": 0.1,
                            "response_format": {"type": "json_object"}
                        }
                    }
                    
                    # Write as a line in the JSONL file
                    f.write(json.dumps(batch_entry) + '\n')
                    
                except Exception as e:
                    logging.error(f"Error creating batch entry for post {post.get('generation_id', 'unknown')}: {e}")
    except Exception as e:
        os.unlink(temp_path)
        raise e
        
    return temp_path

def main():
    """Main execution function."""
    args = setup_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print('Starting Batch Evaluation Process....')
    
    try:
        # Get API key
        api_key = args.api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key must be provided via --api-key or OPENAI_API_KEY environment variable")
            
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Load input data
        logging.info('Loading data...')
        with open(args.input, 'r') as f:
            data = json.load(f)
            
        posts = data.get('generated_posts', [])
        if not posts:
            logging.error("No posts found in the input file")
            return
            
        num_posts = len(posts)
        logging.info(f"Found {num_posts} posts to evaluate")
        
        # 1. Create batch input file
        start_time = time.time()
        logging.info("Creating batch input file...")
        batch_input_path = create_batch_input_file(posts, args.model)
        logging.info(f"Created batch input file at {batch_input_path}")
        
        # 2. Upload the file
        logging.info("Uploading batch input file...")
        with open(batch_input_path, 'rb') as f:
            file_obj = client.files.create(
                file=f,
                purpose="batch"
            )
        logging.info(f"Uploaded file with ID: {file_obj.id}")
        
        # 3. Create the batch
        logging.info("Creating batch processing job...")
        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": f"Post evaluation batch for {num_posts} posts"
            }
        )
        logging.info(f"Created batch with ID: {batch.id}")
        
        # 4. Poll for completion
        logging.info("Waiting for batch to complete...")
        
        # Poll until batch is complete or fails
        while batch.status not in ["completed", "failed", "expired", "cancelled"]:
            logging.info(f"Batch status: {batch.status} - Sleeping for {args.poll_interval} seconds")
            time.sleep(args.poll_interval)
            batch = client.batches.retrieve(batch.id)
            
            # Show progress
            if batch.request_counts and batch.request_counts.total > 0:
                completed = batch.request_counts.completed
                total = batch.request_counts.total
                progress = (completed / total) * 100
                logging.info(f"Progress: {completed}/{total} ({progress:.2f}%)")
        
        # 5. Check final status
        if batch.status != "completed":
            logging.error(f"Batch failed with status: {batch.status}")
            # Try to get error information if available
            if batch.error_file_id:
                error_content = client.files.content(batch.error_file_id)
                logging.error(f"Batch errors: {error_content.text[:1000]}...")
            return
            
        # 6. Retrieve results
        logging.info("Batch completed. Retrieving results...")
        output_content = client.files.content(batch.output_file_id)
        
        # 7. Process results
        results = []
        for line in output_content.text.splitlines():
            try:
                result_data = json.loads(line)
                post_id = result_data.get("custom_id")
                
                if post_id and result_data.get("response") and result_data["response"].get("body"):
                    # Extract the LLM response
                    llm_response = result_data["response"]["body"]
                    content = None
                    
                    # Navigate through the response structure to get the content
                    if llm_response.get("choices") and len(llm_response["choices"]) > 0:
                        content = llm_response["choices"][0].get("message", {}).get("content")
                    
                    if content:
                        try:
                            evaluation = parse_analysis(content)
                            results.append({
                                'post_id': post_id,
                                'evaluation': evaluation
                            })
                        except Exception as e:
                            logging.error(f"Error parsing response for post {post_id}: {e}")
                            results.append({
                                'post_id': post_id,
                                'evaluation': get_default_evaluation()
                            })
                    else:
                        logging.error(f"No content found in response for post {post_id}")
                        results.append({
                            'post_id': post_id,
                            'evaluation': get_default_evaluation()
                        })
                elif result_data.get("error"):
                    logging.error(f"Error in batch result for post {post_id}: {result_data['error']}")
                    results.append({
                        'post_id': post_id,
                        'evaluation': get_default_evaluation()
                    })
            except Exception as e:
                logging.error(f"Error processing batch result line: {e}")
        
        # Calculate processing time
        end_time = time.time()
        total_time = round(end_time - start_time, 2)
        
        # Prepare output path
        output_filename = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_dir = args.output_dir or os.path.dirname(args.input)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        
        # Save results
        save_results({
            'metadata': {
                'total_evaluated': len(results),
                'timestamp': datetime.now().isoformat(),
                'model': args.model,
                'batch_id': batch.id,
                'processing_time_seconds': total_time
            },
            'results': results
        }, output_path)
        
        print(f"\nEvaluation complete! Results saved to: {output_path}")
        print(f"Total posts evaluated: {len(results)}")
        print(f"Total processing time: {total_time} seconds")
        
        # Clean up temp file
        try:
            os.unlink(batch_input_path)
        except:
            pass
            
    except FileNotFoundError:
        logging.error(f"Input file not found: {args.input}")
        sys.exit(1)
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in input file: {args.input}")
        sys.exit(1)
    except Exception as e:
        logging.debug("Traceback:", exc_info=True)
        logging.error(f"Error running evaluation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()