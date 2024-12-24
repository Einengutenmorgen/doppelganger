#conversation_extraion.py
import pandas as pd
import logging
from typing import Dict, List, Set
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationExtractor:
    def __init__(self, replies_path: str, posts_path: str, min_conversation_size: int = 2):
        """
        Initialize the conversation extractor with paths to replies and posts CSV files.
        
        Args:
            replies_path: Path to CSV containing only reply tweets
            posts_path: Path to CSV containing only original posts
            min_conversation_size: Minimum number of tweets to consider a conversation
        """
        self.replies_path = replies_path
        self.posts_path = posts_path
        self.min_conversation_size = min_conversation_size
        self.conversations = defaultdict(list)
        self.processed_ids = set()
        self.conversation_stats = {
            'total_conversations': 0,
            'meaningful_conversations': 0,  # conversations with size >= min_conversation_size
            'total_tweets_processed': 0,
            'conversation_sizes': [],
            'max_conversation_length': 0,
            'min_conversation_length': float('inf'),
            'avg_conversation_length': 0
        }

    def extract_conversations(self, chunk_size: int = 10000) -> Dict[str, List[dict]]:
        """
        Extract conversations from the CSV files efficiently.
        Uses chunking and tracks processed tweets to avoid duplicates.
        
        Args:
            chunk_size: Number of rows to process at once
            
        Returns:
            Dictionary with conversation IDs as keys and lists of related tweets as values
        """
        logger.info("Starting conversation extraction")
        
        # First pass: Build conversation trees and count sizes
        conversation_sizes = defaultdict(int)
        temp_conversations = defaultdict(list)
        
        for chunk_num, replies_chunk in enumerate(pd.read_csv(self.replies_path, chunksize=chunk_size)):
            logger.info(f"Processing chunk {chunk_num + 1}")
            
            # Drop rows we've already processed
            new_replies = replies_chunk[~replies_chunk['tweet_id'].isin(self.processed_ids)]
            
            if len(new_replies) == 0:
                continue
                
            # Process each reply in the chunk
            for _, reply in new_replies.iterrows():
                if pd.notna(reply['reply_to_id']):
                    conv_id = str(reply['reply_to_id'])
                    conversation_sizes[conv_id] += 1
                    temp_conversations[conv_id].append(reply.to_dict())
                    self.processed_ids.add(reply['tweet_id'])
                    
            logger.info(f"Processed {len(self.processed_ids)} tweets")

        # Second pass: Only keep conversations meeting minimum size
        meaningful_conversations = {
            conv_id: tweets 
            for conv_id, tweets in temp_conversations.items() 
            if conversation_sizes[conv_id] >= self.min_conversation_size
        }
        
        # Find original posts for meaningful conversations
        for conv_id in meaningful_conversations.keys():
            original_post = self._find_original_post(conv_id)
            if original_post is not None:
                meaningful_conversations[conv_id].insert(0, original_post)
                conversation_sizes[conv_id] += 1

        # Update statistics
        sizes = list(conversation_sizes.values())
        self.conversation_stats.update({
            'total_conversations': len(temp_conversations),
            'meaningful_conversations': len(meaningful_conversations),
            'total_tweets_processed': len(self.processed_ids),
            #'conversation_sizes': sizes,
            'max_conversation_length': max(sizes) if sizes else 0,
            'min_conversation_length': min(sizes) if sizes else 0,
            'avg_conversation_length': sum(sizes) / len(sizes) if sizes else 0
        })

        logger.info(f"Found {len(meaningful_conversations)} conversations with {self.min_conversation_size}+ tweets")
        return meaningful_conversations

    def _find_original_post(self, post_id: str) -> dict:
        """
        Find the original post for a given post ID from the posts CSV.
        Uses chunking to handle large files efficiently.
        
        Args:
            post_id: ID of the post to find
            
        Returns:
            Dictionary containing the post data or None if not found
        """
        chunk_size = 10000
        for posts_chunk in pd.read_csv(self.posts_path, chunksize=chunk_size):
            matching_post = posts_chunk[posts_chunk['tweet_id'] == float(post_id)]
            if not matching_post.empty:
                return matching_post.iloc[0].to_dict()
        return None

    def get_conversation_stats(self) -> Dict:
        """Get statistics about the extracted conversations."""
        return self.conversation_stats

    def get_longest_conversations(self, n: int = 5) -> List[Dict]:
        """
        Get the n longest conversations.
        
        Args:
            n: Number of conversations to return
            
        Returns:
            List of the n longest conversations
        """
        sorted_convs = sorted(
            self.conversations.items(), 
            key=lambda x: len(x[1]), 
            reverse=True
        )
        return dict(sorted_convs[:n])