"""Optimized data preprocessing module for large CSV files."""

import os
import re
import logging
from typing import Tuple, Dict, Generator
from collections import defaultdict
from datetime import datetime
import pandas as pd
import emoji
from tqdm import tqdm
import psutil
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, input_file: str, chunk_size: int = 100000):
        """Initialize preprocessor with chunking support."""
        self.input_file = input_file
        self.chunk_size = chunk_size
        self.output_dir = None
        self.total_rows = None
        self._count_rows()
        
    def _count_rows(self) -> None:
        """Count total rows in CSV for progress tracking."""
        logger.info("Counting total rows...")
        self.total_rows = sum(1 for _ in open(self.input_file)) - 1  # Subtract header
        logger.info(f"Total rows to process: {self.total_rows:,}")

    def _setup_output_directory(self, test: bool = False) -> None:
        """Set up output directory with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = "Tests" if test else "Results"
        dir_prefix = "test_data" if test else "processed_data"
        self.output_dir = os.path.join(base_dir, f"{dir_prefix}_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Created output directory: {self.output_dir}")
        
    def _process_csv_chunks(self) -> Generator[pd.DataFrame, None, None]:
        """Process CSV in chunks to manage memory."""
        with tqdm(total=self.total_rows, desc="Processing CSV") as pbar:
            for chunk in pd.read_csv(self.input_file, chunksize=self.chunk_size):
                pbar.update(len(chunk))
                yield chunk
                
    def split_posts_replies(self) -> Tuple[str, str]:
        """Split data into posts and replies files using chunks."""
        posts_file = os.path.join(self.output_dir, "intermediate_posts.csv")
        replies_file = os.path.join(self.output_dir, "intermediate_replies.csv")
        
        header = True  # Write header only for first chunk
        for chunk in self._process_csv_chunks():
            # Split chunk
            is_reply = (chunk['reply_to_id'].notna()) | (chunk['reply_to_user'].notna())
            posts = chunk[~is_reply]
            replies = chunk[is_reply]
            
            # Append to files
            posts.to_csv(posts_file, mode='a', header=header, index=False)
            replies.to_csv(replies_file, mode='a', header=header, index=False)
            header = False  # Don't write header for subsequent chunks
            
        return posts_file, replies_file
    
    def filter_tweets(self, input_file: str, output_file: str, min_length: int = 25) -> None:
            """
            Filter tweets from input file to output file in chunks with enhanced filtering criteria:
            - Remove all posts containing URLs (including shortened URLs)
            - Filter out tweets shorter than specified minimum length (default 25 characters)
            - Filter out non-English texts using fasttext
            - Filter out tweets with more than 1 mention
            
            Args:
                input_file: Path to input CSV file
                output_file: Path to output CSV file
                min_length: Minimum length of tweet text (default 25 characters)
            """

            
            def is_valid_tweet(row):
                try:
                    text = row['full_text']
                    
                    # Basic type check
                    if not isinstance(text, str):
                        return False
                        
                    # Check for URLs (enhanced pattern to catch short URLs)
                    url_pattern = r'(https?:\/\/|www\.)\S+|bit\.ly\/\S+|t\.co\/\S+|goo\.gl\/\S+|tinyurl\.com\/\S+'
                    if re.search(url_pattern, text, re.IGNORECASE):
                        return False
                        
                    # Count mentions
                    mention_pattern = r'@\w+'
                    mentions = re.findall(mention_pattern, text)
                    if len(mentions) > 1:
                        return False
                        
                    # Remove mentions and clean text
                    text_without_mentions = re.sub(mention_pattern, '', text)
                    clean_text = text_without_mentions.strip()
                        
                    # Check minimum length
                    if len(clean_text) < min_length:
                        return False
                        
                    # Check for emoji-only content
                    text_without_emojis = ''.join(c for c in clean_text if c not in emoji.EMOJI_DATA)
                    if not text_without_emojis.strip():
                        return False
                        
                    # Check language using fasttext
                    try:
                        lang = detect(clean_text)
                        if lang != 'en':
                            return False
                    except LangDetectException:
                        logger.error("Language detection failed")
                        return False
                        
                    return True
                    
                except Exception as e:
                    logger.error(f"Error processing tweet: {e}")
                    return False

            try:
                header = True
                for chunk in pd.read_csv(input_file, chunksize=self.chunk_size):
                    try:
                        filtered_chunk = chunk[chunk.apply(is_valid_tweet, axis=1)]
                        filtered_chunk.to_csv(output_file, mode='a', header=header, index=False)
                        header = False
                        
                        # Log progress
                        logger.info(f"Processed chunk: {len(chunk)} rows, kept {len(filtered_chunk)} rows")
                        
                    except Exception as e:
                        logger.error(f"Error processing chunk: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Fatal error in filter_tweets: {e}")
                raise
                
            
    def group_users_by_id(self, posts_file: str) -> Dict:
        """Group posts by user ID efficiently."""
        user_groups = defaultdict(list)
        
        for chunk in pd.read_csv(posts_file, chunksize=self.chunk_size):
            for _, row in chunk.iterrows():
                if pd.notna(row['original_user_id']):
                    user_groups[str(row['original_user_id'])].append({
                        'tweet_id': row['tweet_id'],
                        'full_text': row['full_text'],
                        'created_at': row['created_at']
                    })
                    
        return dict(user_groups)
    
    def process(self, test: bool = False) -> Tuple[str, str, str, str]:
        try:
            self._setup_output_directory(test)
            
            # Track initial memory
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
            logger.info(f"Initial memory usage: {initial_memory:.2f} MB")
            
            # Split data
            logger.info("Splitting posts and replies...")
            posts_file, replies_file = self.split_posts_replies()
            
            # Filter posts
            logger.info("Filtering posts...")
            filtered_posts = os.path.join(self.output_dir, "filtered_posts.csv")
            self.filter_tweets(posts_file, filtered_posts)
            
            # Group users
            logger.info("Grouping users...")
            user_groups = self.group_users_by_id(filtered_posts)
            
            # Generate output files
            file_prefix = "test" if test else "processed"
            final_posts = os.path.join(self.output_dir, f"{file_prefix}_posts.csv")
            final_replies = os.path.join(self.output_dir, f"{file_prefix}_replies.csv")
            users_file = os.path.join(self.output_dir, f"{file_prefix}_users.json")
            conversations_file = os.path.join(self.output_dir, f"{file_prefix}_conversations.json")
            
            # Move/rename files
            os.rename(filtered_posts, final_posts)
            os.rename(replies_file, final_replies)
            
            # Save user groups
            pd.Series(user_groups).to_json(users_file)
            
            # Extract conversations
            db_path = os.path.join(self.output_dir, "conversations.db")
            from research_case.processors.conversation_extraction import ConversationExtractor
            conversation_extractor = ConversationExtractor(
                replies_file=final_replies,
                posts_file=final_posts
                #db_path=db_path
            )
            conversations = conversation_extractor.extract_conversations()
            pd.Series(conversations).to_json(conversations_file)
            
            # Clean up temporary database
            #if os.path.exists(db_path):
            #    os.remove(db_path)
            
            # Final memory usage
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            logger.info(f"Final memory usage: {final_memory:.2f} MB")
            logger.info(f"Memory increase: {final_memory - initial_memory:.2f} MB")
            
            return final_posts, final_replies, users_file, conversations_file
            
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            raise