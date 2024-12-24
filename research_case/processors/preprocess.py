"""Data preprocessing module."""

import os
import re
import logging
from typing import Tuple, Dict
from collections import defaultdict  

from datetime import datetime

import pandas as pd
import emoji

from research_case.processors.conversation_extraction import ConversationExtractor

# Set up logging
logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self, input_file: str):
        
        """
        Initialize the preprocessor with input file path.
        
        Args:
            input_file: Path to the input CSV file
        """
        self.input_file = input_file
        self.df = None
        self.posts_df = None
        self.replies_df = None
        self.user_groups = None
        
    def _setup_output_directory(self, test: bool = False) -> None:
        """
        Set up the output directory based on whether this is a test run.
        
        Args:
            test: Boolean indicating if this is a test run
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = "Tests" if test else "Results"
        dir_prefix = "test_data" if test else "processed_data"
        self.output_dir = os.path.join(base_dir, f"{dir_prefix}_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Created output directory: {self.output_dir}")
        
    def load_data(self) -> None:
        """Load the input CSV file."""
        logger.info(f"Loading data from {self.input_file}")
        try:
            self.df = pd.read_csv(self.input_file)
            logger.info(f"Loaded {len(self.df)} rows")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
            
    def split_posts_replies(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        """
        Split data into posts and replies based on reply_to_id and reply_to_user.
        Also saves the split files for potential use by ConversationExtractor.
        
        Returns:
            Tuple of (posts_df, replies_df)
        """
        if self.df is None:
            self.load_data()
            
        # Identify replies (posts with reply_to_id or reply_to_user)
        is_reply = (self.df['reply_to_id'].notna()) | (self.df['reply_to_user'].notna())
        
        # Split into posts and replies
        self.posts_df = self.df[~is_reply].copy()
        self.replies_df = self.df[is_reply].copy()
        
        # Save split files if output directory exists
        if hasattr(self, 'output_dir'):
            posts_file = os.path.join(self.output_dir, "intermediate_posts.csv")
            replies_file = os.path.join(self.output_dir, "intermediate_replies.csv")
            
            self.posts_df.to_csv(posts_file, index=False)
            self.replies_df.to_csv(replies_file, index=False)
            
            logger.info(f"Saved intermediate split files to {self.output_dir}")
        
        logger.info(f"Split data into {len(self.posts_df)} posts and {len(self.replies_df)} replies")
        return self.posts_df, self.replies_df
    
    def filter_tweets(self, df, min_length=10):  # You can adjust min_length as needed
        # Create a copy to avoid modifying the original DataFrame
        filtered_df = df.copy()
        
        def is_valid_tweet(text):
            if not isinstance(text, str):
                return False
                
            # Remove URLs from text for length checking
            url_pattern = r'https?://\S+|www\.\S+'
            text_without_urls = re.sub(url_pattern, '', text).strip()
        
            # Check if text is only URLs
            if not text_without_urls:
                return False
                
            # Check minimum length after removing URLs
            if len(text_without_urls) < min_length:
                return False
                
            # Check if text contains only emojis
            text_without_emojis = ''.join(c for c in text_without_urls if c not in emoji.EMOJI_DATA)
            if not text_without_emojis.strip():
                return False
                
            # Check if text contains only @mentions
            mentions_pattern = r'^(@\S+\s*)+$'
            if re.match(mentions_pattern, text_without_urls.strip()):
                return False
                
            return True
    
        # Apply the filtering
        filtered_df = filtered_df[filtered_df['full_text'].apply(is_valid_tweet)]
        
        return filtered_df
    
        
    def group_users_by_id(self) -> Dict:
        """Group posts by user ID with error handling and memory efficiency."""
        logger.info("Grouping posts by user ID")
        
        try:
            # Ensure the correct column exists
            user_id_col = 'original_user_id'
            if user_id_col not in self.posts_df.columns:
                raise KeyError(f"Required column '{user_id_col}' not found in DataFrame")
                
            # Remove any rows with null user IDs
            valid_posts = self.posts_df[self.posts_df[user_id_col].notna()].copy()
            
            # Ensure consistent ID type (convert to string)
            valid_posts[user_id_col] = valid_posts[user_id_col].astype(str)
            
            # Select only necessary columns for grouping to save memory
            needed_columns = ['tweet_id', 'full_text', 'created_at', user_id_col]
            subset_df = valid_posts[needed_columns]
            
            # Group by user ID more efficiently
            self.user_groups = subset_df.groupby(user_id_col).apply(
                lambda x: x.drop(columns=[user_id_col]).to_dict('records')
            ).to_dict()
            
            logger.info(f"Grouped posts for {len(self.user_groups)} unique users")
            logger.debug(f"First user group sample: {next(iter(self.user_groups.values()))[:1]}")
            
            return self.user_groups
        
        except Exception as e:
            logger.error(f"Error in group_users_by_id: {str(e)}")
            raise
    
    def process(self, test: bool = False) -> Tuple[str, str, str, str]:
        """
        Process the data: load, split, filter, group, and save.
        Integrates with ConversationExtractor for conversation processing.
        
        Args:
            test: Boolean indicating if this is a test run
            
        Returns:
            Tuple of (posts_file_path, replies_file_path, users_file_path, conversations_file_path)
        """
        try:
            self._setup_output_directory(test)
            
            # Split and filter data
            self.split_posts_replies()
            
            logger.info("Filtering posts...")
            self.posts_df = self.filter_tweets(self.posts_df)
            logger.info(f"Retained {len(self.posts_df)} valid posts after filtering")
            
            # No Filtering of the replies to keep meaning 
            #logger.info("Filtering replies...")
            #self.replies_df = self.filter_tweets(self.replies_df)
            #logger.info(f"Retained {len(self.replies_df)} valid replies after filtering")
            
            # Group users
            self.group_users_by_id()
            
            # Initialize conversation extractor with our preprocessed data
            conversation_extractor = ConversationExtractor(
                replies_path=os.path.join(self.output_dir, "intermediate_replies.csv"),
                posts_path=os.path.join(self.output_dir, "intermediate_posts.csv")
            )
            
            # Extract conversations
            conversations = conversation_extractor.extract_conversations()
            
            # Generate filenames
            file_prefix = "test" if test else "processed"
            posts_file = os.path.join(self.output_dir, f"{file_prefix}_posts.csv")
            replies_file = os.path.join(self.output_dir, f"{file_prefix}_replies.csv")
            users_file = os.path.join(self.output_dir, f"{file_prefix}_users.json")
            conversations_file = os.path.join(self.output_dir, f"{file_prefix}_conversations.json")
            
            # Save outputs
            self.posts_df.to_csv(posts_file, index=False)
            self.replies_df.to_csv(replies_file, index=False)
            pd.Series(self.user_groups).to_json(users_file)
            pd.Series(conversations).to_json(conversations_file)
            
            # Get conversation statistics
            conv_stats = conversation_extractor.get_conversation_stats()
            logger.info(f"Conversation Statistics: {conv_stats}")
            
            logger.info(f"Saved all processed files to {self.output_dir}")
            
            # Clean up intermediate files
            os.remove(os.path.join(self.output_dir, "intermediate_posts.csv"))
            os.remove(os.path.join(self.output_dir, "intermediate_replies.csv"))
            
            return posts_file, replies_file, users_file, conversations_file
            
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            raise
