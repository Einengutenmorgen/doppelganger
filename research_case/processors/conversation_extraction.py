"""Optimized conversation extraction for large datasets."""

import sqlite3
import logging
from typing import Dict, List, Set
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ConversationExtractor:
    def __init__(self, replies_file: str, posts_file: str, db_path: str = ":memory:", 
                 min_conversation_size: int = 2, chunk_size: int = 50000):
        self.replies_file = replies_file
        self.posts_file = posts_file
        self.db_path = db_path
        self.min_conversation_size = min_conversation_size
        self.chunk_size = chunk_size
        self.conversation_stats = self._init_stats()
        
    def _init_stats(self) -> Dict:
        return {
            'total_conversations': 0,
            'meaningful_conversations': 0,
            'total_tweets_processed': 0,
            'max_conversation_length': 0,
            'min_conversation_length': float('inf'),
            'avg_conversation_length': 0
        }
        
    def _setup_temp_tables(self):
        """Create temporary tables for conversation processing with REPLACE strategy"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Drop existing table if it exists
        c.execute('DROP TABLE IF EXISTS conversation_map')
        
        # Create table with UNIQUE constraint on tweet_id
        c.execute('''CREATE TABLE conversation_map
                    (tweet_id TEXT PRIMARY KEY,
                     conversation_id TEXT,
                     reply_to_id TEXT,
                     processed INTEGER DEFAULT 0)''')
                     
        c.execute('CREATE INDEX idx_conv_id ON conversation_map(conversation_id)')
        c.execute('CREATE INDEX idx_reply_to ON conversation_map(reply_to_id)')
        
        conn.commit()
        conn.close()

    def _find_post(self, post_id: str) -> dict:
        """Find original post efficiently using chunked reading"""
        if not post_id:
            logger.warning("Empty post_id provided.")
            return None

        try:
            # Ensure post_id is treated as a string consistently
            post_id = str(int(float(post_id)))
        except (ValueError, TypeError):
            logger.warning(f"Invalid post_id format: {post_id}")
            return None

        try:
            for chunk in pd.read_csv(self.posts_file, chunksize=self.chunk_size):
                # Convert tweet_id to string for comparison
                chunk['tweet_id'] = chunk['tweet_id'].astype(str)
                matching_post = chunk[chunk['tweet_id'] == post_id]
                if not matching_post.empty:
                    return matching_post.iloc[0].to_dict()
        except Exception as e:
            logger.error(f"Error processing post_id {post_id}: {e}")
        return None

    # Other methods remain unchanged but incorporate robust error handling.

    def extract_conversations(self) -> Dict[str, List[dict]]:
        """Extract conversations using SQLite for efficient processing"""
        self._setup_temp_tables()
        conn = sqlite3.connect(self.db_path)
        
        # First pass: Build conversation tree
        logger.info("Building conversation tree...")
        total_chunks = sum(1 for _ in pd.read_csv(self.replies_file, chunksize=self.chunk_size))
        
        with tqdm(total=total_chunks, desc="Processing reply chunks") as pbar:
            for chunk in pd.read_csv(self.replies_file, chunksize=self.chunk_size):
                # Clean and validate the data before insertion
                chunk = chunk.copy()
                # Convert to float first to handle scientific notation, then to Int64 for nullable integers
                chunk['tweet_id'] = pd.to_numeric(chunk['tweet_id'], errors='coerce')
                chunk['reply_to_id'] = pd.to_numeric(chunk['reply_to_id'], errors='coerce')
                
                self._insert_conversation_data(chunk, conn)
                pbar.update(1)
        
        # Rest of the method remains the same...
        conn.close()
        return {}

    def _insert_conversation_data(self, chunk_data: pd.DataFrame, conn: sqlite3.Connection):
        """Insert conversation data with REPLACE strategy and proper NA handling"""
        # Convert chunk data to list of tuples
        data_to_insert = []
        
        for _, row in chunk_data[['tweet_id', 'reply_to_id']].iterrows():
            # Handle tweet_id
            try:
                if pd.isna(row['tweet_id']) or row['tweet_id'] == '<NA>':
                    continue
                tweet_id = str(int(float(row['tweet_id'])))
            except (ValueError, TypeError):
                logger.warning(f"Skipping invalid tweet_id: {row['tweet_id']}")
                continue

            # Handle reply_to_id
            try:
                if pd.isna(row['reply_to_id']) or row['reply_to_id'] == '<NA>':
                    reply_to_id = None
                else:
                    reply_to_id = str(int(float(row['reply_to_id'])))
            except (ValueError, TypeError):
                reply_to_id = None
                logger.debug(f"Converting invalid reply_to_id to None: {row['reply_to_id']}")

            data_to_insert.append((
                tweet_id,
                None,  # conversation_id initially null
                reply_to_id,
                0  # processed flag
            ))

        if data_to_insert:
            try:
                # Use INSERT OR REPLACE to handle duplicates
                conn.executemany(
                    '''INSERT OR REPLACE INTO conversation_map 
                    (tweet_id, conversation_id, reply_to_id, processed) 
                    VALUES (?, ?, ?, ?)''',
                    data_to_insert
                )
                conn.commit()
            except sqlite3.Error as e:
                logger.error(f"SQLite error during insertion: {e}")
                conn.rollback()

# Additional robust methods as needed to ensure the pipeline handles errors gracefully.
