"""Optimized conversation extraction for large datasets."""

import sqlite3
import logging
from typing import Dict, List
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import re
import emoji
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException


logger = logging.getLogger(__name__)

class ConversationExtractor:
    def __init__(self, replies_file: str, posts_file: str, 
                 min_conversation_size: int = 2, chunk_size: int = 50000):
        self.replies_file = replies_file
        self.posts_file = posts_file
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
            'avg_conversation_length': 0,
            'total_conversations_before_filter': 0,  # Added for filtering stats
            'conversations_after_filter': 0,         # Added for filtering stats
            'filter_retention_rate': 0              # Added for filtering stats
        }

    def _setup_database(self):
        """Initialize in-memory SQLite database with proper schema"""
        conn = sqlite3.connect(':memory:')
        c = conn.cursor()
        
        # Create table with proper type enforcement
        c.execute('''CREATE TABLE conversation_map
                    (tweet_id TEXT PRIMARY KEY, 
                     reply_to_id TEXT,
                     created_at TEXT,
                     full_text TEXT,
                     original_user_id TEXT,
                     CONSTRAINT tweet_id_not_null CHECK (tweet_id IS NOT NULL))''')
        
        # Create index for better query performance
        c.execute('CREATE INDEX idx_reply_to ON conversation_map(reply_to_id)')
        return conn

    def _process_chunk(self, chunk: pd.DataFrame, conn: sqlite3.Connection):
        """Process a single chunk of data"""
        try:
            # Convert float64 to string properly
            chunk['tweet_id'] = chunk['tweet_id'].apply(lambda x: f'{x:.0f}' if pd.notna(x) else None)
            chunk['reply_to_id'] = chunk['reply_to_id'].apply(lambda x: f'{x:.0f}' if pd.notna(x) else None)
            
            # Filter out nulls
            valid_data = chunk.dropna(subset=['tweet_id'])
            
            # Insert with IGNORE for the rare duplicates
            records = valid_data[['tweet_id', 'reply_to_id', 'created_at', 'full_text', 'original_user_id']].values.tolist()
            conn.executemany(
                'INSERT OR IGNORE INTO conversation_map VALUES (?, ?, ?, ?, ?)',
                records
            )
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            conn.rollback()
            raise

    def extract_conversations(self) -> Dict[str, List[dict]]:
        """Extract conversations using SQLite for efficient processing"""
        conn = self._setup_database()
        
        try:
            # Load replies into SQLite
            logger.info("Loading replies into database...")
            total_replies = 0
            valid_replies = 0
            
            for chunk in tqdm(pd.read_csv(self.replies_file, chunksize=self.chunk_size)):
                chunk_size = len(chunk)
                total_replies += chunk_size
                self._process_chunk(chunk, conn)
                
                c = conn.cursor()
                c.execute('SELECT COUNT(*) FROM conversation_map')
                valid_replies = c.fetchone()[0]
                
            logger.info(f"Processed {total_replies:,} total replies, {valid_replies:,} valid replies stored")

            c = conn.cursor()
            c.execute('SELECT COUNT(DISTINCT reply_to_id) FROM conversation_map WHERE reply_to_id IS NOT NULL')
            unique_parents = c.fetchone()[0]
            logger.info(f"Found {unique_parents:,} unique parent tweets")

            # Find conversations more efficiently
            logger.info("Identifying conversation threads...")
            c.execute('''
                WITH conversation_sizes AS (
                    SELECT reply_to_id, COUNT(*) as reply_count,
                        GROUP_CONCAT(tweet_id) as reply_ids
                    FROM conversation_map
                    WHERE reply_to_id IS NOT NULL
                    GROUP BY reply_to_id
                    HAVING COUNT(*) >= ?
                )
                SELECT reply_to_id, reply_count, reply_ids
                FROM conversation_sizes
                ORDER BY reply_count DESC
            ''', (self.min_conversation_size,))
            
            potential_threads = c.fetchall()
            logger.info(f"Found {len(potential_threads):,} potential conversation threads")

            conversations = {}
            processed = 0
            
            for root_id, reply_count, reply_ids in potential_threads:
                try:
                    # Split reply_ids string into list
                    reply_id_list = reply_ids.split(',')
                    
                    # Create parameterized query with correct number of placeholders
                    placeholders = ','.join(['?' for _ in range(len(reply_id_list) + 1)])  # +1 for root_id
                    query = f'''
                        SELECT tweet_id, reply_to_id, created_at, full_text, original_user_id
                        FROM conversation_map
                        WHERE tweet_id IN ({placeholders})
                        ORDER BY created_at
                    '''
                    
                    # Execute query with all IDs (including root_id)
                    c.execute(query, [root_id] + reply_id_list)
                    
                    messages = [
                        dict(zip(['tweet_id', 'reply_to_id', 'created_at', 'full_text', 'original_user_id'], row))
                        for row in c.fetchall()
                    ]
                    
                    if len(messages) >= self.min_conversation_size:
                        conversations[root_id] = messages
                        self._update_stats(len(messages))
                        processed += 1
                        
                        if processed % 1000 == 0:
                            logger.info(f"Processed {processed:,} conversations")
                            
                except Exception as e:
                    logger.error(f"Error processing conversation {root_id}: {e}")
                    continue

            logger.info(f"Extracted {len(conversations):,} complete conversations")
            
            # Apply filtering to conversation roots
            logger.info("Applying root message filtering...")
            filtered_conversations = self.filter_conversations(conversations)
            
            logger.info(f"Filtering complete. Retained {len(filtered_conversations)} conversations")
            
            if filtered_conversations:
                sample_convo = next(iter(filtered_conversations.values()))
                logger.info(f"Sample filtered conversation size: {len(sample_convo)}")
                logger.info(f"Sample conversation messages:")
                for msg in sample_convo[:3]:  # Show first 3 messages
                    logger.info(f"Tweet ID: {msg['tweet_id']}, Reply to: {msg['reply_to_id']}")
            
            return filtered_conversations
            
        except Exception as e:
            logger.error(f"Error extracting conversations: {e}")
            raise
            
        finally:
            conn.close()

    def _update_stats(self, conv_size: int) -> None:
        """Update conversation statistics"""
        self.conversation_stats['total_conversations'] += 1
        if conv_size >= self.min_conversation_size:
            self.conversation_stats['meaningful_conversations'] += 1
        self.conversation_stats['max_conversation_length'] = max(
            self.conversation_stats['max_conversation_length'],
            conv_size
        )
        self.conversation_stats['min_conversation_length'] = min(
            self.conversation_stats['min_conversation_length'],
            conv_size
        )
        self.conversation_stats['total_tweets_processed'] += conv_size

    def get_conversation_stats(self) -> Dict:
        """Get current conversation statistics"""
        if self.conversation_stats['meaningful_conversations'] > 0:
            self.conversation_stats['avg_conversation_length'] = (
                self.conversation_stats['total_tweets_processed'] /
                self.conversation_stats['meaningful_conversations']
            )
        return self.conversation_stats

    def filter_conversations(self, conversations: Dict[str, List[dict]], min_length: int = 25) -> Dict[str, List[dict]]:
        """
        Filter conversations based on root message criteria.
        Uses the same filtering logic as posts to validate conversation root messages.
        
        Args:
            conversations: Dictionary of conversation threads
            min_length: Minimum length for root messages (default: 25 characters)
            
        Returns:
            Dictionary of filtered conversations
        """
        
        def is_valid_root(text: str) -> bool:
            try:
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
                logger.error(f"Error processing root message: {e}")
                return False
        
        try:
            filtered_conversations = {}
            total_convs = len(conversations)
            kept_convs = 0
            
            logger.info(f"Starting conversation filtering. Total conversations: {total_convs}")
            
            for conv_id, messages in conversations.items():
                if not messages:  # Skip empty conversations
                    continue
                    
                # Get root message (first message in the conversation)
                root_message = messages[0]
                root_text = root_message.get('full_text', '')
                
                # Apply filtering to root message
                if is_valid_root(root_text):
                    filtered_conversations[conv_id] = messages
                    kept_convs += 1
                    
                    if kept_convs % 1000 == 0:
                        logger.info(f"Processed {kept_convs} valid conversations so far...")
            
            logger.info(f"Conversation filtering complete. Kept {kept_convs} out of {total_convs} conversations")
            
            # Update conversation stats
            self._update_filtered_stats(total_convs, kept_convs)
            
            return filtered_conversations
            
        except Exception as e:
            logger.error(f"Error in conversation filtering: {e}")
            raise
     
            

    def _update_filtered_stats(self, total: int, kept: int) -> None:
        """Update statistics after filtering."""
        self.conversation_stats['total_conversations_before_filter'] = total
        self.conversation_stats['conversations_after_filter'] = kept
        self.conversation_stats['filter_retention_rate'] = (kept / total * 100) if total > 0 else 0