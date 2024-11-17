import os
import base64
from typing import List, Dict, Any
import psycopg2
from psycopg2.extras import Json
import faiss
import numpy as np
from openai import OpenAI
import logging
from PIL import Image
from io import BytesIO
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def clean_database(self):
        """Clean the database and reset the FAISS index."""
        try:
            with psycopg2.connect(self.postgres_conn_str) as conn:
                with conn.cursor() as cur:
                    cur.execute("TRUNCATE TABLE documents RESTART IDENTITY")
                    conn.commit()
            self.vector_index = faiss.IndexFlatL2(3072)
            logger.info("Successfully cleaned database and reset FAISS index")
        except Exception as e:
            logger.error(f"Error cleaning database: {str(e)}")
            raise

    def __init__(self, openai_api_key: str, postgres_conn_str: str, clean_db: bool = False):
        """
        Initialize the document processor with necessary credentials and connections.
        
        Args:
            openai_api_key (str): OpenAI API key
            postgres_conn_str (str): PostgreSQL connection string
            clean_db (bool): Whether to clean the database on initialization
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.postgres_conn_str = postgres_conn_str
        self.vector_index = None
        
        if clean_db:
            self.clean_database()
        
        self.setup_database()
        self.initialize_faiss()

    def setup_database(self):
        """Create necessary database tables if they don't exist."""
        with psycopg2.connect(self.postgres_conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id SERIAL PRIMARY KEY,
                        filename VARCHAR(255) UNIQUE,
                        category VARCHAR(100),
                        summary TEXT,
                        embedding BYTEA,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
                logger.info("Database tables created successfully")

    def initialize_faiss(self):
        """Initialize FAISS index for vector similarity search."""
        self.vector_index = faiss.IndexFlatL2(3072)
        
        try:
            with psycopg2.connect(self.postgres_conn_str) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT embedding FROM documents WHERE embedding IS NOT NULL LIMIT 1")
                    first_embedding = cur.fetchone()
                    
                    if first_embedding:
                        existing_dim = len(np.frombuffer(first_embedding[0], dtype=np.float32))
                        
                        if existing_dim != 3072:
                            logger.warning(f"Found embeddings with dimension {existing_dim}. Need to migrate to 3072 dimensions.")
                            cur.execute("SELECT id, summary FROM documents")
                            documents = cur.fetchall()
                            
                            for doc_id, summary in documents:
                                if summary:
                                    new_embedding = self.get_embedding(summary)
                                    cur.execute("""
                                        UPDATE documents 
                                        SET embedding = %s 
                                        WHERE id = %s
                                    """, (new_embedding.tobytes(), doc_id))
                            
                            conn.commit()
                            logger.info("Successfully migrated embeddings to new dimension")
                    
                    cur.execute("SELECT id, embedding FROM documents WHERE embedding IS NOT NULL")
                    results = cur.fetchall()
                    
                    if results:
                        embeddings = [np.frombuffer(r[1], dtype=np.float32) for r in results]
                        self.vector_index.add(np.vstack(embeddings))
                        logger.info(f"Loaded {len(results)} embeddings into FAISS index")
                        
        except Exception as e:
            logger.error(f"Error initializing FAISS index: {str(e)}")
            self.vector_index = faiss.IndexFlatL2(3072)

    def encode_image_to_base64(self, image_path: str) -> str:
        """
        Encode image to base64 string.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: Base64 encoded string of the image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def get_image_summary(self, image_path: str) -> str:
        """
        Get summary of an image using OpenAI's Vision model.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: Summary of the image
        """
        try:
            with open(image_path, "rb") as image_file:
                image = Image.open(image_file)
                max_size = (2000, 2000)
                if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                    image.thumbnail(max_size, Image.Resampling.LANCZOS)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                buffered = BytesIO()
                image.save(buffered, format="JPEG", quality=85)
                image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Please provide a detailed summary of this document image. Focus on the main content, purpose, and key information visible in the document. Include any relevant details about the structure, formatting, and visual elements present."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_data}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=1000
                )
                
                return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error getting image summary: {str(e)}")
            raise

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text using OpenAI's embedding model.
        
        Args:
            text (str): Text to get embedding for
            
        Returns:
            np.ndarray: Embedding vector
        """
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-large",
                input=text,
                dimensions=3072
            )
            return np.array(response.data[0].embedding, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            raise

    def process_document(self, image_path: str, category: str):
        """
        Process a single document: generate summary, embedding, and store in databases.
        
        Args:
            image_path (str): Path to the image file
            category (str): Document category
        """
        try:
            filename = os.path.basename(image_path)
            summary = self.get_image_summary(image_path)
            embedding = self.get_embedding(summary)

            with psycopg2.connect(self.postgres_conn_str) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO documents (filename, category, summary, embedding)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (filename) 
                        DO UPDATE SET 
                            category = EXCLUDED.category,
                            summary = EXCLUDED.summary,
                            embedding = EXCLUDED.embedding
                    """, (filename, category, summary, embedding.tobytes()))
                conn.commit()

            self.vector_index.add(embedding.reshape(1, -1))
            logger.info(f"Successfully processed document: {filename}")
            
            return {
                "filename": filename,
                "category": category,
                "summary": summary
            }
        except Exception as e:
            logger.error(f"Error processing document {image_path}: {str(e)}")
            raise

    def process_directory(self, directory_path: str):
        """
        Process all images in a directory and its subdirectories.
        
        Args:
            directory_path (str): Path to the directory containing images
        """
        processed_count = 0
        total_files = sum([len(files) for _, _, files in os.walk(directory_path) 
                          if any(f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')) 
                          for f in files)])
        
        logger.info(f"Found {total_files} images to process")
        
        for root, _, files in os.walk(directory_path):
            category = os.path.basename(root)
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                    try:
                        image_path = os.path.join(root, file)
                        self.process_document(image_path, category)
                        processed_count += 1
                        logger.info(f"Progress: {processed_count}/{total_files} ({(processed_count/total_files)*100:.2f}%)")
                    except Exception as e:
                        logger.error(f"Error processing {file}: {str(e)}")
                        continue

    def semantic_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform semantic search using FAISS.
        
        Args:
            query (str): Search query
            k (int): Number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of matching documents
        """
        try:
            query_embedding = self.get_embedding(query)
            distances, indices = self.vector_index.search(query_embedding.reshape(1, -1), k)
            
            results = []
            with psycopg2.connect(self.postgres_conn_str) as conn:
                with conn.cursor() as cur:
                    for idx, distance in zip(indices[0], distances[0]):
                        idx_int = int(idx)
                        if idx_int < 0:
                            continue
                            
                        cur.execute("""
                            SELECT filename, category, summary 
                            FROM documents 
                            LIMIT 1 OFFSET %s
                        """, (idx_int,))
                        result = cur.fetchone()
                        if result:
                            results.append({
                                "filename": result[0],
                                "category": result[1],
                                "summary": result[2],
                                "similarity_score": float(1 / (1 + float(distance)))
                            })
            
            return results
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            raise

    def keyword_search(self, keyword: str) -> List[Dict[str, Any]]:
        """
        Perform keyword-based search using PostgreSQL.
        
        Args:
            keyword (str): Keyword to search for
            
        Returns:
            List[Dict[str, Any]]: List of matching documents
        """
        try:
            with psycopg2.connect(self.postgres_conn_str) as conn:
                with conn.cursor() as cur:
                    search_pattern = f'%{keyword}%'
                    cur.execute("""
                        SELECT filename, category, summary
                        FROM documents
                        WHERE 
                            summary ILIKE %s OR
                            filename ILIKE %s OR
                            category ILIKE %s
                        ORDER BY 
                            CASE 
                                WHEN summary ILIKE %s THEN 1
                                WHEN filename ILIKE %s THEN 2
                                WHEN category ILIKE %s THEN 3
                                ELSE 4
                            END
                    """, (search_pattern, search_pattern, search_pattern, 
                         search_pattern, search_pattern, search_pattern))
                    
                    results = []
                    for row in cur.fetchall():
                        results.append({
                            "filename": row[0],
                            "category": row[1],
                            "summary": row[2]
                        })
                    
                    return results
        except Exception as e:
            logger.error(f"Error in keyword search: {str(e)}")
            raise

    def get_summary_by_filename(self, filename: str) -> Dict[str, Any]:
        """
        Get document summary by filename.
        
        Args:
            filename (str): Name of the file
            
        Returns:
            Dict[str, Any]: Document information including summary
        """
        try:
            with psycopg2.connect(self.postgres_conn_str) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT filename, category, summary
                        FROM documents
                        WHERE filename = %s
                    """, (filename,))
                    
                    result = cur.fetchone()
                    if result:
                        return {
                            "filename": result[0],
                            "category": result[1],
                            "summary": result[2]
                        }
                    return None
        except Exception as e:
            logger.error(f"Error getting summary by filename: {str(e)}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the processed documents.
        
        Returns:
            Dict[str, Any]: Statistics including total documents, documents per category, etc.
        """
        try:
            with psycopg2.connect(self.postgres_conn_str) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM documents")
                    total_docs = cur.fetchone()[0]
                    
                    cur.execute("""
                        SELECT category, COUNT(*) 
                        FROM documents 
                        GROUP BY category 
                        ORDER BY COUNT(*) DESC
                    """)
                    categories = {row[0]: row[1] for row in cur.fetchall()}
                    
                    return {
                        "total_documents": total_docs,
                        "documents_per_category": categories,
                        "vector_dimension": 3072,
                        "embedding_model": "text-embedding-3-large",
                        "summary_model": "gpt-4"
                    }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            raise

class ConcurrentDocumentProcessor(DocumentProcessor):
    def __init__(self, openai_api_key: str, postgres_conn_str: str, clean_db: bool = False, 
                 max_workers: int = 4, batch_size: int = 10):
        """
        Initialize the concurrent document processor.
        
        Args:
            openai_api_key (str): OpenAI API key
            postgres_conn_str (str): PostgreSQL connection string
            clean_db (bool): Whether to clean the database on initialization
            max_workers (int): Maximum number of worker threads
            batch_size (int): Size of batches for processing
        """
        super().__init__(openai_api_key, postgres_conn_str, clean_db)
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def process_image_concurrent(self, image_path: str, category: str) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Process a single image concurrently: generate summary and embedding in parallel.
        
        Args:
            image_path (str): Path to the image file
            category (str): Document category
            
        Returns:
            Tuple[Dict[str, Any], np.ndarray]: Document info and embedding
        """
        try:
            filename = os.path.basename(image_path)
            loop = asyncio.get_event_loop()
            get_summary_task = partial(self.get_image_summary, image_path)
            summary = await loop.run_in_executor(self.executor, get_summary_task)
            embedding = await loop.run_in_executor(self.executor, self.get_embedding, summary)
            document_info = {
                "filename": filename,
                "category": category,
                "summary": summary
            }
            return document_info, embedding
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            raise

    async def process_batch(self, batch: List[Tuple[str, str]]) -> List[Tuple[Dict[str, Any], np.ndarray]]:
        """
        Process a batch of images concurrently.
        
        Args:
            batch: List of tuples containing (image_path, category)
            
        Returns:
            List[Tuple[Dict[str, Any], np.ndarray]]: Processed documents and embeddings
        """
        tasks = [self.process_image_concurrent(image_path, category) 
                for image_path, category in batch]
        return await asyncio.gather(*tasks)

    async def save_batch_results(self, results: List[Tuple[Dict[str, Any], np.ndarray]]):
        """
        Save batch processing results to database.
        
        Args:
            results: List of tuples containing (document_info, embedding)
        """
        try:
            with psycopg2.connect(self.postgres_conn_str) as conn:
                with conn.cursor() as cur:
                    for doc_info, embedding in results:
                        cur.execute("""
                            INSERT INTO documents (filename, category, summary, embedding)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (filename) 
                            DO UPDATE SET 
                                category = EXCLUDED.category,
                                summary = EXCLUDED.summary,
                                embedding = EXCLUDED.embedding
                        """, (doc_info["filename"], doc_info["category"], 
                              doc_info["summary"], embedding.tobytes()))
                conn.commit()
            embeddings = np.vstack([embedding for _, embedding in results])
            self.vector_index.add(embeddings)
        except Exception as e:
            logger.error(f"Error saving batch results: {str(e)}")
            raise

    async def process_directory_concurrent(self, directory_path: str):
        """
        Process all images in a directory concurrently.
        
        Args:
            directory_path (str): Path to the directory containing images
        """
        image_files = []
        for root, _, files in os.walk(directory_path):
            category = os.path.basename(root)
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                    image_path = os.path.join(root, file)
                    image_files.append((image_path, category))

        total_files = len(image_files)
        processed_count = 0
        logger.info(f"Found {total_files} images to process")

        for i in range(0, total_files, self.batch_size):
            batch = image_files[i:i + self.batch_size]
            try:
                results = await self.process_batch(batch)
                await self.save_batch_results(results)
                processed_count += len(batch)
                logger.info(f"Progress: {processed_count}/{total_files} "
                          f"({(processed_count/total_files)*100:.2f}%)")
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                continue

        logger.info("Directory processing completed")

    def process_directory(self, directory_path: str):
        """
        Override parent class method to use concurrent processing.
        """
        asyncio.run(self.process_directory_concurrent(directory_path))