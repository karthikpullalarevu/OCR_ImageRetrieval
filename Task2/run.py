import os
import time
import asyncio
from dotenv import load_dotenv
from concurrent_processor import ConcurrentDocumentProcessor
import logging
from typing import Dict, Any
import psutil
import json
from datetime import datetime

# Configure logging with timestamp
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'processing_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

class ProcessingStats:
    def __init__(self):
        self.start_time = time.time()
        self.end_time = None
        self.total_files = 0
        self.processed_files = 0
        self.failed_files = 0
        self.processing_times = []
        self.memory_usage = []
        self.cpu_usage = []
        
    def update_resource_usage(self):
        """Update system resource usage statistics."""
        process = psutil.Process(os.getpid())
        self.memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB
        self.cpu_usage.append(psutil.cpu_percent(interval=1))
        
    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate and return processing statistics."""
        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        
        return {
            "total_time_seconds": round(total_time, 2),
            "files_processed": self.processed_files,
            "files_failed": self.failed_files,
            "average_time_per_file": round(total_time / max(self.processed_files, 1), 2),
            "success_rate": round((self.processed_files / max(self.total_files, 1)) * 100, 2),
            "peak_memory_usage_mb": round(max(self.memory_usage) if self.memory_usage else 0, 2),
            "average_memory_usage_mb": round(sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0, 2),
            "peak_cpu_usage_percent": round(max(self.cpu_usage) if self.cpu_usage else 0, 2),
            "average_cpu_usage_percent": round(sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0, 2)
        }

async def test_document_processing(
    test_folder: str,
    max_workers: int = 4,
    batch_size: int = 10,
    clean_db: bool = False
) -> Dict[str, Any]:
    """
    Test concurrent document processing functionality
    
    Args:
        test_folder (str): Path to the test folder containing images
        max_workers (int): Number of concurrent workers
        batch_size (int): Size of processing batches
        clean_db (bool): Whether to clean the database before processing
        
    Returns:
        Dict[str, Any]: Processing statistics and results
    """
    try:
        # Load environment variables
        load_dotenv()
        
        # Get credentials from environment variables
        openai_api_key = os.getenv("OPENAI_API_KEY")
        postgres_conn_str = os.getenv("POSTGRES_CONN_STRING")
        
        if not all([openai_api_key, postgres_conn_str]):
            raise ValueError("Missing required environment variables. Please check .env file.")
        
        # Initialize stats
        stats = ProcessingStats()
        
        # Initialize processor
        logger.info(f"Initializing Concurrent Document Processor with {max_workers} workers and batch size {batch_size}")
        processor = ConcurrentDocumentProcessor(
            openai_api_key=openai_api_key,
            postgres_conn_str=postgres_conn_str,
            clean_db=clean_db,
            max_workers=max_workers,
            batch_size=batch_size
        )
        
        # Count total files
        for root, _, files in os.walk(test_folder):
            stats.total_files += sum(1 for f in files 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')))
        
        logger.info(f"Found {stats.total_files} images to process in {test_folder}")
        
        # Process directory with resource monitoring
        async def monitor_resources():
            while True:
                stats.update_resource_usage()
                await asyncio.sleep(5)  # Update every 5 seconds
        
        # Create monitoring task
        monitor_task = asyncio.create_task(monitor_resources())
        
        # Process directory
        try:
            await processor.process_directory_concurrent(test_folder)
        except Exception as e:
            logger.error(f"Error during directory processing: {str(e)}")
        finally:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
        
        # Test retrieval and search functionality
        logger.info("\nTesting document retrieval and search functionality:")
        
        # Test summary retrieval
        test_file = next(f for f in os.listdir(test_folder) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')))
        summary = processor.get_summary_by_filename(test_file)
        logger.info(f"\nTest file summary ({test_file}):")
        logger.info(f"Category: {summary['category']}")
        logger.info(f"Summary: {summary['summary'][:]}...")

        # Test semantic search
        test_query = "Show me documents containing technical information"
        semantic_results = processor.semantic_search(test_query, k=3)
        logger.info(f"\nSemantic search results for: '{test_query}'")
        for idx, result in enumerate(semantic_results, 1):
            logger.info(f"\n{idx}. File: {result['filename']}")
            logger.info(f"   Score: {result['similarity_score']:.4f}")
            logger.info(f"   Summary: {result['summary'][:]}...")

        # Test keyword search
        test_keyword = "report"
        keyword_results = processor.keyword_search(test_keyword)
        logger.info(f"\nKeyword search results for: '{test_keyword}'")
        for idx, result in enumerate(keyword_results, 1):
            logger.info(f"\n{idx}. File: {result['filename']}")
            logger.info(f"   Category: {result['category']}")
            logger.info(f"   Summary: {result['summary'][:]}...")
        
        # Calculate final statistics
        final_stats = stats.calculate_statistics()
        
        # Save statistics to file
        stats_filename = f'processing_stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(stats_filename, 'w') as f:
            json.dump(final_stats, f, indent=4)
        
        logger.info("\nProcessing Statistics:")
        for key, value in final_stats.items():
            logger.info(f"{key}: {value}")
        
        return final_stats
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

async def main():
    # Configuration
    TEST_FOLDER = "/mnt/private/personal/task/dataset/test"  # Replace with your actual test folder path
    MAX_WORKERS = 4  # Adjust based on your system's capabilities
    BATCH_SIZE = 30  # Adjust based on your needs
    CLEAN_DB = False  # Set to True to start with a clean database
    
    # Run the test
    stats = await test_document_processing(
        test_folder=TEST_FOLDER,
        max_workers=MAX_WORKERS,
        batch_size=BATCH_SIZE,
        clean_db=CLEAN_DB
    )
    
    return stats

if __name__ == "__main__":
    asyncio.run(main())