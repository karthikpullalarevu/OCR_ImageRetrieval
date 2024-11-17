from fastapi import FastAPI, HTTPException, Query, Body
from typing import List, Optional
from pydantic import BaseModel, Field
import os
from concurrent_processor import ConcurrentDocumentProcessor
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Concurrent Document Processing API")

processor = ConcurrentDocumentProcessor(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    postgres_conn_str=os.getenv("POSTGRES_CONN_STRING"),
    clean_db=True
)

class ProcessingConfig(BaseModel):
    directory_path: str = Field(..., description="Path to the directory containing images")
    max_workers: int = Field(4, gt=0, description="Number of concurrent workers")
    batch_size: int = Field(10, gt=0, description="Size of processing batches")

class DocumentSummary(BaseModel):
    filename: str
    category: str
    summary: str

class SearchResult(BaseModel):
    filename: str
    category: str
    summary: str
    similarity_score: Optional[float] = None

@app.post("/process-directory")
async def process_directory(config: ProcessingConfig):
    try:
        if not os.path.exists(config.directory_path):
            raise HTTPException(
                status_code=404,
                detail=f"Directory not found: {config.directory_path}"
            )
        
        if not os.path.isdir(config.directory_path):
            raise HTTPException(
                status_code=400,
                detail=f"Path is not a directory: {config.directory_path}"
            )

        processor.max_workers = config.max_workers
        processor.batch_size = config.batch_size

        await processor.process_directory_concurrent(config.directory_path)
        
        return {
            "message": f"Successfully processed directory: {config.directory_path}",
            "config": {
                "max_workers": config.max_workers,
                "batch_size": config.batch_size
            }
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except NotADirectoryError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/summary/{filename}", response_model=DocumentSummary)
async def get_summary(filename: str):
    try:
        result = processor.get_summary_by_filename(filename)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {filename}"
            )
        return result
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Document not found: {filename}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/semantic", response_model=List[SearchResult])
async def semantic_search(
    query: str = Query(..., min_length=1),
    k: int = Query(default=5, gt=0, le=20)
):
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Search query cannot be empty")
        
        results = processor.semantic_search(query, k)
        return results
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/keyword", response_model=List[SearchResult])
async def keyword_search(
    keyword: str = Query(..., min_length=1)
):
    try:
        if not keyword.strip():
            raise HTTPException(status_code=400, detail="Keyword cannot be empty")
            
        results = processor.keyword_search(keyword)
        return results
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"status": "healthy", "message": "API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)