import os
import pytest
from fastapi.testclient import TestClient
from dotenv import load_dotenv
from app import app

load_dotenv()

client = TestClient(app)

class TestConfig:
    BASE_URL = "http://localhost:8000"
    TEST_DIR = os.path.abspath("/mnt/private/personal/task/dataset/test")
    TEST_FILENAME = "2501053670.jpg"

@pytest.fixture(scope="module")
def test_directory():
    if not os.path.exists(TestConfig.TEST_DIR):
        pytest.skip(f"Test directory not found: {TestConfig.TEST_DIR}")
    return TestConfig.TEST_DIR

def test_process_directory(test_directory):
    config = {
        "directory_path": test_directory,
        "max_workers": 2,
        "batch_size": 5
    }
    
    try:
        response = client.post(
            "/process-directory",
            json=config,
            headers={"Content-Type": "application/json"}
        )
        
        if not os.path.exists(test_directory):
            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()
        else:
            assert response.status_code == 200
            result = response.json()
            assert "message" in result
            assert "config" in result
            assert result["config"]["max_workers"] == 2
            assert result["config"]["batch_size"] == 5
    except Exception as e:
        pytest.fail(f"Test failed: {str(e)}")

def test_get_summary():
    try:
        response = client.get(f"/summary/{TestConfig.TEST_FILENAME}")
        
        if response.status_code == 200:
            result = response.json()
            assert "filename" in result
            assert "category" in result
            assert "summary" in result
        else:
            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()
    except Exception as e:
        pytest.fail(f"Test failed: {str(e)}")

def test_semantic_search():
    try:
        response = client.get(
            "/search/semantic",
            params={"query": "test document", "k": 3}
        )
        
        assert response.status_code == 200
        results = response.json()
        assert isinstance(results, list)
        
        response = client.get(
            "/search/semantic",
            params={"query": "", "k": 3}
        )
        assert response.status_code == 422
        
        response = client.get(
            "/search/semantic",
            params={"query": "test", "k": 0}
        )
        assert response.status_code == 422
    except Exception as e:
        pytest.fail(f"Test failed: {str(e)}")

def test_keyword_search():
    try:
        response = client.get(
            "/search/keyword",
            params={"keyword": "report"}
        )
        
        assert response.status_code == 200
        results = response.json()
        assert isinstance(results, list)
        
        response = client.get(
            "/search/keyword",
            params={"keyword": ""}
        )
        assert response.status_code == 422
    except Exception as e:
        pytest.fail(f"Test failed: {str(e)}")

def test_error_handling():
    try:
        config = {
            "directory_path": "/non/existent/path",
            "max_workers": 2,
            "batch_size": 5
        }
        response = client.post(
            "/process-directory",
            json=config,
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code != 202
        assert "not found" in response.json()["detail"].lower()

        response = client.get("/summary/nonexistent.jpg")
        assert response.status_code != 200
        assert "not found" in response.json()["detail"].lower()

        response = client.get("/search/semantic")
        assert response.status_code != 200
    except Exception as e:
        pytest.fail(f"Test failed: {str(e)}")

if __name__ == "__main__":
    pytest.main(["-v", "test_api.py"])