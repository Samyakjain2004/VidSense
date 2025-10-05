import os
import json
import base64
import http.client
import urllib.parse
import requests
import argparse
import time
from pathlib import Path
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswParameters, HnswAlgorithmConfiguration, SimpleField, SearchField,
    SearchFieldDataType, SearchIndex, VectorSearch, VectorSearchAlgorithmKind,
    VectorSearchAlgorithmMetric, VectorSearchProfile
)
from azure.core.exceptions import ResourceNotFoundError
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('frames.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

# Environment variables
SERVICE_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT1", "")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME2", "frames_index")
SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY1", "").strip()
AI_VISION_KEY = os.getenv("AZURE_AI_VISION_API_KEY", "")
AI_VISION_REGION = os.getenv("AZURE_AI_VISION_REGION", "")
AI_VISION_ENDPOINT = os.getenv("AZURE_AI_VISION_ENDPOINT", "")
DEFAULT_IMAGES_DIR = Path(os.getenv("IMAGES_DIR", "frames"))
DEFAULT_OUTPUT_JSON = Path(os.getenv("OUTPUT_JSON", "output/frameVectors.json"))

# Constants
WINDOW_SIZE = 20  # 10 sec * 2 fps
RATE_LIMIT_DELAY = 0.1  # Delay to ensure < 10 calls/sec (100ms per call)

def _is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}

def _collect_images(root: Path):
    if root.is_file() and _is_image_file(root):
        yield root
        return
    for p in root.rglob("*"):
        if _is_image_file(p):
            yield p

def _normalize_region(r: str) -> str:
    return r.strip().lower().replace(" ", "") if r else r

def _safe_key(s: str) -> str:
    try:
        return base64.urlsafe_b64encode(str(s).encode("utf-8")).decode("ascii")
    except Exception:
        import re
        return re.sub(r"[^A-Za-z0-9_\-=]", "-", str(s))

@retry(
    stop=stop_after_attempt(5),  # Increased to 5 retries
    wait=wait_fixed(5),  # Wait 5 seconds between retries
    retry=retry_if_exception_type((http.client.HTTPException, requests.exceptions.RequestException, TimeoutError))
)
def get_image_vector(image_path, key, region, endpoint=None):
    headers = {'Ocp-Apim-Subscription-Key': key}
    params = urllib.parse.urlencode({'model-version': '2023-04-15'})

    if endpoint and str(endpoint).strip():
        parsed = urllib.parse.urlparse(str(endpoint).strip())
        host = parsed.netloc
        if not host:
            raise Exception(f"Invalid AZURE_AI_VISION_ENDPOINT: {endpoint}")
    else:
        host = f"{region}.api.cognitive.microsoft.com"

    if isinstance(image_path, (str, os.PathLike)) and str(image_path).startswith(("http://", "https://")):
        headers['Content-Type'] = 'application/json'
        body = json.dumps({"url": str(image_path)})
    else:
        headers['Content-Type'] = 'application/octet-stream'
        with open(image_path, "rb") as f:
            body = f.read()

    conn = http.client.HTTPSConnection(host, timeout=30)  # Increased timeout to 30 seconds
    url_path = f"/computervision/retrieval:vectorizeImage?api-version=2024-02-01&{params}"
    try:
        conn.request("POST", url_path, body, headers)
        resp = conn.getresponse()
        raw = resp.read()
        try:
            data = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception as e:
            logger.error(f"Failed to parse response for {image_path}: {e}")
            data = {"raw": raw[:200].decode("utf-8", errors="replace") if raw else ""}
        finally:
            conn.close()

        if resp.status != 200:
            msg = (isinstance(data, dict) and (data.get("message") or data.get("error") or data.get("code") or data.get("raw"))) or ""
            raise Exception(f"Vision API {resp.status} {resp.reason}: {msg}")

        vec = data.get("vector")
        if not isinstance(vec, list):
            raise Exception(f"No 'vector' in response for {image_path}")
        logger.info(f"Successfully embedded {image_path}")
        return vec
    except Exception as e:
        logger.error(f"Error embedding {image_path}: {e}")
        raise
    finally:
        time.sleep(RATE_LIMIT_DELAY)  # Enforce rate limit

def embed_folder(images_dir: Path = DEFAULT_IMAGES_DIR, output_json: Path = DEFAULT_OUTPUT_JSON, video_id: str = None):
    if not AI_VISION_KEY or not AI_VISION_REGION:
        raise RuntimeError("Missing AZURE_AI_VISION_API_KEY or AZURE_AI_VISION_REGION")

    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")

    images = sorted(list(_collect_images(images_dir)), key=lambda p: int(p.stem.split('-')[-1]))
    if not images:
        logger.warning("No images found in directory")
        return {"count": 0, "docs_path": str(output_json), "docs": []}

    if video_id is None:
        video_id = images[0].stem.split('-')[0]  # Extract from first frame
    logger.info(f"Processing images for video_id: {video_id}")

    region = _normalize_region(AI_VISION_REGION)
    docs = []
    for win_idx in range(0, len(images), WINDOW_SIZE):
        window_frames = images[win_idx:win_idx + WINDOW_SIZE]
        window_vecs = []
        for img in window_frames:
            try:
                vec = get_image_vector(str(img), AI_VISION_KEY, region, AI_VISION_ENDPOINT)
                window_vecs.append(vec)
            except Exception as e:
                logger.error(f"Failed to embed image {img}: {e}")
                continue
        
        if not window_vecs:
            logger.warning(f"No valid embeddings for window {win_idx // WINDOW_SIZE}")
            continue

        avg_vec = np.mean(window_vecs, axis=0).tolist()
        start_time = win_idx / 2.0  # 2 fps
        end_time = start_time + 10.0
        rel_paths = [str(img.relative_to(images_dir)).replace("\\", "/") for img in window_frames]
        
        docs.append({
            "id": f"win_{win_idx // WINDOW_SIZE}",
            "description": f"10-sec window from {start_time}s to {end_time}s",
            "image_vector": avg_vec,
            "video_id": video_id,
            "start_time": start_time,
            "end_time": end_time,
            "prev_win_id": f"win_{(win_idx // WINDOW_SIZE) - 1}" if win_idx > 0 else None,
            "next_win_id": f"win_{(win_idx // WINDOW_SIZE) + 1}" if win_idx + WINDOW_SIZE < len(images) else None,
            "frame_paths": rel_paths
        })
        logger.info(f"Processed window {win_idx // WINDOW_SIZE} ({start_time}s - {end_time}s)")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)
    logger.info(f"Saved {len(docs)} documents to {output_json}")

    return {"count": len(docs), "docs_path": str(output_json), "docs": docs}

def create_or_update_index(index_name: str = INDEX_NAME):
    cred = AzureKeyCredential(SEARCH_ADMIN_KEY)
    ic = SearchIndexClient(endpoint=SERVICE_ENDPOINT, credential=cred)

    try:
        ic.get_index(index_name)
        ic.delete_index(index_name)
        logger.info(f"Deleted existing index '{index_name}'.")
    except ResourceNotFoundError:
        logger.info(f"Index '{index_name}' does not exist. Proceeding to create.")
    except Exception as e:
        logger.error(f"Error deleting index: {e}")
        raise
    
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchField(name="description", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
        SearchField(
            name="image_vector",
            hidden=False,
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1024,
            vector_search_profile_name="myHnswProfile",
        ),
        SearchField(name="video_id", type=SearchFieldDataType.String, filterable=True, sortable=True),
        SearchField(name="start_time", type=SearchFieldDataType.Double, filterable=True, sortable=True),
        SearchField(name="end_time", type=SearchFieldDataType.Double, filterable=True, sortable=True),
        SearchField(name="prev_win_id", type=SearchFieldDataType.String, filterable=True),
        SearchField(name="next_win_id", type=SearchFieldDataType.String, filterable=True),
        SearchField(name="frame_paths", type=SearchFieldDataType.Collection(SearchFieldDataType.String), searchable=True),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="myHnsw",
                kind=VectorSearchAlgorithmKind.HNSW,
                parameters=HnswParameters(
                    m=4, ef_construction=400, ef_search=1000, metric=VectorSearchAlgorithmMetric.COSINE
                ),
            ),
        ],
        profiles=[VectorSearchProfile(name="myHnswProfile", algorithm_configuration_name="myHnsw")],
    )

    idx = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)
    ic.create_or_update_index(idx)
    logger.info(f"Index '{index_name}' created or updated.")
    return {"index": index_name, "status": "created_or_updated"}

def upload_docs(output_json: Path = DEFAULT_OUTPUT_JSON, index_name: str = INDEX_NAME):
    with open(output_json, "r", encoding="utf-8") as f:
        docs = json.load(f)

    for d in docs:
        if isinstance(d, dict) and "id" in d:
            d["id"] = _safe_key(d["id"])

    cred = AzureKeyCredential(SEARCH_ADMIN_KEY)
    sc = SearchClient(endpoint=SERVICE_ENDPOINT, index_name=index_name, credential=cred)
    sc.upload_documents(docs)
    logger.info(f"Uploaded {len(docs)} documents to index '{index_name}'")
    return {"uploaded": len(docs)}

def frames(video_id: str = None):
    images_dir = DEFAULT_IMAGES_DIR
    output_json = DEFAULT_OUTPUT_JSON
    index_name = INDEX_NAME

    result = {
        "step": "frames",
        "status": "failed",
        "error": None,
        "embed": None,
        "index": None,
        "upload": None
    }

    try:
        if not all([AI_VISION_KEY, AI_VISION_REGION, SERVICE_ENDPOINT, SEARCH_ADMIN_KEY]):
            raise ValueError("Missing required environment variables: AZURE_AI_VISION_API_KEY, AZURE_AI_VISION_REGION, AZURE_SEARCH_ENDPOINT1, or AZURE_SEARCH_ADMIN_KEY1")

        # Embed all images
        res = embed_folder(images_dir=images_dir, output_json=output_json, video_id=video_id)
        result["embed"] = res
        logger.info(json.dumps({"step": "embed", **res}, indent=2))

        # Create or update index
        res = create_or_update_index(index_name=index_name)
        result["index"] = res
        logger.info(json.dumps({"step": "index", **res}, indent=2))

        # Upload JSON docs
        res = upload_docs(output_json=output_json, index_name=index_name)
        result["upload"] = res
        logger.info(json.dumps({"step": "upload", **res}, indent=2))

        result["status"] = "success"
    except Exception as e:
        result["error"] = str(e)
        logger.error(json.dumps(result, indent=2))

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process frames for CV embeddings")
    parser.add_argument("--video-id", help="Video ID to process")
    args = parser.parse_args()
    frames(video_id=args.video_id)