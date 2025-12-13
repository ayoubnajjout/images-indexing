"""
FastAPI backend for YOLO object detection and visual descriptor extraction.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from PIL import Image
import io
import base64
from typing import List, Optional
import os

from detection import YOLODetector
from descriptors import DescriptorExtractor

app = FastAPI(
    title="Image Indexing API",
    description="API for object detection and visual descriptor extraction",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:5000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector and descriptor extractor
detector = YOLODetector()
extractor = DescriptorExtractor()


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Load image from bytes to numpy array (BGR format)."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    return img


def load_image_from_base64(base64_str: str) -> np.ndarray:
    """Load image from base64 string to numpy array (BGR format)."""
    # Remove data URL prefix if present
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    image_bytes = base64.b64decode(base64_str)
    return load_image_from_bytes(image_bytes)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Image Indexing API is running"}


@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    """
    Detect objects in an uploaded image using YOLOv8.
    
    Returns bounding boxes, labels, and confidence scores.
    """
    try:
        # Read image
        contents = await file.read()
        image = load_image_from_bytes(contents)
        
        # Run detection
        detections = detector.detect(image)
        
        return {
            "success": True,
            "detections": detections,
            "count": len(detections)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/base64")
async def detect_objects_base64(data: dict):
    """
    Detect objects in a base64-encoded image using YOLOv8.
    
    Expects: { "image": "base64_encoded_string" }
    """
    try:
        if "image" not in data:
            raise HTTPException(status_code=400, detail="Missing 'image' field")
        
        image = load_image_from_base64(data["image"])
        detections = detector.detect(image)
        
        return {
            "success": True,
            "detections": detections,
            "count": len(detections)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/url")
async def detect_objects_url(data: dict):
    """
    Detect objects in an image from URL.
    
    Expects: { "url": "image_url" }
    """
    try:
        import requests
        
        if "url" not in data:
            raise HTTPException(status_code=400, detail="Missing 'url' field")
        
        response = requests.get(data["url"], timeout=10)
        response.raise_for_status()
        
        image = load_image_from_bytes(response.content)
        detections = detector.detect(image)
        
        return {
            "success": True,
            "detections": detections,
            "count": len(detections)
        }
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/descriptors")
async def extract_descriptors(file: UploadFile = File(...), bbox: Optional[str] = None):
    """
    Extract visual descriptors from an uploaded image or a region of it.
    
    Optional bbox parameter: "x1,y1,x2,y2" to extract from a specific region.
    """
    try:
        contents = await file.read()
        image = load_image_from_bytes(contents)
        
        # If bbox provided, crop to that region
        if bbox:
            coords = [int(x) for x in bbox.split(',')]
            x1, y1, x2, y2 = coords
            image = image[y1:y2, x1:x2]
        
        # Extract descriptors
        descriptors = extractor.extract_all(image)
        
        return {
            "success": True,
            "descriptors": descriptors
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/descriptors/base64")
async def extract_descriptors_base64(data: dict):
    """
    Extract visual descriptors from a base64-encoded image.
    
    Expects: { "image": "base64_string", "bbox": [x1, y1, x2, y2] (optional) }
    """
    try:
        if "image" not in data:
            raise HTTPException(status_code=400, detail="Missing 'image' field")
        
        image = load_image_from_base64(data["image"])
        
        # If bbox provided, crop to that region
        if "bbox" in data and data["bbox"]:
            x1, y1, x2, y2 = data["bbox"]
            image = image[int(y1):int(y2), int(x1):int(x2)]
        
        descriptors = extractor.extract_all(image)
        
        return {
            "success": True,
            "descriptors": descriptors
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/descriptors/url")
async def extract_descriptors_url(data: dict):
    """
    Extract visual descriptors from an image URL.
    
    Expects: { "url": "image_url", "bbox": [x1, y1, x2, y2] (optional) }
    """
    try:
        import requests
        
        if "url" not in data:
            raise HTTPException(status_code=400, detail="Missing 'url' field")
        
        response = requests.get(data["url"], timeout=10)
        response.raise_for_status()
        
        image = load_image_from_bytes(response.content)
        
        if "bbox" in data and data["bbox"]:
            x1, y1, x2, y2 = data["bbox"]
            image = image[int(y1):int(y2), int(x1):int(x2)]
        
        descriptors = extractor.extract_all(image)
        
        return {
            "success": True,
            "descriptors": descriptors
        }
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect-and-describe")
async def detect_and_describe(file: UploadFile = File(...)):
    """
    Detect objects and extract descriptors for each detected object.
    
    Returns detections with their visual descriptors.
    """
    try:
        contents = await file.read()
        image = load_image_from_bytes(contents)
        
        # Run detection
        detections = detector.detect(image)
        
        # Extract descriptors for each detection
        results = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            # Ensure valid crop coordinates
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(image.shape[1], int(x2)), min(image.shape[0], int(y2))
            
            if x2 > x1 and y2 > y1:
                cropped = image[y1:y2, x1:x2]
                descriptors = extractor.extract_all(cropped)
            else:
                descriptors = None
            
            results.append({
                **det,
                "descriptors": descriptors
            })
        
        return {
            "success": True,
            "detections": results,
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect-and-describe/url")
async def detect_and_describe_url(data: dict):
    """
    Detect objects and extract descriptors from an image URL.
    """
    try:
        import requests
        
        if "url" not in data:
            raise HTTPException(status_code=400, detail="Missing 'url' field")
        
        response = requests.get(data["url"], timeout=10)
        response.raise_for_status()
        
        image = load_image_from_bytes(response.content)
        
        # Run detection
        detections = detector.detect(image)
        
        # Extract descriptors for each detection
        results = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(image.shape[1], int(x2)), min(image.shape[0], int(y2))
            
            if x2 > x1 and y2 > y1:
                cropped = image[y1:y2, x1:x2]
                descriptors = extractor.extract_all(cropped)
            else:
                descriptors = None
            
            results.append({
                **det,
                "descriptors": descriptors
            })
        
        return {
            "success": True,
            "detections": results,
            "count": len(results)
        }
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/similarity")
async def compute_similarity(data: dict):
    """
    Compute similarity between two sets of descriptors.
    
    Expects: { 
        "descriptors1": {...}, 
        "descriptors2": {...},
        "weights": { "color": true, "texture": true, "shape": true }  // optional
    }
    """
    try:
        if "descriptors1" not in data or "descriptors2" not in data:
            raise HTTPException(status_code=400, detail="Missing descriptor fields")
        
        weights_config = data.get("weights", None)
        
        similarity_result = extractor.compute_similarity_detailed(
            data["descriptors1"],
            data["descriptors2"],
            weights_config
        )
        
        return {
            "success": True,
            "similarity": similarity_result['total'],
            "details": similarity_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search_similar(data: dict):
    """
    Search for similar objects in a database of descriptors.
    
    Expects: {
        "query_descriptors": {...},
        "database": [{ "id": "...", "descriptors": {...} }, ...],
        "top_k": 10,
        "weights": { "color": true, "texture": true, "shape": true }  // optional
    }
    
    Returns detailed similarity scores for each result.
    """
    try:
        if "query_descriptors" not in data or "database" not in data:
            raise HTTPException(status_code=400, detail="Missing required fields")
        
        query = data["query_descriptors"]
        database = data["database"]
        top_k = data.get("top_k", 10)
        weights_config = data.get("weights", None)
        
        # Compute similarities with detailed breakdown
        results = []
        for item in database:
            if "descriptors" in item and item["descriptors"]:
                sim_result = extractor.compute_similarity_detailed(query, item["descriptors"], weights_config)
                results.append({
                    "id": item.get("id"),
                    "imageId": item.get("imageId"),
                    "objectId": item.get("objectId"),
                    "label": item.get("label"),
                    "similarity": sim_result['total'],
                    "scores": {
                        "color": sim_result.get('color'),
                        "texture": sim_result.get('texture'),
                        "shape": sim_result.get('shape')
                    }
                })
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        return {
            "success": True,
            "results": results[:top_k]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare-images")
async def compare_images(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """
    Compare two images and compute their similarity.
    
    Returns similarity score and detailed breakdown.
    """
    try:
        # Read images
        contents1 = await file1.read()
        contents2 = await file2.read()
        
        image1 = load_image_from_bytes(contents1)
        image2 = load_image_from_bytes(contents2)
        
        # Extract descriptors
        desc1 = extractor.extract_all(image1)
        desc2 = extractor.extract_all(image2)
        
        # Compute similarity
        similarity_result = extractor.compute_similarity_detailed(desc1, desc2)
        
        return {
            "success": True,
            "file1": file1.filename,
            "file2": file2.filename,
            "similarity": similarity_result['total'],
            "details": similarity_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare-images/base64")
async def compare_images_base64(data: dict):
    """
    Compare two base64-encoded images.
    
    Expects: { "image1": "base64_string", "image2": "base64_string" }
    """
    try:
        if "image1" not in data or "image2" not in data:
            raise HTTPException(status_code=400, detail="Missing 'image1' or 'image2' field")
        
        image1 = load_image_from_base64(data["image1"])
        image2 = load_image_from_base64(data["image2"])
        
        desc1 = extractor.extract_all(image1)
        desc2 = extractor.extract_all(image2)
        
        similarity_result = extractor.compute_similarity_detailed(desc1, desc2)
        
        return {
            "success": True,
            "similarity": similarity_result['total'],
            "details": similarity_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info")
async def get_model_info():
    """
    Get information about the loaded detection model.
    
    Returns model path and supported classes.
    """
    from detection import IMAGENET_CLASSES, IMAGENET_READABLE_NAMES
    
    return {
        "success": True,
        "model_path": detector.model_path,
        "num_classes": len(IMAGENET_CLASSES),
        "classes": [
            {
                "id": k,
                "synset": v,
                "name": IMAGENET_READABLE_NAMES.get(k, v)
            }
            for k, v in IMAGENET_CLASSES.items()
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
