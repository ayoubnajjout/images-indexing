"""
FastAPI backend for YOLO object detection, visual descriptor extraction,
and 3D model shape descriptor extraction using OpenGL.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from PIL import Image
import io
import base64
from typing import List, Optional, Dict
import os
import tempfile
import shutil
import glob

from detection import YOLODetector
from descriptors import DescriptorExtractor
from shape_descriptors_3d import (
    Shape3DDescriptorExtractor,
    OBJLoader,
    compute_similarity,
    euclidean_distance,
    chi_square_distance,
    generate_thumbnail,
    ThumbnailGenerator
)

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

# Initialize 3D shape descriptor extractor
extractor_3d = Shape3DDescriptorExtractor()

# Initialize thumbnail generator (lazy - will be created on first use)
thumbnail_generator: ThumbnailGenerator = None

# 3D models database path
MODELS_3D_PATH = os.environ.get('MODELS_3D_PATH', '/app/3d-data/3D Models')
THUMBNAILS_3D_PATH = os.environ.get('THUMBNAILS_3D_PATH', '/app/3d-data/Thumbnails')


def _find_thumbnail(thumbnails_base: str, category: str, obj_filename: str) -> Optional[str]:
    """Find thumbnail file with various extensions and case variations."""
    base_name = obj_filename.replace('.obj', '')
    
    # Try various extensions and case variations
    for ext in ['.jpg', '.png', '.JPG', '.PNG', '.jpeg', '.JPEG']:
        # Try exact match
        test_path = os.path.join(thumbnails_base, category, base_name + ext)
        if os.path.exists(test_path):
            return test_path
        
        # Try capitalized first letter
        cap_name = base_name[0].upper() + base_name[1:] if base_name else base_name
        test_path = os.path.join(thumbnails_base, category, cap_name + ext)
        if os.path.exists(test_path):
            return test_path
        
        # Try lowercase
        test_path = os.path.join(thumbnails_base, category, base_name.lower() + ext)
        if os.path.exists(test_path):
            return test_path
    
    return None


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
async def detect_objects_base64(request: Request):
    """
    Detect objects in a base64-encoded image using YOLOv8.
    
    Expects: { "image": "base64_encoded_string" }
    """
    try:
        data = await request.json()
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
async def detect_objects_url(request: Request):
    """
    Detect objects in an image from URL.
    
    Expects: { "url": "image_url" }
    """
    try:
        import requests
        
        data = await request.json()
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
async def extract_descriptors_base64(request: Request):
    """
    Extract visual descriptors from a base64-encoded image.
    
    Expects: { "image": "base64_string", "bbox": [x1, y1, x2, y2] (optional) }
    """
    try:
        data = await request.json()
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
async def extract_descriptors_url(request: Request):
    """
    Extract visual descriptors from an image URL.
    
    Expects: { "url": "image_url", "bbox": [x1, y1, x2, y2] (optional) }
    """
    try:
        import requests
        
        data = await request.json()
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
async def detect_and_describe_url(request: Request):
    """
    Detect objects and extract descriptors from an image URL.
    """
    try:
        import requests
        import traceback
        
        # Parse JSON body
        data = await request.json()
        print(f"Received data: {data}")
        
        if "url" not in data:
            raise HTTPException(status_code=400, detail="Missing 'url' field")
        
        url = data["url"]
        print(f"Fetching image from URL: {url}")
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            print(f"Successfully fetched image, size: {len(response.content)} bytes")
        except requests.RequestException as e:
            print(f"Failed to fetch image: {str(e)}")
            traceback.print_exc()
            raise HTTPException(status_code=400, detail=f"Failed to fetch image: {str(e)}")
        
        image = load_image_from_bytes(response.content)
        print(f"Loaded image, shape: {image.shape}")
        
        # Run detection
        detections = detector.detect(image)
        print(f"Detection complete, found {len(detections)} objects")
        
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
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/similarity")
async def compute_similarity_endpoint(request: Request):
    """
    Compute similarity between two sets of descriptors.
    
    Expects: { 
        "descriptors1": {...}, 
        "descriptors2": {...},
        "weights": { "color": true, "texture": true, "shape": true }  // optional
    }
    """
    try:
        data = await request.json()
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
async def search_similar(request: Request):
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
        data = await request.json()
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
async def compare_images_base64(request: Request):
    """
    Compare two base64-encoded images.
    
    Expects: { "image1": "base64_string", "image2": "base64_string" }
    """
    try:
        data = await request.json()
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


# ========================
# 3D Model Endpoints
# ========================

@app.get("/3d/categories")
async def get_3d_categories():
    """
    Get all available 3D model categories from the database.
    """
    try:
        categories = []
        if os.path.exists(MODELS_3D_PATH):
            for item in os.listdir(MODELS_3D_PATH):
                item_path = os.path.join(MODELS_3D_PATH, item)
                if os.path.isdir(item_path):
                    # Count models in category
                    model_count = len([f for f in os.listdir(item_path) 
                                      if f.endswith('.obj')])
                    categories.append({
                        "name": item,
                        "count": model_count
                    })
        
        categories.sort(key=lambda x: x['name'])
        
        return {
            "success": True,
            "categories": categories,
            "total": len(categories)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/3d/models")
async def get_3d_models(category: Optional[str] = None, limit: int = 50, offset: int = 0):
    """
    Get list of 3D models, optionally filtered by category.
    """
    try:
        models = []
        
        if category:
            search_path = os.path.join(MODELS_3D_PATH, category)
            if os.path.exists(search_path):
                for filename in os.listdir(search_path):
                    if filename.endswith('.obj'):
                        model_path = os.path.join(search_path, filename)
                        thumbnail_path = _find_thumbnail(THUMBNAILS_3D_PATH, category, filename)
                        
                        models.append({
                            "filename": filename,
                            "category": category,
                            "path": model_path,
                            "thumbnail": thumbnail_path
                        })
        else:
            # Get all models from all categories
            if os.path.exists(MODELS_3D_PATH):
                for cat in os.listdir(MODELS_3D_PATH):
                    cat_path = os.path.join(MODELS_3D_PATH, cat)
                    if os.path.isdir(cat_path):
                        for filename in os.listdir(cat_path):
                            if filename.endswith('.obj'):
                                model_path = os.path.join(cat_path, filename)
                                thumbnail_path = _find_thumbnail(THUMBNAILS_3D_PATH, cat, filename)
                                
                                models.append({
                                    "filename": filename,
                                    "category": cat,
                                    "path": model_path,
                                    "thumbnail": thumbnail_path
                                })
        
        total = len(models)
        models = models[offset:offset + limit]
        
        return {
            "success": True,
            "models": models,
            "total": total,
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/3d/descriptors")
async def extract_3d_descriptors(file: UploadFile = File(...)):
    """
    Extract shape descriptors from an uploaded .obj file.
    
    Returns all computed shape descriptors (D1, D2, D3, D4, A3, geometric features).
    """
    try:
        if not file.filename.endswith('.obj'):
            raise HTTPException(status_code=400, detail="File must be a .obj file")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.obj') as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Extract descriptors
            result = extractor_3d.extract(tmp_path)
            
            if result is None:
                raise HTTPException(status_code=400, detail="Failed to process OBJ file")
            
            result['filename'] = file.filename
            
            return {
                "success": True,
                "descriptors": result
            }
        finally:
            # Clean up temp file
            os.unlink(tmp_path)
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/3d/descriptors/path")
async def extract_3d_descriptors_from_path(request: Request):
    """
    Extract shape descriptors from a model file path.
    
    Expects: { "path": "/path/to/model.obj" }
    """
    try:
        data = await request.json()
        if "path" not in data:
            raise HTTPException(status_code=400, detail="Missing 'path' field")
        
        model_path = data["path"]
        
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model file not found")
        
        result = extractor_3d.extract(model_path)
        
        if result is None:
            raise HTTPException(status_code=400, detail="Failed to process OBJ file")
        
        return {
            "success": True,
            "descriptors": result
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/3d/search")
async def search_3d_models(file: UploadFile = File(...), 
                           limit: int = 10,
                           category: Optional[str] = None):
    """
    Search for similar 3D models by uploading a query .obj file.
    
    Returns the most similar models from the database ranked by similarity.
    """
    try:
        if not file.filename.endswith('.obj'):
            raise HTTPException(status_code=400, detail="File must be a .obj file")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.obj') as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Extract query descriptors
            query_desc = extractor_3d.extract(tmp_path)
            
            if query_desc is None:
                raise HTTPException(status_code=400, detail="Failed to process query OBJ file")
            
            # Search in database
            results = await _search_similar_models(query_desc, limit, category)
            
            return {
                "success": True,
                "query": {
                    "filename": file.filename,
                    "num_vertices": query_desc['num_vertices'],
                    "num_faces": query_desc['num_faces']
                },
                "results": results,
                "count": len(results)
            }
        finally:
            os.unlink(tmp_path)
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/3d/search/path")
async def search_3d_models_from_path(request: Request):
    """
    Search for similar 3D models using a model path from the database.
    
    Expects: { "path": "/path/to/model.obj", "limit": 10, "category": null }
    """
    try:
        data = await request.json()
        if "path" not in data:
            raise HTTPException(status_code=400, detail="Missing 'path' field")
        
        model_path = data["path"]
        limit = data.get("limit", 10)
        category = data.get("category", None)
        
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model file not found")
        
        # Extract query descriptors
        query_desc = extractor_3d.extract(model_path)
        
        if query_desc is None:
            raise HTTPException(status_code=400, detail="Failed to process OBJ file")
        
        # Search in database
        results = await _search_similar_models(query_desc, limit, category, exclude_path=model_path)
        
        return {
            "success": True,
            "query": {
                "path": model_path,
                "filename": os.path.basename(model_path),
                "num_vertices": query_desc['num_vertices'],
                "num_faces": query_desc['num_faces']
            },
            "results": results,
            "count": len(results)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _search_similar_models(query_desc: Dict, 
                                  limit: int = 10, 
                                  category: Optional[str] = None,
                                  exclude_path: Optional[str] = None) -> List[Dict]:
    """
    Internal function to search for similar models in the database.
    """
    similarities = []
    
    # Determine search paths
    if category:
        search_paths = [os.path.join(MODELS_3D_PATH, category)]
    else:
        search_paths = [os.path.join(MODELS_3D_PATH, cat) 
                       for cat in os.listdir(MODELS_3D_PATH)
                       if os.path.isdir(os.path.join(MODELS_3D_PATH, cat))]
    
    # Search through all models
    for search_path in search_paths:
        if not os.path.exists(search_path):
            continue
            
        cat_name = os.path.basename(search_path)
        
        for filename in os.listdir(search_path):
            if not filename.endswith('.obj'):
                continue
                
            model_path = os.path.join(search_path, filename)
            
            # Skip the query model itself
            if exclude_path and os.path.abspath(model_path) == os.path.abspath(exclude_path):
                continue
            
            try:
                # Extract descriptors for database model
                db_desc = extractor_3d.extract(model_path)
                
                if db_desc is None:
                    continue
                
                # Compute similarity
                similarity = compute_similarity(query_desc, db_desc)
                
                # Get thumbnail path
                thumbnail_path = _find_thumbnail(THUMBNAILS_3D_PATH, cat_name, filename)
                
                similarities.append({
                    "filename": filename,
                    "category": cat_name,
                    "path": model_path,
                    "thumbnail": thumbnail_path,
                    "similarity": float(similarity),
                    "num_vertices": db_desc['num_vertices'],
                    "num_faces": db_desc['num_faces']
                })
                
            except Exception as e:
                print(f"Error processing {model_path}: {e}")
                continue
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    
    return similarities[:limit]


@app.post("/3d/compare")
async def compare_3d_models(file1: UploadFile = File(...), 
                            file2: UploadFile = File(...)):
    """
    Compare two 3D models and compute their similarity.
    
    Returns detailed similarity scores for each descriptor type.
    """
    try:
        if not file1.filename.endswith('.obj') or not file2.filename.endswith('.obj'):
            raise HTTPException(status_code=400, detail="Both files must be .obj files")
        
        # Save uploaded files temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.obj') as tmp1:
            content1 = await file1.read()
            tmp1.write(content1)
            tmp1_path = tmp1.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.obj') as tmp2:
            content2 = await file2.read()
            tmp2.write(content2)
            tmp2_path = tmp2.name
        
        try:
            # Extract descriptors
            desc1 = extractor_3d.extract(tmp1_path)
            desc2 = extractor_3d.extract(tmp2_path)
            
            if desc1 is None or desc2 is None:
                raise HTTPException(status_code=400, detail="Failed to process one or both OBJ files")
            
            # Compute overall similarity
            similarity = compute_similarity(desc1, desc2)
            
            # Compute individual descriptor similarities
            descriptor_similarities = {}
            for desc_name in ['d1', 'd2', 'd3', 'd4', 'a3', 'bbox', 'moments', 'mesh_stats']:
                arr1 = np.array(desc1['descriptors'].get(desc_name, []))
                arr2 = np.array(desc2['descriptors'].get(desc_name, []))
                
                if len(arr1) > 0 and len(arr2) > 0 and len(arr1) == len(arr2):
                    if desc_name in ['d1', 'd2', 'd3', 'd4', 'a3']:
                        # Histogram intersection
                        sim = float(np.sum(np.minimum(arr1, arr2)))
                    else:
                        # Cosine similarity
                        norm1 = np.linalg.norm(arr1)
                        norm2 = np.linalg.norm(arr2)
                        if norm1 > 1e-10 and norm2 > 1e-10:
                            sim = float((np.dot(arr1, arr2) / (norm1 * norm2) + 1) / 2)
                        else:
                            sim = 0.0
                    
                    descriptor_similarities[desc_name] = sim
            
            return {
                "success": True,
                "file1": file1.filename,
                "file2": file2.filename,
                "similarity": float(similarity),
                "details": descriptor_similarities,
                "model1_info": {
                    "num_vertices": desc1['num_vertices'],
                    "num_faces": desc1['num_faces'],
                    "surface_area": desc1['surface_area']
                },
                "model2_info": {
                    "num_vertices": desc2['num_vertices'],
                    "num_faces": desc2['num_faces'],
                    "surface_area": desc2['surface_area']
                }
            }
        finally:
            os.unlink(tmp1_path)
            os.unlink(tmp2_path)
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/3d/descriptor-info")
async def get_3d_descriptor_info():
    """
    Get information about available 3D shape descriptors.
    """
    return {
        "success": True,
        "descriptors": [
            {
                "name": "D1",
                "description": "Distribution of distances from centroid to random surface points",
                "type": "shape_distribution",
                "bins": 64
            },
            {
                "name": "D2",
                "description": "Distribution of distances between pairs of random surface points (most discriminative)",
                "type": "shape_distribution",
                "bins": 64
            },
            {
                "name": "D3",
                "description": "Distribution of sqrt(area) of triangles formed by 3 random points",
                "type": "shape_distribution",
                "bins": 64
            },
            {
                "name": "D4",
                "description": "Distribution of cbrt(volume) of tetrahedra formed by 4 random points",
                "type": "shape_distribution",
                "bins": 64
            },
            {
                "name": "A3",
                "description": "Distribution of angles between 3 random surface points",
                "type": "shape_distribution",
                "bins": 64
            },
            {
                "name": "Bounding Box",
                "description": "Features from oriented bounding box (aspect ratios, volume)",
                "type": "geometric",
                "dimensions": 6
            },
            {
                "name": "Moments",
                "description": "3D moment invariants (inertia tensor, higher-order moments)",
                "type": "geometric",
                "dimensions": 9
            },
            {
                "name": "Mesh Statistics",
                "description": "Statistical features of the mesh (vertex/face count, density)",
                "type": "geometric",
                "dimensions": 6
            },
            {
                "name": "Multi-view Histogram",
                "description": "Intensity histograms from multiple rendered viewpoints (OpenGL)",
                "type": "view_based",
                "requires_opengl": True
            },
            {
                "name": "Multi-view Silhouette",
                "description": "Silhouette features from multiple rendered viewpoints (OpenGL)",
                "type": "view_based",
                "requires_opengl": True
            }
        ],
        "opengl_enabled": True
    }


@app.post("/3d/thumbnail")
async def generate_3d_thumbnail(request: Request):
    """
    Generate a thumbnail image for a 3D model using OpenGL rendering.
    
    Expects: { 
        "model_path": "/path/to/model.obj",
        "output_path": "/path/to/thumbnail.png" (optional),
        "azimuth": 45.0 (optional),
        "elevation": 30.0 (optional)
    }
    
    If output_path is not provided, returns the thumbnail as base64.
    """
    global thumbnail_generator
    
    try:
        data = await request.json()
        if "model_path" not in data:
            raise HTTPException(status_code=400, detail="Missing 'model_path' field")
        
        model_path = data["model_path"]
        output_path = data.get("output_path")
        azimuth = float(data.get("azimuth", 45.0))
        elevation = float(data.get("elevation", 30.0))
        
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model file not found")
        
        # Initialize thumbnail generator if needed
        if thumbnail_generator is None:
            thumbnail_generator = ThumbnailGenerator(width=256, height=256)
            if not thumbnail_generator.initialize():
                raise HTTPException(status_code=500, detail="Failed to initialize OpenGL renderer")
        
        if output_path:
            # Generate and save to file
            success = thumbnail_generator.generate(model_path, output_path, azimuth, elevation)
            if not success:
                raise HTTPException(status_code=500, detail="Failed to generate thumbnail")
            
            return {
                "success": True,
                "output_path": output_path
            }
        else:
            # Generate and return as base64
            obj = OBJLoader(model_path)
            if not obj.load():
                raise HTTPException(status_code=400, detail="Failed to load OBJ file")
            
            image = thumbnail_generator.renderer.render_view(obj, azimuth, elevation)
            if image is None:
                raise HTTPException(status_code=500, detail="Failed to render model")
            
            # Convert to base64
            from PIL import Image as PILImage
            import io
            import base64
            
            pil_img = PILImage.fromarray(image)
            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return {
                "success": True,
                "thumbnail": f"data:image/png;base64,{img_base64}"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/3d/generate-missing-thumbnails")
async def generate_missing_thumbnails(request: Request):
    """
    Generate thumbnails for all models that don't have existing thumbnails.
    
    Expects: { "category": "optional_category_filter" }
    """
    global thumbnail_generator
    
    try:
        data = await request.json() if request.headers.get("content-type") == "application/json" else {}
        category_filter = data.get("category")
        
        # Initialize thumbnail generator if needed
        if thumbnail_generator is None:
            thumbnail_generator = ThumbnailGenerator(width=256, height=256)
            if not thumbnail_generator.initialize():
                raise HTTPException(status_code=500, detail="Failed to initialize OpenGL renderer")
        
        generated = []
        failed = []
        skipped = []
        
        # Get categories to process
        if category_filter:
            categories = [category_filter]
        else:
            categories = [d for d in os.listdir(MODELS_3D_PATH) 
                         if os.path.isdir(os.path.join(MODELS_3D_PATH, d)) and d != 'All Models']
        
        for category in categories:
            models_dir = os.path.join(MODELS_3D_PATH, category)
            thumbs_dir = os.path.join(THUMBNAILS_3D_PATH, category)
            
            if not os.path.exists(models_dir):
                continue
            
            # Ensure thumbnails directory exists
            os.makedirs(thumbs_dir, exist_ok=True)
            
            # Get all OBJ files
            obj_files = [f for f in os.listdir(models_dir) if f.lower().endswith('.obj')]
            
            for obj_file in obj_files:
                obj_path = os.path.join(models_dir, obj_file)
                base_name = obj_file.replace('.obj', '').replace('.OBJ', '')
                
                # Check if thumbnail already exists
                existing = _find_thumbnail(THUMBNAILS_3D_PATH, category, obj_file)
                if existing:
                    skipped.append({"model": obj_file, "category": category, "existing": existing})
                    continue
                
                # Generate thumbnail
                thumb_path = os.path.join(thumbs_dir, f"{base_name}.png")
                try:
                    success = thumbnail_generator.generate(obj_path, thumb_path)
                    if success:
                        generated.append({"model": obj_file, "category": category, "thumbnail": thumb_path})
                    else:
                        failed.append({"model": obj_file, "category": category, "error": "Generation failed"})
                except Exception as e:
                    failed.append({"model": obj_file, "category": category, "error": str(e)})
        
        return {
            "success": True,
            "generated": len(generated),
            "failed": len(failed),
            "skipped": len(skipped),
            "details": {
                "generated": generated[:50],  # Limit response size
                "failed": failed[:50],
                "skipped_count": len(skipped)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
