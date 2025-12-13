"""
YOLO Object Detection module using custom ImageNet model.
"""

import numpy as np
import cv2
from ultralytics import YOLO
import os

# Get the directory where this file is located
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Default model path (model.pt in the same directory)
DEFAULT_MODEL_PATH = os.path.join(CURRENT_DIR, "model.pt")

# ImageNet class names (15 classes from custom model)
IMAGENET_CLASSES = {
    0: 'tench',
    1: 'great_white_shark',
    2: 'eft',
    3: 'bullfrog',
    4: 'african_crocodile',
    5: 'vine_snake',
    6: 'black_gold_garden_spider',
    7: 'barn_spider',
    8: 'sulphur_crested_cockatoo',
    9: 'chambered_nautilus',
    10: 'american_egret',
    11: 'staffordshire_bullterrier',
    12: 'chesapeake_bay_retriever',
    13: 'greater_swiss_mountain_dog',
    14: 'mexican_hairless',
}

# Human-readable names for ImageNet classes
IMAGENET_READABLE_NAMES = {
    0: 'Tench',
    1: 'Great White Shark',
    2: 'Eft (Newt)',
    3: 'Bullfrog',
    4: 'African Crocodile',
    5: 'Vine Snake',
    6: 'Black & Gold Garden Spider',
    7: 'Barn Spider',
    8: 'Sulphur-crested Cockatoo',
    9: 'Chambered Nautilus',
    10: 'American Egret',
    11: 'Staffordshire Bullterrier',
    12: 'Chesapeake Bay Retriever',
    13: 'Greater Swiss Mountain Dog',
    14: 'Mexican Hairless Dog',
}

# All category indices (0-14 for the custom model)
ALL_CATEGORY_INDICES = set(range(15))


class YOLODetector:
    """YOLO object detector with custom ImageNet model."""
    
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.25, filter_categories: bool = False):
        """
        Initialize the YOLO detector.
        
        Args:
            model_path: Path to custom model weights. If None, uses model.pt from current directory.
            confidence_threshold: Minimum confidence for detections.
            filter_categories: If True, only return detections for specified categories.
        """
        self.confidence_threshold = confidence_threshold
        self.filter_categories = filter_categories
        
        # Determine model path
        if model_path and os.path.exists(model_path):
            self.model_path = model_path
        elif os.path.exists(DEFAULT_MODEL_PATH):
            self.model_path = DEFAULT_MODEL_PATH
        else:
            raise FileNotFoundError(f"Model not found. Please ensure model.pt exists at {DEFAULT_MODEL_PATH}")
        
        # Load model
        self.model = YOLO(self.model_path)
        
        # Use all categories by default
        self.selected_indices = ALL_CATEGORY_INDICES
    
    def detect(self, image: np.ndarray) -> list:
        """
        Detect objects in an image.
        
        Args:
            image: BGR image as numpy array.
            
        Returns:
            List of detections, each containing:
            - id: unique detection ID
            - label: class name (ImageNet synset ID)
            - label_readable: human-readable class name
            - confidence: detection confidence
            - bbox: [x1, y1, x2, y2] bounding box
        """
        # Run inference
        results = self.model(image, device='cpu', verbose=False)[0]
        
        detections = []
        detection_id = 1
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            # Skip if below threshold
            if confidence < self.confidence_threshold:
                continue
            
            # Skip if not in selected categories (when filtering is enabled)
            if self.filter_categories and cls_id not in self.selected_indices:
                continue
            
            # Get class name (ImageNet synset ID and readable name)
            label = IMAGENET_CLASSES.get(cls_id, f"class_{cls_id}")
            label_readable = IMAGENET_READABLE_NAMES.get(cls_id, f"class_{cls_id}")
            
            # Get bounding box
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            detections.append({
                "id": detection_id,
                "label": label,
                "label_readable": label_readable,
                "class_id": cls_id,
                "confidence": round(confidence, 3),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
            })
            
            detection_id += 1
        
        return detections
    
    def detect_batch(self, images: list) -> list:
        """
        Detect objects in multiple images.
        
        Args:
            images: List of BGR images as numpy arrays.
            
        Returns:
            List of detection lists, one per image.
        """
        all_detections = []
        for image in images:
            detections = self.detect(image)
            all_detections.append(detections)
        return all_detections
