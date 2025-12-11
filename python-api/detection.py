"""
YOLO Object Detection module using YOLOv8n.
"""

import numpy as np
import cv2
from ultralytics import YOLO
import os

# 15 categories to focus on (subset of COCO classes)
SELECTED_CATEGORIES = [
    'person', 'car', 'dog', 'cat', 'bird',
    'bicycle', 'motorcycle', 'bus', 'truck', 'boat',
    'horse', 'sheep', 'cow', 'elephant', 'bear'
]

# COCO class names (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class YOLODetector:
    """YOLOv8n object detector."""
    
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.25, filter_categories: bool = True):
        """
        Initialize the YOLO detector.
        
        Args:
            model_path: Path to custom model weights. If None, uses pretrained yolov8n.
            confidence_threshold: Minimum confidence for detections.
            filter_categories: If True, only return detections for SELECTED_CATEGORIES.
        """
        self.confidence_threshold = confidence_threshold
        self.filter_categories = filter_categories
        
        # Load model
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            # Use pretrained YOLOv8n
            self.model = YOLO('yolov8n.pt')
        
        # Build category index mapping
        self.selected_indices = set()
        for i, cls in enumerate(COCO_CLASSES):
            if cls in SELECTED_CATEGORIES:
                self.selected_indices.add(i)
    
    def detect(self, image: np.ndarray) -> list:
        """
        Detect objects in an image.
        
        Args:
            image: BGR image as numpy array.
            
        Returns:
            List of detections, each containing:
            - id: unique detection ID
            - label: class name
            - confidence: detection confidence
            - bbox: [x1, y1, x2, y2] bounding box
        """
        # Run inference
        results = self.model(image, verbose=False)[0]
        
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
            
            # Get class name
            label = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"class_{cls_id}"
            
            # Get bounding box
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            detections.append({
                "id": detection_id,
                "label": label,
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
