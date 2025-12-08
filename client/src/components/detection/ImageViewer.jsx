import React, { useState, useEffect, useRef } from 'react';
import { Card } from '../ui';

const ImageViewer = ({ image, detections = [], selectedObjectId, onObjectSelect }) => {
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const imageRef = useRef(null);
  const containerRef = useRef(null);
  
  useEffect(() => {
    if (imageRef.current) {
      const updateDimensions = () => {
        setDimensions({
          width: imageRef.current.offsetWidth,
          height: imageRef.current.offsetHeight,
        });
      };
      
      imageRef.current.onload = updateDimensions;
      if (imageRef.current.complete) {
        updateDimensions();
      }
      
      window.addEventListener('resize', updateDimensions);
      return () => window.removeEventListener('resize', updateDimensions);
    }
  }, [image]);
  
  const getBoundingBoxStyle = (bbox) => {
    if (!dimensions.width || !dimensions.height) return {};
    
    const [x1, y1, x2, y2] = bbox;
    
    // Assuming bbox is in pixel coordinates relative to original image
    // We need to scale to displayed image size
    const img = imageRef.current;
    if (!img) return {};
    
    const scaleX = dimensions.width / img.naturalWidth;
    const scaleY = dimensions.height / img.naturalHeight;
    
    return {
      left: `${x1 * scaleX}px`,
      top: `${y1 * scaleY}px`,
      width: `${(x2 - x1) * scaleX}px`,
      height: `${(y2 - y1) * scaleY}px`,
    };
  };
  
  return (
    <Card className="overflow-hidden">
      <Card.Body className="p-0">
        <div ref={containerRef} className="relative bg-slate-900">
          <img
            ref={imageRef}
            src={image.url}
            alt={image.filename}
            className="w-full h-auto"
          />
          
          {/* Bounding Boxes Overlay */}
          {detections.map((detection) => (
            <div
              key={detection.id}
              style={getBoundingBoxStyle(detection.bbox)}
              className={`absolute border-2 transition cursor-pointer ${
                selectedObjectId === detection.id
                  ? 'border-indigo-500'
                  : 'border-green-400 hover:border-green-500'
              }`}
              onClick={() => onObjectSelect(detection.id)}
            >
              <div className={`absolute -top-6 left-0 px-2 py-0.5 text-xs font-medium rounded ${
                selectedObjectId === detection.id
                  ? 'bg-indigo-500 text-white'
                  : 'bg-green-400 text-white'
              }`}>
                {detection.label} ({Math.round(detection.confidence * 100)}%)
              </div>
            </div>
          ))}
        </div>
      </Card.Body>
    </Card>
  );
};

export default ImageViewer;
