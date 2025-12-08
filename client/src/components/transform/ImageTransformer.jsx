import React, { useState, useRef, useEffect } from 'react';
import { Card, Button } from '../ui';

const ImageTransformer = ({ image, transformations, onTransformationChange }) => {
  const canvasRef = useRef(null);
  const [previewUrl, setPreviewUrl] = useState(image?.url);
  
  useEffect(() => {
    if (image && canvasRef.current) {
      applyTransformations();
    }
  }, [image, transformations]);
  
  const applyTransformations = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const img = new Image();
    img.crossOrigin = 'anonymous';
    
    img.onload = () => {
      let width = img.width;
      let height = img.height;
      
      // Apply scale transformation
      if (transformations.scale && transformations.scale !== 100) {
        const scaleFactor = transformations.scale / 100;
        width = img.width * scaleFactor;
        height = img.height * scaleFactor;
      }
      
      canvas.width = width;
      canvas.height = height;
      
      ctx.save();
      
      // Apply rotation
      if (transformations.rotate) {
        ctx.translate(width / 2, height / 2);
        ctx.rotate((transformations.rotate * Math.PI) / 180);
        ctx.translate(-width / 2, -height / 2);
      }
      
      // Apply flip
      if (transformations.flipH || transformations.flipV) {
        ctx.translate(
          transformations.flipH ? width : 0,
          transformations.flipV ? height : 0
        );
        ctx.scale(
          transformations.flipH ? -1 : 1,
          transformations.flipV ? -1 : 1
        );
      }
      
      ctx.drawImage(img, 0, 0, width, height);
      ctx.restore();
      
      setPreviewUrl(canvas.toDataURL('image/jpeg'));
    };
    
    img.src = image.url;
  };
  
  return (
    <Card>
      <Card.Header>
        <h3 className="text-lg font-semibold text-slate-900">Preview</h3>
      </Card.Header>
      <Card.Body>
        <div className="bg-slate-900 rounded-lg p-4 flex items-center justify-center min-h-[400px]">
          <canvas ref={canvasRef} className="hidden" />
          {previewUrl ? (
            <img
              src={previewUrl}
              alt="Transformed preview"
              className="max-w-full max-h-[500px] rounded"
            />
          ) : (
            <div className="text-slate-400">Select an image to preview</div>
          )}
        </div>
      </Card.Body>
    </Card>
  );
};

export default ImageTransformer;
