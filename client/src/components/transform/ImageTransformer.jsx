import React, { useState, useRef, useEffect } from 'react';
import { Card, Button } from '../ui';

const ImageTransformer = ({ image, transformations, onTransformationChange }) => {
  const canvasRef = useRef(null);
  const [previewUrl, setPreviewUrl] = useState(image?.url);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    if (image && canvasRef.current) {
      applyTransformations();
    }
  }, [image, transformations]);
  
  const applyTransformations = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    setLoading(true);
    setError(null);
    
    const ctx = canvas.getContext('2d');
    const img = new Image();
    // Set crossOrigin to anonymous to allow canvas operations with CORS-enabled images
    img.crossOrigin = 'anonymous';
    
    img.onerror = (error) => {
      console.error('Image load error:', error, 'URL:', image.url);
      setError('Failed to load image. Check if the server is running.');
      setLoading(false);
    };
    
    img.onload = () => {
      try {
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
        
        // Try to export canvas - this will fail if CORS is not properly configured
        try {
          const dataUrl = canvas.toDataURL('image/jpeg');
          setPreviewUrl(dataUrl);
          setLoading(false);
        } catch (canvasError) {
          console.error('Canvas toDataURL error:', canvasError);
          setError('CORS error: Cannot export canvas. Image might be from a different origin.');
          setLoading(false);
        }
      } catch (transformError) {
        console.error('Transform error:', transformError);
        setError('Failed to apply transformations.');
        setLoading(false);
      }
    };
    
    // Add cache-busting parameter to ensure fresh CORS headers
    const imageUrl = image.url.includes('?') 
      ? `${image.url}&t=${Date.now()}` 
      : `${image.url}?t=${Date.now()}`;
    img.src = imageUrl;
  };
  
  return (
    <Card>
      <Card.Header>
        <h3 className="text-lg font-semibold text-slate-900">Preview</h3>
      </Card.Header>
      <Card.Body>
        <div className="bg-slate-900 rounded-lg p-4 flex items-center justify-center min-h-[400px]">
          <canvas ref={canvasRef} className="hidden" />
          {loading ? (
            <div className="flex flex-col items-center text-white">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mb-4"></div>
              <p>Loading image...</p>
            </div>
          ) : error ? (
            <div className="text-center text-red-400">
              <svg className="w-16 h-16 mx-auto mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
              <p className="text-lg mb-2">{error}</p>
              <p className="text-sm text-slate-400">{image?.url}</p>
            </div>
          ) : previewUrl ? (
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
