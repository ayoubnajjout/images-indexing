import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ImageViewer, ObjectList } from '../components/detection';
import { Button, Toast, Spinner } from '../components/ui';
import { imageService } from '../services';

const ImageDetailPage = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(true);
  const [detecting, setDetecting] = useState(false);
  const [selectedObjectId, setSelectedObjectId] = useState(null);
  const [toast, setToast] = useState({ visible: false, message: '', type: 'info' });
  
  useEffect(() => {
    loadImage();
  }, [id]);
  
  const loadImage = async () => {
    try {
      setLoading(true);
      const data = await imageService.getImage(id);
      setImage(data);
      
      // Auto-select first object if available
      if (data.detections && data.detections.length > 0) {
        setSelectedObjectId(data.detections[0].id);
      }
    } catch (error) {
      showToast('Failed to load image', 'error');
    } finally {
      setLoading(false);
    }
  };
  
  const handleRunDetection = async () => {
    try {
      setDetecting(true);
      showToast('Running object detection...', 'info');
      
      const result = await imageService.detectObjects(id);
      
      setImage(prev => ({
        ...prev,
        detections: result.detections,
        objectCount: result.detections.length,
      }));
      
      if (result.detections.length > 0) {
        setSelectedObjectId(result.detections[0].id);
        showToast(`Detected ${result.detections.length} objects`, 'success');
      } else {
        showToast('No objects detected', 'warning');
      }
    } catch (error) {
      showToast('Object detection failed', 'error');
    } finally {
      setDetecting(false);
    }
  };
  
  const handleObjectSelect = (objectId) => {
    setSelectedObjectId(objectId);
  };
  
  const showToast = (message, type) => {
    setToast({ visible: true, message, type });
  };
  
  const hideToast = () => {
    setToast({ ...toast, visible: false });
  };
  
  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <Spinner size="lg" />
      </div>
    );
  }
  
  if (!image) {
    return (
      <div className="text-center py-16">
        <h2 className="text-2xl font-semibold text-slate-900 mb-2">Image not found</h2>
        <p className="text-slate-600 mb-4">The requested image could not be found</p>
        <Button onClick={() => navigate('/')}>Back to Collection</Button>
      </div>
    );
  }
  
  return (
    <div>
      {/* Header */}
      <div className="mb-6 flex items-center justify-between">
        <div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => navigate('/')}
            className="mb-2"
          >
            <svg className="w-4 h-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
            Back to Collection
          </Button>
          <h1 className="text-3xl font-semibold text-slate-900">{image.filename}</h1>
          <p className="text-slate-600 mt-1">
            Uploaded on {image.uploadDate}
          </p>
        </div>
        
        <div className="flex gap-2">
          {image.detections && image.detections.length > 0 ? (
            <Button
              variant="secondary"
              onClick={handleRunDetection}
              disabled={detecting}
            >
              {detecting ? (
                <>
                  <Spinner size="sm" className="mr-2" />
                  Re-detecting...
                </>
              ) : (
                'Re-detect Objects'
              )}
            </Button>
          ) : (
            <Button
              variant="primary"
              onClick={handleRunDetection}
              disabled={detecting}
            >
              {detecting ? (
                <>
                  <Spinner size="sm" className="mr-2" />
                  Detecting...
                </>
              ) : (
                'Run Detection'
              )}
            </Button>
          )}
        </div>
      </div>
      
      {/* Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Image Viewer */}
        <div className="lg:col-span-2">
          <ImageViewer
            image={image}
            detections={image.detections}
            selectedObjectId={selectedObjectId}
            onObjectSelect={handleObjectSelect}
          />
        </div>
        
        {/* Right: Object List */}
        <div>
          <ObjectList
            detections={image.detections}
            selectedObjectId={selectedObjectId}
            onObjectSelect={handleObjectSelect}
            imageId={image.id}
          />
        </div>
      </div>
      
      {/* Toast */}
      <Toast
        message={toast.message}
        type={toast.type}
        isVisible={toast.visible}
        onClose={hideToast}
      />
    </div>
  );
};

export default ImageDetailPage;
