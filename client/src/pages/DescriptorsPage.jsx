import React, { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { ColorDescriptors, TextureDescriptors, ShapeDescriptors, ObjectInfoPanel } from '../components/descriptors';
import { Button, Toast, Spinner, Badge, Card } from '../components/ui';
import { imageService, descriptorService } from '../services';

const DescriptorsPage = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  
  const [images, setImages] = useState([]);
  const [selectedImageId, setSelectedImageId] = useState(searchParams.get('imageId') || '');
  const [selectedObjectId, setSelectedObjectId] = useState(searchParams.get('objectId') || '');
  const [selectedImage, setSelectedImage] = useState(null);
  const [selectedObject, setSelectedObject] = useState(null);
  const [descriptors, setDescriptors] = useState(null);
  const [loading, setLoading] = useState(false);
  const [detecting, setDetecting] = useState(false);
  const [activeTab, setActiveTab] = useState('color');
  const [toast, setToast] = useState({ visible: false, message: '', type: 'info' });
  
  useEffect(() => {
    loadImages();
  }, []);
  
  useEffect(() => {
    if (selectedImageId) {
      loadImageDetails();
      setSelectedObjectId('');
      setDescriptors(null);
    }
  }, [selectedImageId]);
  
  useEffect(() => {
    if (selectedImageId && selectedObjectId) {
      loadDescriptors();
    }
  }, [selectedImageId, selectedObjectId]);
  
  const loadImages = async () => {
    try {
      setLoading(true);
      // Load all images including categories
      const [uploadedRes, editedRes, categoriesRes] = await Promise.all([
        imageService.getImages({ category: 'uploaded', limit: 100 }),
        imageService.getImages({ category: 'edited', limit: 100 }),
        imageService.getCategoryImages(1, 100)
      ]);
      
      const allImages = [
        ...uploadedRes.images,
        ...editedRes.images,
        ...categoriesRes.images
      ];
      
      setImages(allImages);
    } catch (error) {
      showToast('Failed to load images', 'error');
    } finally {
      setLoading(false);
    }
  };
  
  const loadImageDetails = async () => {
    try {
      const image = await imageService.getImage(selectedImageId);
      setSelectedImage(image);
      
      // Auto-select first object if available and URL param exists
      const objectIdFromUrl = searchParams.get('objectId');
      if (objectIdFromUrl && image.detections?.find(d => d.id.toString() === objectIdFromUrl)) {
        setSelectedObjectId(objectIdFromUrl);
      } else if (image.detections && image.detections.length > 0) {
        setSelectedObjectId(image.detections[0].id.toString());
      }
    } catch (error) {
      showToast('Failed to load image details', 'error');
    }
  };
  
  const handleRunDetection = async () => {
    try {
      setDetecting(true);
      showToast('Running object detection...', 'info');
      
      const result = await imageService.detectObjects(selectedImageId);
      
      // Update image with new detections
      setSelectedImage(prev => ({
        ...prev,
        detections: result.detections,
        objectCount: result.count
      }));
      
      if (result.detections.length > 0) {
        setSelectedObjectId(result.detections[0].id.toString());
        showToast(`Detected ${result.detections.length} objects`, 'success');
      } else {
        showToast('No objects detected in this image', 'warning');
      }
    } catch (error) {
      showToast('Detection failed', 'error');
    } finally {
      setDetecting(false);
    }
  };
  
  const loadDescriptors = async () => {
    try {
      setLoading(true);
      
      const object = selectedImage?.detections?.find(
        d => d.id.toString() === selectedObjectId
      );
      setSelectedObject(object);
      
      // First check if descriptors are already in the detection object
      if (object?.descriptors) {
        setDescriptors(object.descriptors);
      } else {
        // Fetch from API if not available locally
        const data = await descriptorService.getDescriptors(selectedImageId, selectedObjectId);
        setDescriptors(data.descriptors);
      }
    } catch (error) {
      showToast('Failed to load descriptors', 'error');
    } finally {
      setLoading(false);
    }
  };
  
  const showToast = (message, type) => {
    setToast({ visible: true, message, type });
  };
  
  const hideToast = () => {
    setToast({ ...toast, visible: false });
  };

  const hasDetections = selectedImage?.detections?.length > 0;
  
  return (
    <div className="max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-semibold text-slate-900">Descriptor Viewer</h1>
        <p className="text-slate-600 mt-1">
          Browse and explore visual descriptors for any object in your gallery
        </p>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Left Panel - Image & Object Selection */}
        <div className="lg:col-span-1 space-y-4">
          {/* Image Gallery */}
          <div className="bg-white border border-slate-200 rounded-lg p-4">
            <h3 className="font-semibold text-slate-900 mb-3">Select Image</h3>
            <div className="max-h-64 overflow-y-auto space-y-2">
              {loading && images.length === 0 ? (
                <div className="flex justify-center py-4">
                  <Spinner size="sm" />
                </div>
              ) : (
                images.map((image) => (
                  <div
                    key={image.id}
                    onClick={() => setSelectedImageId(image.id.toString())}
                    className={`flex items-center gap-2 p-2 rounded-lg cursor-pointer transition-colors ${
                      selectedImageId === image.id.toString()
                        ? 'bg-indigo-50 border border-indigo-200'
                        : 'hover:bg-slate-50 border border-transparent'
                    }`}
                  >
                    <img 
                      src={image.url} 
                      alt={image.filename}
                      className="w-10 h-10 object-cover rounded"
                    />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-slate-900 truncate">{image.filename}</p>
                      <p className="text-xs text-slate-500">
                        {image.objectCount > 0 ? `${image.objectCount} objects` : 'No detections'}
                      </p>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
          
          {/* Object Selection */}
          {selectedImage && (
            <div className="bg-white border border-slate-200 rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="font-semibold text-slate-900">Detected Objects</h3>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={handleRunDetection}
                  disabled={detecting}
                >
                  {detecting ? <Spinner size="sm" /> : hasDetections ? 'Re-detect' : 'Detect'}
                </Button>
              </div>
              
              {!hasDetections ? (
                <p className="text-sm text-slate-500 text-center py-4">
                  No objects detected. Click "Detect" to run YOLO detection.
                </p>
              ) : (
                <div className="space-y-2">
                  {selectedImage.detections.map((det) => (
                    <div
                      key={det.id}
                      onClick={() => setSelectedObjectId(det.id.toString())}
                      className={`p-3 rounded-lg cursor-pointer transition-colors ${
                        selectedObjectId === det.id.toString()
                          ? 'bg-indigo-50 border border-indigo-200'
                          : 'bg-slate-50 hover:bg-slate-100 border border-transparent'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-medium text-slate-900 capitalize">{det.label}</span>
                        <Badge variant={det.descriptors ? 'success' : 'warning'} className="text-xs">
                          {det.descriptors ? 'Has descriptors' : 'No descriptors'}
                        </Badge>
                      </div>
                      <p className="text-xs text-slate-500 mt-1">
                        Confidence: {Math.round(det.confidence * 100)}%
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
          
          {/* Selected Image Preview */}
          {selectedImage && (
            <div className="bg-white border border-slate-200 rounded-lg p-4">
              <h3 className="font-semibold text-slate-900 mb-3">Preview</h3>
              <img 
                src={selectedImage.url} 
                alt={selectedImage.filename}
                className="w-full rounded-lg border border-slate-200"
              />
              <p className="text-sm text-slate-500 mt-2 text-center">{selectedImage.filename}</p>
            </div>
          )}
        </div>
        
        {/* Right Panel - Descriptors Display */}
        <div className="lg:col-span-3">
          {loading ? (
            <div className="flex items-center justify-center min-h-[400px] bg-white border border-slate-200 rounded-lg">
              <Spinner size="lg" />
            </div>
          ) : selectedImageId && selectedObjectId && descriptors ? (
            <div className="space-y-4">
              {/* Object Info */}
              <div className="bg-white border border-slate-200 rounded-lg p-4">
                <div className="flex items-center gap-4">
                  <div className="w-16 h-16 bg-indigo-100 rounded-lg flex items-center justify-center">
                    <span className="text-2xl">🎯</span>
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-slate-900 capitalize">{selectedObject?.label}</h3>
                    <p className="text-sm text-slate-500">
                      Confidence: {Math.round((selectedObject?.confidence || 0) * 100)}% | 
                      From: {selectedImage?.filename}
                    </p>
                  </div>
                  <div className="ml-auto">
                    <Button
                      variant="primary"
                      size="sm"
                      onClick={() => navigate(`/search?imageId=${selectedImageId}&objectId=${selectedObjectId}`)}
                    >
                      Find Similar →
                    </Button>
                  </div>
                </div>
              </div>
              
              {/* Descriptor Tabs */}
              <div className="bg-white border border-slate-200 rounded-lg">
                <div className="flex border-b border-slate-200">
                  {[
                    { id: 'color', label: 'Color', icon: '🎨' },
                    { id: 'texture', label: 'Texture', icon: '🧱' },
                    { id: 'shape', label: 'Shape', icon: '📐' },
                  ].map((tab) => (
                    <button
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id)}
                      className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
                        activeTab === tab.id
                          ? 'text-indigo-600 border-b-2 border-indigo-600 bg-indigo-50'
                          : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                      }`}
                    >
                      <span className="mr-2">{tab.icon}</span>
                      {tab.label}
                    </button>
                  ))}
                </div>
                
                <div className="p-4">
                  {activeTab === 'color' && <ColorDescriptors descriptors={descriptors.color} />}
                  {activeTab === 'texture' && <TextureDescriptors descriptors={descriptors.texture} />}
                  {activeTab === 'shape' && <ShapeDescriptors descriptors={descriptors.shape} />}
                </div>
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center min-h-[400px] bg-white border border-slate-200 rounded-lg">
              <div className="inline-flex items-center justify-center w-16 h-16 bg-slate-100 rounded-full mb-4">
                <svg className="w-8 h-8 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <h3 className="text-lg font-medium text-slate-900 mb-2">No Selection</h3>
              <p className="text-slate-500 text-center max-w-sm">
                Select an image from the left panel, then choose an object to view its descriptors.
                {!hasDetections && selectedImage && (
                  <span className="block mt-2 text-indigo-600">
                    Run detection first to find objects in this image.
                  </span>
                )}
              </p>
            </div>
          )}
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

export default DescriptorsPage;
