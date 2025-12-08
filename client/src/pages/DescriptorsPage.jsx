import React, { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { ColorDescriptors, TextureDescriptors, ShapeDescriptors, ObjectInfoPanel } from '../components/descriptors';
import { Button, Dropdown, Toast, Spinner } from '../components/ui';
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
  const [toast, setToast] = useState({ visible: false, message: '', type: 'info' });
  
  useEffect(() => {
    loadImages();
  }, []);
  
  useEffect(() => {
    if (selectedImageId) {
      loadImageDetails();
    }
  }, [selectedImageId]);
  
  useEffect(() => {
    if (selectedImageId && selectedObjectId) {
      loadDescriptors();
    }
  }, [selectedImageId, selectedObjectId]);
  
  const loadImages = async () => {
    try {
      const response = await imageService.getImages({ hasDetections: true });
      setImages(response.images.filter(img => img.objectCount > 0));
    } catch (error) {
      showToast('Failed to load images', 'error');
    }
  };
  
  const loadImageDetails = async () => {
    try {
      const image = await imageService.getImage(selectedImageId);
      setSelectedImage(image);
      
      // Auto-select first object if not already selected
      if (!selectedObjectId && image.detections && image.detections.length > 0) {
        setSelectedObjectId(image.detections[0].id.toString());
      }
    } catch (error) {
      showToast('Failed to load image details', 'error');
    }
  };
  
  const loadDescriptors = async () => {
    try {
      setLoading(true);
      
      const object = selectedImage?.detections?.find(
        d => d.id.toString() === selectedObjectId
      );
      setSelectedObject(object);
      
      const data = await descriptorService.getDescriptors(selectedImageId, selectedObjectId);
      setDescriptors(data.descriptors);
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
  
  const imageOptions = images.map(img => ({
    value: img.id.toString(),
    label: `${img.filename} (${img.objectCount} objects)`,
  }));
  
  const objectOptions = selectedImage?.detections?.map(obj => ({
    value: obj.id.toString(),
    label: `${obj.label} (${Math.round(obj.confidence * 100)}%)`,
  })) || [];
  
  return (
    <div>
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-semibold text-slate-900">Descriptor Viewer</h1>
        <p className="text-slate-600 mt-1">
          Explore visual descriptors for detected objects
        </p>
      </div>
      
      {/* Selection Panel */}
      <div className="bg-white border border-slate-200 rounded-lg p-6 mb-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Image Selection */}
          <Dropdown
            label="Select Image"
            options={imageOptions}
            value={selectedImageId}
            onChange={setSelectedImageId}
            placeholder="Choose an image..."
          />
          
          {/* Object Selection */}
          <Dropdown
            label="Select Object"
            options={objectOptions}
            value={selectedObjectId}
            onChange={setSelectedObjectId}
            placeholder={selectedImageId ? 'Choose an object...' : 'Select an image first'}
          />
        </div>
      </div>
      
      {/* Content */}
      {loading ? (
        <div className="flex items-center justify-center min-h-[400px]">
          <Spinner size="lg" />
        </div>
      ) : selectedImageId && selectedObjectId && descriptors ? (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Panel: Object Info */}
          <div>
            <ObjectInfoPanel image={selectedImage} object={selectedObject} />
          </div>
          
          {/* Right Panel: Descriptors */}
          <div className="lg:col-span-2 space-y-6">
            <ColorDescriptors descriptors={descriptors.color} />
            <TextureDescriptors descriptors={descriptors.texture} />
            <ShapeDescriptors descriptors={descriptors.shape} />
          </div>
        </div>
      ) : (
        <div className="text-center py-16">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-slate-100 rounded-full mb-4">
            <svg className="w-8 h-8 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <h3 className="text-lg font-medium text-slate-900 mb-2">No selection</h3>
          <p className="text-slate-500">Select an image and object to view descriptors</p>
        </div>
      )}
      
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
