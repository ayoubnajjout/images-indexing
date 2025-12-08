import React, { useState, useEffect } from 'react';
import { UploadPanel, FilterBar, ImageGrid } from '../components/collection';
import { Toast } from '../components/ui';
import { imageService } from '../services';

const CollectionPage = () => {
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filters, setFilters] = useState({ search: '', category: '', hasDetections: false });
  const [toast, setToast] = useState({ visible: false, message: '', type: 'info' });
  const [categories, setCategories] = useState(['person', 'car', 'dog', 'cat', 'laptop', 'umbrella']);
  
  useEffect(() => {
    loadImages();
  }, [filters]);
  
  const loadImages = async () => {
    try {
      setLoading(true);
      const response = await imageService.getImages(filters);
      setImages(response.images);
    } catch (error) {
      showToast('Failed to load images', 'error');
    } finally {
      setLoading(false);
    }
  };
  
  const handleUpload = async (files) => {
    try {
      const response = await imageService.uploadImages(files);
      showToast(`Successfully uploaded ${response.images.length} image(s)`, 'success');
      await loadImages();
    } catch (error) {
      showToast('Failed to upload images', 'error');
    }
  };
  
  const handleDelete = async (imageId) => {
    if (!window.confirm('Are you sure you want to delete this image?')) {
      return;
    }
    
    try {
      await imageService.deleteImage(imageId);
      showToast('Image deleted successfully', 'success');
      await loadImages();
    } catch (error) {
      showToast('Failed to delete image', 'error');
    }
  };
  
  const handleDetect = async (imageId) => {
    try {
      showToast('Running object detection...', 'info');
      await imageService.detectObjects(imageId);
      showToast('Object detection completed', 'success');
      await loadImages();
    } catch (error) {
      showToast('Object detection failed', 'error');
    }
  };
  
  const handleFilterChange = (newFilters) => {
    setFilters(newFilters);
  };
  
  const showToast = (message, type) => {
    setToast({ visible: true, message, type });
  };
  
  const hideToast = () => {
    setToast({ ...toast, visible: false });
  };
  
  return (
    <div>
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-semibold text-slate-900">Image Collection</h1>
        <p className="text-slate-600 mt-1">
          Upload, manage, and explore your image collection
        </p>
      </div>
      
      {/* Upload Panel */}
      <UploadPanel onUpload={handleUpload} />
      
      {/* Filter Bar */}
      <FilterBar onFilterChange={handleFilterChange} categories={categories} />
      
      {/* Stats */}
      <div className="mb-6 flex items-center gap-4 text-sm text-slate-600">
        <span>Total images: {images.length}</span>
        <span>•</span>
        <span>With detections: {images.filter(img => img.objectCount > 0).length}</span>
      </div>
      
      {/* Image Grid */}
      <ImageGrid
        images={images}
        loading={loading}
        onDelete={handleDelete}
        onDetect={handleDetect}
      />
      
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

export default CollectionPage;
