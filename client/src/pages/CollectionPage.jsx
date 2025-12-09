import React, { useState, useEffect, useRef, useCallback } from 'react';
import { UploadPanel, FilterBar, ImageGrid } from '../components/collection';
import { Toast } from '../components/ui';
import { imageService } from '../services';

const CollectionPage = () => {
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [filters, setFilters] = useState({ category: 'all' });
  const [pagination, setPagination] = useState({ page: 1, hasMore: true });
  const [toast, setToast] = useState({ visible: false, message: '', type: 'info' });
  const observer = useRef();
  
  // Ref for the last image element (for intersection observer)
  const lastImageRef = useCallback(node => {
    if (loading || loadingMore) return;
    if (observer.current) observer.current.disconnect();
    
    observer.current = new IntersectionObserver(entries => {
      if (entries[0].isIntersecting && pagination.hasMore) {
        loadMoreImages();
      }
    });
    
    if (node) observer.current.observe(node);
  }, [loading, loadingMore, pagination.hasMore]);
  
  useEffect(() => {
    setImages([]);
    setPagination({ page: 1, hasMore: true });
    loadImages(true);
  }, [filters]);
  
  const loadImages = async (reset = false) => {
    try {
      if (reset) {
        setLoading(true);
        setImages([]);
      }
      
      const currentPage = reset ? 1 : pagination.page;
      let allImages = [];
      let hasMore = false;
      
      // Load based on filter with pagination
      if (filters.category === 'all') {
        // Load all categories with pagination
        const [uploadedRes, editedRes, categoriesRes] = await Promise.all([
          imageService.getImages({ category: 'uploaded', page: currentPage, limit: 10 }),
          imageService.getImages({ category: 'edited', page: currentPage, limit: 10 }),
          imageService.getCategoryImages(currentPage, 10)
        ]);
        
        allImages = [...uploadedRes.images, ...editedRes.images, ...categoriesRes.images];
        hasMore = uploadedRes.hasMore || editedRes.hasMore || categoriesRes.hasMore;
      } else if (filters.category === 'categories') {
        const categoriesRes = await imageService.getCategoryImages(currentPage, 20);
        allImages = categoriesRes.images;
        hasMore = categoriesRes.hasMore;
      } else {
        const response = await imageService.getImages({ 
          category: filters.category, 
          page: currentPage, 
          limit: 20 
        });
        allImages = response.images;
        hasMore = response.hasMore;
      }
      
      setImages(prev => reset ? allImages : [...prev, ...allImages]);
      setPagination({ page: currentPage, hasMore });
    } catch (error) {
      console.error('Load images error:', error);
      showToast('Failed to load images', 'error');
    } finally {
      setLoading(false);
    }
  };
  
  const loadMoreImages = async () => {
    if (loadingMore || !pagination.hasMore) return;
    
    try {
      setLoadingMore(true);
      setPagination(prev => ({ ...prev, page: prev.page + 1 }));
      
      const nextPage = pagination.page + 1;
      let moreImages = [];
      let hasMore = false;
      
      if (filters.category === 'all') {
        const [uploadedRes, editedRes, categoriesRes] = await Promise.all([
          imageService.getImages({ category: 'uploaded', page: nextPage, limit: 10 }),
          imageService.getImages({ category: 'edited', page: nextPage, limit: 10 }),
          imageService.getCategoryImages(nextPage, 10)
        ]);
        moreImages = [...uploadedRes.images, ...editedRes.images, ...categoriesRes.images];
        hasMore = uploadedRes.hasMore || editedRes.hasMore || categoriesRes.hasMore;
      } else if (filters.category === 'categories') {
        const response = await imageService.getCategoryImages(nextPage, 20);
        moreImages = response.images;
        hasMore = response.hasMore;
      } else if (filters.category !== 'categories') {
        const response = await imageService.getImages({ 
          category: filters.category, 
          page: nextPage, 
          limit: 20 
        });
        moreImages = response.images;
        hasMore = response.hasMore;
      }
      
      setImages(prev => [...prev, ...moreImages]);
      setPagination({ page: nextPage, hasMore });
    } catch (error) {
      console.error('Load more error:', error);
    } finally {
      setLoadingMore(false);
    }
  };
  
  const handleUpload = async (file) => {
    try {
      const response = await imageService.uploadImage(file);
      showToast('Image uploaded successfully', 'success');
      setImages([]);
      setPagination({ page: 1, hasMore: true });
      await loadImages(true);
    } catch (error) {
      console.error('Upload error:', error);
      showToast('Failed to upload image', 'error');
    }
  };
  
  const handleDelete = async (imageId) => {
    if (!window.confirm('Are you sure you want to delete this image?')) {
      return;
    }
    
    try {
      await imageService.deleteImage(imageId);
      showToast('Image deleted successfully', 'success');
      setImages([]);
      setPagination({ page: 1, hasMore: true });
      await loadImages(true);
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
      <FilterBar onFilterChange={handleFilterChange} />
      
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
        loadingMore={loadingMore}
        onDelete={handleDelete}
        onDetect={handleDetect}
        lastImageRef={lastImageRef}
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
