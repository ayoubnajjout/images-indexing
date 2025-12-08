import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { QueryBuilder, QuerySummary, SearchResults } from '../components/search';
import { Toast } from '../components/ui';
import { imageService, searchService, descriptorService } from '../services';

const SearchPage = () => {
  const [searchParams] = useSearchParams();
  
  const [images, setImages] = useState([]);
  const [selectedImageId, setSelectedImageId] = useState(searchParams.get('imageId') || '');
  const [selectedObjectId, setSelectedObjectId] = useState(searchParams.get('objectId') || '');
  const [selectedImage, setSelectedImage] = useState(null);
  const [selectedObject, setSelectedObject] = useState(null);
  const [descriptors, setDescriptors] = useState(null);
  const [searchResults, setSearchResults] = useState([]);
  const [searching, setSearching] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
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
      const imagesWithDetections = response.images.filter(img => img.objectCount > 0);
      setImages(imagesWithDetections);
      
      // Auto-select from URL params
      if (searchParams.get('imageId')) {
        const imageId = searchParams.get('imageId');
        setSelectedImageId(imageId);
      }
    } catch (error) {
      showToast('Failed to load images', 'error');
    }
  };
  
  const loadImageDetails = async () => {
    try {
      const image = await imageService.getImage(selectedImageId);
      setSelectedImage(image);
      
      // Auto-select object from URL params or first object
      const objectId = searchParams.get('objectId') || 
                      (image.detections?.[0]?.id.toString());
      if (objectId) {
        setSelectedObjectId(objectId);
        const object = image.detections?.find(d => d.id.toString() === objectId);
        setSelectedObject(object);
      }
    } catch (error) {
      showToast('Failed to load image details', 'error');
    }
  };
  
  const loadDescriptors = async () => {
    try {
      const data = await descriptorService.getDescriptors(selectedImageId, selectedObjectId);
      setDescriptors(data.descriptors);
    } catch (error) {
      console.error('Failed to load descriptors:', error);
    }
  };
  
  const handleImageSelect = (imageId) => {
    setSelectedImageId(imageId);
    setSelectedObjectId('');
    setSelectedObject(null);
    setSearchResults([]);
    setHasSearched(false);
  };
  
  const handleObjectSelect = (objectId) => {
    setSelectedObjectId(objectId);
    const object = selectedImage?.detections?.find(d => d.id.toString() === objectId);
    setSelectedObject(object);
    setSearchResults([]);
    setHasSearched(false);
  };
  
  const handleSearch = async () => {
    try {
      setSearching(true);
      setHasSearched(true);
      showToast('Searching for similar objects...', 'info');
      
      const response = await searchService.searchSimilar(
        selectedImageId,
        selectedObjectId,
        { topK: 10 }
      );
      
      setSearchResults(response.results);
      showToast(`Found ${response.results.length} similar objects`, 'success');
    } catch (error) {
      showToast('Search failed', 'error');
    } finally {
      setSearching(false);
    }
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
        <h1 className="text-3xl font-semibold text-slate-900">Similarity Search</h1>
        <p className="text-slate-600 mt-1">
          Find objects similar to your query using visual descriptors
        </p>
      </div>
      
      {/* Query Builder */}
      <div className="mb-6">
        <QueryBuilder
          images={images}
          selectedImageId={selectedImageId}
          onImageSelect={handleImageSelect}
          selectedObjectId={selectedObjectId}
          onObjectSelect={handleObjectSelect}
          onSearch={handleSearch}
          searching={searching}
        />
      </div>
      
      {/* Query Summary */}
      {selectedImage && selectedObject && (
        <div className="mb-6">
          <QuerySummary
            image={selectedImage}
            object={selectedObject}
            descriptors={descriptors}
          />
        </div>
      )}
      
      {/* Search Results */}
      {hasSearched && (
        <SearchResults
          results={searchResults}
          loading={searching}
        />
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

export default SearchPage;
