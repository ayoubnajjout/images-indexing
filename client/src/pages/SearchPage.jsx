import React, { useState, useEffect } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { Toast, Button, Spinner, Card, Badge } from '../components/ui';
import { imageService, searchService, descriptorService } from '../services';

const SearchPage = () => {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  
  // Pipeline step state
  const [currentStep, setCurrentStep] = useState(1);
  
  // Data states
  const [images, setImages] = useState([]);
  const [selectedImageId, setSelectedImageId] = useState(searchParams.get('imageId') || '');
  const [selectedObjectId, setSelectedObjectId] = useState(searchParams.get('objectId') || '');
  const [selectedImage, setSelectedImage] = useState(null);
  const [selectedObject, setSelectedObject] = useState(null);
  const [descriptors, setDescriptors] = useState(null);
  const [searchResults, setSearchResults] = useState([]);
  const [searchStats, setSearchStats] = useState(null);
  
  // Loading states
  const [loading, setLoading] = useState(false);
  const [searching, setSearching] = useState(false);
  
  // Descriptor type selection
  const [selectedDescriptor, setSelectedDescriptor] = useState('all');
  
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
      setLoading(true);
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
      
      // Filter to only show images with detections
      const imagesWithDetections = allImages.filter(img => img.objectCount > 0);
      setImages(imagesWithDetections);
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
    setDescriptors(null);
    setSearchResults([]);
    setSearchStats(null);
    setCurrentStep(2);
  };
  
  const handleObjectSelect = (objectId) => {
    setSelectedObjectId(objectId);
    const object = selectedImage?.detections?.find(d => d.id.toString() === objectId);
    setSelectedObject(object);
    setSearchResults([]);
    setSearchStats(null);
    setCurrentStep(3);
  };
  
  const handleDescriptorSelect = (descriptor) => {
    setSelectedDescriptor(descriptor);
    setSearchResults([]);
    setSearchStats(null);
  };
  
  const handleSearch = async () => {
    try {
      setSearching(true);
      const startTime = Date.now();
      
      // Build weights based on selected descriptor
      let weights = { color: true, texture: true, shape: true };
      if (selectedDescriptor !== 'all') {
        weights = { color: false, texture: false, shape: false };
        weights[selectedDescriptor] = true;
      }
      
      const response = await searchService.searchSimilar(
        selectedImageId,
        selectedObjectId,
        { topK: 6, weights }
      );
      
      const endTime = Date.now();
      
      setSearchResults(response.results);
      setSearchStats({
        searchTime: endTime - startTime,
        totalResults: response.results.length,
        descriptorUsed: selectedDescriptor,
        queryObject: selectedObject?.label,
        avgSimilarity: response.results.length > 0 
          ? (response.results.reduce((sum, r) => sum + r.similarity, 0) / response.results.length * 100).toFixed(1)
          : 0
      });
      setCurrentStep(4);
      
      showToast(`Found ${response.results.length} similar objects`, 'success');
    } catch (error) {
      showToast('Search failed', 'error');
    } finally {
      setSearching(false);
    }
  };
  
  const resetPipeline = () => {
    setSelectedImageId('');
    setSelectedObjectId('');
    setSelectedImage(null);
    setSelectedObject(null);
    setDescriptors(null);
    setSearchResults([]);
    setSearchStats(null);
    setSelectedDescriptor('all');
    setCurrentStep(1);
  };
  
  const showToast = (message, type) => {
    setToast({ visible: true, message, type });
  };
  
  const hideToast = () => {
    setToast({ ...toast, visible: false });
  };
  
  const steps = [
    { id: 1, name: 'Select Image', description: 'Choose an image from gallery' },
    { id: 2, name: 'Select Object', description: 'Pick a detected object' },
    { id: 3, name: 'Choose Descriptor', description: 'Select similarity criteria' },
    { id: 4, name: 'View Results', description: 'See similar objects' },
  ];
  
  return (
    <div className="max-w-6xl mx-auto">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-semibold text-slate-900">Similarity Search Pipeline</h1>
        <p className="text-slate-600 mt-1">
          Find similar objects step by step using visual descriptors
        </p>
      </div>
      
      {/* Progress Steps */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          {steps.map((step, index) => (
            <React.Fragment key={step.id}>
              <div className="flex flex-col items-center">
                <div 
                  className={`w-10 h-10 rounded-full flex items-center justify-center font-semibold text-sm transition-colors ${
                    currentStep >= step.id 
                      ? 'bg-indigo-600 text-white' 
                      : 'bg-slate-200 text-slate-500'
                  }`}
                >
                  {currentStep > step.id ? (
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                  ) : step.id}
                </div>
                <div className="mt-2 text-center">
                  <div className={`text-sm font-medium ${currentStep >= step.id ? 'text-slate-900' : 'text-slate-400'}`}>
                    {step.name}
                  </div>
                  <div className="text-xs text-slate-400 hidden sm:block">{step.description}</div>
                </div>
              </div>
              {index < steps.length - 1 && (
                <div className={`flex-1 h-1 mx-4 rounded ${currentStep > step.id ? 'bg-indigo-600' : 'bg-slate-200'}`} />
              )}
            </React.Fragment>
          ))}
        </div>
      </div>
      
      {/* Step Content */}
      <div className="bg-white border border-slate-200 rounded-lg p-6">
        {/* Step 1: Select Image */}
        {currentStep === 1 && (
          <div>
            <h3 className="text-lg font-semibold text-slate-900 mb-4">Select an Image with Detected Objects</h3>
            {loading ? (
              <div className="flex justify-center py-12">
                <Spinner size="lg" />
              </div>
            ) : images.length === 0 ? (
              <div className="text-center py-12">
                <p className="text-slate-500">No images with detected objects found.</p>
                <p className="text-sm text-slate-400 mt-2">Run the batch processing script first.</p>
              </div>
            ) : (
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
                {images.map((image) => (
                  <div
                    key={image.id}
                    onClick={() => handleImageSelect(image.id)}
                    className={`cursor-pointer rounded-lg border-2 overflow-hidden transition-all hover:shadow-lg ${
                      selectedImageId === image.id ? 'border-indigo-500 ring-2 ring-indigo-200' : 'border-slate-200'
                    }`}
                  >
                    <div className="aspect-square relative">
                      <img src={image.url} alt={image.filename} className="w-full h-full object-cover" />
                      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent p-2">
                        <Badge variant="primary" className="text-xs">
                          {image.objectCount} objects
                        </Badge>
                      </div>
                    </div>
                    <div className="p-2">
                      <p className="text-xs text-slate-600 truncate">{image.filename}</p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
        
        {/* Step 2: Select Object */}
        {currentStep === 2 && selectedImage && (
          <div>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-slate-900">Select an Object to Search</h3>
              <Button variant="secondary" size="sm" onClick={() => setCurrentStep(1)}>
                ← Back
              </Button>
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Selected Image Preview */}
              <div>
                <img 
                  src={selectedImage.url} 
                  alt={selectedImage.filename}
                  className="w-full rounded-lg border border-slate-200"
                />
                <p className="text-sm text-slate-500 mt-2 text-center">{selectedImage.filename}</p>
              </div>
              
              {/* Object List */}
              <div className="space-y-3">
                <p className="text-sm text-slate-600 mb-3">
                  Found {selectedImage.detections?.length || 0} objects in this image:
                </p>
                {selectedImage.detections?.map((detection) => (
                  <div
                    key={detection.id}
                    onClick={() => handleObjectSelect(detection.id.toString())}
                    className={`p-4 rounded-lg border-2 cursor-pointer transition-all hover:shadow-md ${
                      selectedObjectId === detection.id.toString() 
                        ? 'border-indigo-500 bg-indigo-50' 
                        : 'border-slate-200 hover:border-slate-300'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <span className="font-medium text-slate-900 capitalize">{detection.label}</span>
                        <span className="ml-2 text-sm text-slate-500">
                          ({Math.round(detection.confidence * 100)}% confidence)
                        </span>
                      </div>
                      <Badge variant={detection.descriptors ? 'success' : 'warning'}>
                        {detection.descriptors ? 'Has Descriptors' : 'No Descriptors'}
                      </Badge>
                    </div>
                    {detection.bbox && (
                      <p className="text-xs text-slate-400 mt-1">
                        Bounding box: [{detection.bbox.map(v => Math.round(v)).join(', ')}]
                      </p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
        
        {/* Step 3: Choose Descriptor */}
        {currentStep === 3 && selectedObject && (
          <div>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-slate-900">Choose Descriptor Type</h3>
              <Button variant="secondary" size="sm" onClick={() => setCurrentStep(2)}>
                ← Back
              </Button>
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Query Summary */}
              <div className="bg-slate-50 rounded-lg p-4">
                <h4 className="font-medium text-slate-900 mb-3">Query Object</h4>
                <div className="flex items-start gap-4">
                  <img 
                    src={selectedImage.url} 
                    alt={selectedImage.filename}
                    className="w-24 h-24 object-cover rounded-lg"
                  />
                  <div>
                    <p className="font-medium capitalize">{selectedObject.label}</p>
                    <p className="text-sm text-slate-500">
                      Confidence: {Math.round(selectedObject.confidence * 100)}%
                    </p>
                    <p className="text-sm text-slate-500">
                      From: {selectedImage.filename}
                    </p>
                  </div>
                </div>
              </div>
              
              {/* Descriptor Selection */}
              <div className="space-y-3">
                <p className="text-sm text-slate-600 mb-3">
                  Select which descriptor(s) to use for similarity search:
                </p>
                
                {[
                  { id: 'all', name: 'All Descriptors', description: 'Use color, texture, and shape combined', icon: '🎯' },
                  { id: 'color', name: 'Color Only', description: 'HSV histogram & dominant colors', icon: '🎨' },
                  { id: 'texture', name: 'Texture Only', description: 'Tamura features & Gabor filters', icon: '🧱' },
                  { id: 'shape', name: 'Shape Only', description: 'Hu moments & orientation histogram', icon: '📐' },
                ].map((desc) => (
                  <div
                    key={desc.id}
                    onClick={() => handleDescriptorSelect(desc.id)}
                    className={`p-4 rounded-lg border-2 cursor-pointer transition-all hover:shadow-md ${
                      selectedDescriptor === desc.id 
                        ? 'border-indigo-500 bg-indigo-50' 
                        : 'border-slate-200 hover:border-slate-300'
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <span className="text-2xl">{desc.icon}</span>
                      <div>
                        <p className="font-medium text-slate-900">{desc.name}</p>
                        <p className="text-sm text-slate-500">{desc.description}</p>
                      </div>
                    </div>
                  </div>
                ))}
                
                <Button
                  variant="primary"
                  className="w-full mt-4"
                  onClick={handleSearch}
                  disabled={searching}
                >
                  {searching ? (
                    <>
                      <Spinner size="sm" className="mr-2" />
                      Searching...
                    </>
                  ) : (
                    <>
                      <svg className="w-5 h-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                      </svg>
                      Search Similar Objects
                    </>
                  )}
                </Button>
              </div>
            </div>
          </div>
        )}
        
        {/* Step 4: Results */}
        {currentStep === 4 && (
          <div>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-slate-900">Search Results</h3>
              <div className="flex gap-2">
                <Button variant="secondary" size="sm" onClick={() => setCurrentStep(3)}>
                  ← Back
                </Button>
                <Button variant="primary" size="sm" onClick={resetPipeline}>
                  New Search
                </Button>
              </div>
            </div>
            
            {/* Query Image & Stats Panel */}
            {searchStats && (
              <div className="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-lg p-4 mb-6">
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                  {/* Query Image */}
                  <div className="bg-white rounded-lg p-4">
                    <h4 className="font-medium text-slate-900 mb-3 text-center">Query Image</h4>
                    <div className="relative">
                      <img 
                        src={selectedImage?.url} 
                        alt={selectedImage?.filename}
                        className="w-full aspect-video object-cover rounded-lg border-2 border-indigo-500"
                      />
                      <div className="absolute top-2 left-2">
                        <Badge variant="primary">QUERY</Badge>
                      </div>
                      <div className="absolute top-2 right-2">
                        <Badge variant="success">100%</Badge>
                      </div>
                      <div className="absolute bottom-2 left-2">
                        <Badge variant="default" className="bg-black/70 text-white capitalize">
                          {selectedObject?.label}
                        </Badge>
                      </div>
                    </div>
                    <p className="text-sm text-slate-600 mt-2 text-center truncate">{selectedImage?.filename}</p>
                    <p className="text-xs text-slate-400 text-center">
                      Confidence: {Math.round((selectedObject?.confidence || 0) * 100)}%
                    </p>
                  </div>
                  
                  {/* Statistics */}
                  <div className="lg:col-span-2">
                    <h4 className="font-medium text-slate-900 mb-3">Search Statistics</h4>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                      <div className="bg-white rounded-lg p-3 text-center">
                        <p className="text-2xl font-bold text-indigo-600">{searchStats.totalResults}</p>
                        <p className="text-xs text-slate-500">Results Found</p>
                      </div>
                      <div className="bg-white rounded-lg p-3 text-center">
                        <p className="text-2xl font-bold text-indigo-600">{searchStats.searchTime}ms</p>
                        <p className="text-xs text-slate-500">Search Time</p>
                      </div>
                      <div className="bg-white rounded-lg p-3 text-center">
                        <p className="text-2xl font-bold text-indigo-600">{searchStats.avgSimilarity}%</p>
                        <p className="text-xs text-slate-500">Avg Similarity</p>
                      </div>
                      <div className="bg-white rounded-lg p-3 text-center">
                        <p className="text-lg font-bold text-indigo-600 capitalize">{searchStats.descriptorUsed}</p>
                        <p className="text-xs text-slate-500">Descriptor Used</p>
                      </div>
                    </div>
                    
                    {/* Additional Stats */}
                    <div className="mt-3 bg-white rounded-lg p-3">
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="text-slate-500">Query Object:</span>
                          <span className="ml-2 font-medium capitalize">{searchStats.queryObject}</span>
                        </div>
                        <div>
                          <span className="text-slate-500">Source:</span>
                          <span className="ml-2 font-medium">{selectedImage?.filename}</span>
                        </div>
                        <div>
                          <span className="text-slate-500">Database Objects:</span>
                          <span className="ml-2 font-medium">{searchStats.totalResults > 0 ? 'Compared' : 'None'}</span>
                        </div>
                        <div>
                          <span className="text-slate-500">Similarity Method:</span>
                          <span className="ml-2 font-medium capitalize">{searchStats.descriptorUsed} Descriptors</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            {/* Results Grid */}
            <div className="mb-4">
              <h4 className="font-medium text-slate-900">
                Top {searchResults.length} Similar Objects (by descriptor values only)
              </h4>
              <p className="text-xs text-slate-500">Similarity based purely on {selectedDescriptor === 'all' ? 'color, texture & shape' : selectedDescriptor} descriptor values</p>
            </div>
            
            {searchResults.length === 0 ? (
              <div className="text-center py-12">
                <p className="text-slate-500">No similar objects found.</p>
                <p className="text-sm text-slate-400 mt-2">Try using a different descriptor type.</p>
              </div>
            ) : (
              <div className="space-y-4">
                {searchResults.map((result) => (
                  <div
                    key={`${result.imageId}-${result.objectId}`}
                    className="bg-white border border-slate-200 rounded-lg p-4 hover:shadow-lg transition-shadow cursor-pointer"
                    onClick={() => navigate(`/descriptors?imageId=${result.imageId}&objectId=${result.objectId}`)}
                  >
                    <div className="flex gap-4">
                      {/* Image Thumbnail */}
                      <div className="relative w-32 h-24 flex-shrink-0">
                        <img
                          src={result.image?.url}
                          alt={result.image?.filename}
                          className="w-full h-full object-cover rounded-lg"
                        />
                        <div className="absolute top-1 left-1">
                          <Badge variant="primary" className="text-xs">#{result.rank}</Badge>
                        </div>
                      </div>
                      
                      {/* Info & Scores */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-start justify-between mb-2">
                          <div>
                            <h5 className="font-medium text-slate-900 truncate">{result.image?.filename}</h5>
                            <p className="text-sm text-slate-500 capitalize">Object: {result.object?.label}</p>
                          </div>
                          <Badge variant="success" className="text-lg px-3 py-1">
                            {(result.similarity * 100).toFixed(1)}%
                          </Badge>
                        </div>
                        
                        {/* Detailed Similarity Scores */}
                        <div className="grid grid-cols-3 gap-2 mt-3">
                          {/* Color Score */}
                          <div className="bg-pink-50 rounded-lg p-2 text-center">
                            <div className="flex items-center justify-center gap-1 mb-1">
                              <span className="text-pink-500">🎨</span>
                              <span className="text-xs font-medium text-pink-700">Color</span>
                            </div>
                            <p className="text-lg font-bold text-pink-600">
                              {result.scores && result.scores.color != null ? `${(result.scores.color * 100).toFixed(1)}%` : 'N/A'}
                            </p>
                          </div>
                          
                          {/* Texture Score */}
                          <div className="bg-amber-50 rounded-lg p-2 text-center">
                            <div className="flex items-center justify-center gap-1 mb-1">
                              <span className="text-amber-500">🧱</span>
                              <span className="text-xs font-medium text-amber-700">Texture</span>
                            </div>
                            <p className="text-lg font-bold text-amber-600">
                              {result.scores && result.scores.texture != null ? `${(result.scores.texture * 100).toFixed(1)}%` : 'N/A'}
                            </p>
                          </div>
                          
                          {/* Shape Score */}
                          <div className="bg-blue-50 rounded-lg p-2 text-center">
                            <div className="flex items-center justify-center gap-1 mb-1">
                              <span className="text-blue-500">📐</span>
                              <span className="text-xs font-medium text-blue-700">Shape</span>
                            </div>
                            <p className="text-lg font-bold text-blue-600">
                              {result.scores && result.scores.shape != null ? `${(result.scores.shape * 100).toFixed(1)}%` : 'N/A'}
                            </p>
                          </div>
                        </div>
                        
                        {/* Weighted Formula */}
                        <div className="mt-2 text-xs text-slate-400 bg-slate-50 rounded px-2 py-1">
                          Total = {selectedDescriptor === 'all' ? '(Color×0.4 + Texture×0.3 + Shape×0.3)' : `${selectedDescriptor} × 1.0`}
                          {' = '}
                          <span className="font-medium text-slate-600">{(result.similarity * 100).toFixed(2)}%</span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
      
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
