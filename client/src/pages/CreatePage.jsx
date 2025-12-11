import React, { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { ImageTransformer, TransformControls } from '../components/transform';
import { Button, Dropdown, Toast, Spinner, Badge } from '../components/ui';
import { imageService, transformService } from '../services';

const CreatePage = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  
  const [images, setImages] = useState([]);
  const [selectedImageId, setSelectedImageId] = useState(searchParams.get('imageId') || '');
  const [selectedImage, setSelectedImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [transformations, setTransformations] = useState({
    scale: 100,
    rotate: 0,
    flipH: false,
    flipV: false,
  });
  const [applying, setApplying] = useState(false);
  const [toast, setToast] = useState({ visible: false, message: '', type: 'info' });
  
  useEffect(() => {
    loadImages();
  }, []);
  
  useEffect(() => {
    if (selectedImageId) {
      loadImageDetails();
    }
  }, [selectedImageId]);
  
  const loadImages = async () => {
    try {
      setLoading(true);
      // Load all images from gallery (uploaded, edited, and categories)
      const [uploadedRes, editedRes, categoriesRes] = await Promise.all([
        imageService.getImages({ category: 'uploaded', limit: 100 }),
        imageService.getImages({ category: 'edited', limit: 100 }),
        imageService.getCategoryImages(1, 100)
      ]);
      
      const allImages = [
        ...uploadedRes.images.map(img => ({ ...img, source: 'uploaded' })),
        ...editedRes.images.map(img => ({ ...img, source: 'edited' })),
        ...categoriesRes.images.map(img => ({ ...img, source: 'gallery' }))
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
    } catch (error) {
      showToast('Failed to load image', 'error');
    }
  };
  
  const handleTransformationChange = (newTransformations) => {
    setTransformations(newTransformations);
  };
  
  const handleReset = () => {
    setTransformations({
      scale: 100,
      rotate: 0,
      flipH: false,
      flipV: false,
    });
  };
  
  const handleApply = async () => {
    // Ask user for image name
    const imageName = prompt('Enter a name for the transformed image:');
    
    if (!imageName || imageName.trim() === '') {
      showToast('Please provide a name for the image', 'error');
      return;
    }
    
    try {
      setApplying(true);
      showToast('Applying transformations...', 'info');
      
      const result = await transformService.applyTransformation(
        selectedImageId,
        transformations,
        imageName.trim()
      );
      
      showToast('Transformation applied successfully!', 'success');
      
      // Redirect to collection page
      setTimeout(() => {
        navigate('/');
      }, 1500);
    } catch (error) {
      showToast('Failed to apply transformations', 'error');
    } finally {
      setApplying(false);
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
    label: `${img.filename} (${img.source})`,
  }));
  
  const hasTransformations = 
    transformations.scale !== 100 ||
    transformations.rotate !== 0 ||
    transformations.flipH ||
    transformations.flipV;
  
  return (
    <div>
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-semibold text-slate-900">Create Transformation</h1>
        <p className="text-slate-600 mt-1">
          Apply transformations to any image from your gallery
        </p>
      </div>
      
      {/* Image Selector - Gallery View */}
      <div className="bg-white border border-slate-200 rounded-lg p-6 mb-6">
        <h3 className="text-lg font-semibold text-slate-900 mb-4">Select an Image to Transform</h3>
        
        {loading ? (
          <div className="flex justify-center py-12">
            <Spinner size="lg" />
          </div>
        ) : images.length === 0 ? (
          <div className="text-center py-12">
            <p className="text-slate-500">No images found in gallery.</p>
          </div>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6 gap-4">
            {images.map((image) => (
              <div
                key={image.id}
                onClick={() => setSelectedImageId(image.id.toString())}
                className={`cursor-pointer rounded-lg border-2 overflow-hidden transition-all hover:shadow-lg ${
                  selectedImageId === image.id.toString() 
                    ? 'border-indigo-500 ring-2 ring-indigo-200' 
                    : 'border-slate-200'
                }`}
              >
                <div className="aspect-square relative">
                  <img src={image.url} alt={image.filename} className="w-full h-full object-cover" />
                  <div className="absolute top-1 right-1">
                    <Badge 
                      variant={image.source === 'uploaded' ? 'primary' : image.source === 'edited' ? 'success' : 'default'}
                      className="text-xs"
                    >
                      {image.source}
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
      
      {selectedImage ? (
        <>
          {/* Main Content */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
            {/* Left: Transform Controls */}
            <div>
              <TransformControls
                transformations={transformations}
                onTransformationChange={handleTransformationChange}
                onReset={handleReset}
              />
            </div>
            
            {/* Right: Preview */}
            <div className="lg:col-span-2">
              <ImageTransformer
                image={selectedImage}
                transformations={transformations}
                onTransformationChange={handleTransformationChange}
              />
            </div>
          </div>
          
          {/* Action Buttons */}
          <div className="flex items-center justify-between bg-white border border-slate-200 rounded-lg p-6">
            <div className="text-sm text-slate-600">
              {hasTransformations ? (
                <span className="flex items-center">
                  <svg className="w-4 h-4 text-indigo-500 mr-2" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                  </svg>
                  Transformations will be saved as a new image
                </span>
              ) : (
                'No transformations applied yet'
              )}
            </div>
            
            <div className="flex gap-2">
              <Button
                variant="secondary"
                onClick={handleReset}
                disabled={!hasTransformations || applying}
              >
                Reset
              </Button>
              <Button
                variant="primary"
                onClick={handleApply}
                disabled={!hasTransformations || applying}
              >
                {applying ? (
                  <>
                    <Spinner size="sm" className="mr-2" />
                    Applying...
                  </>
                ) : (
                  <>
                    <svg className="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                    Apply & Save as New
                  </>
                )}
              </Button>
            </div>
          </div>
        </>
      ) : (
        <div className="text-center py-16">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-slate-100 rounded-full mb-4">
            <svg className="w-8 h-8 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
          </div>
          <h3 className="text-lg font-medium text-slate-900 mb-2">No image selected</h3>
          <p className="text-slate-500">Select an image above to start applying transformations</p>
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

export default CreatePage;
