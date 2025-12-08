import apiClient, { MOCK_MODE } from './api';

const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// Transform Service
export const transformService = {
  // Apply transformations to an image
  async applyTransformation(imageId, transformations) {
    if (MOCK_MODE) {
      await delay(1500);
      
      return {
        success: true,
        newImageId: Date.now(),
        newImage: {
          id: Date.now(),
          filename: `transformed_${Date.now()}.jpg`,
          url: 'https://images.unsplash.com/photo-1501594907352-04cda38ebc29?w=400',
          uploadDate: new Date().toISOString().split('T')[0],
          objectCount: 0,
          transformations,
        },
      };
    }
    
    return apiClient.post(`/images/${imageId}/transform`, {
      transformations,
    });
  },

  // Get available transformation types
  getTransformationTypes() {
    return [
      { value: 'crop', label: 'Crop', icon: '✂️' },
      { value: 'resize', label: 'Resize', icon: '📏' },
      { value: 'rotate', label: 'Rotate', icon: '🔄' },
      { value: 'flip', label: 'Flip', icon: '🔀' },
    ];
  },
};

export default transformService;
