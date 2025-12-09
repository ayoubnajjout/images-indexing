import apiClient from './api';

// Transform Service
export const transformService = {
  // Apply transformations to an image and save with custom name
  async applyTransformation(imageId, transformations, imageName) {
    try {
      // Get the canvas element with the transformed image
      const canvas = document.querySelector('canvas');
      if (!canvas) {
        throw new Error('No canvas found');
      }
      
      // Get the image data from canvas
      const imageData = canvas.toDataURL('image/png');
      
      // Save to backend
      const response = await apiClient.post('/images/save-edited', {
        imageData,
        name: imageName,
        originalId: imageId,
      });
      
      return {
        success: true,
        newImageId: response.image._id,
        newImage: response.image,
      };
    } catch (error) {
      console.error('Failed to apply transformation:', error);
      throw error;
    }
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
