import apiClient from './api';

// Descriptor Service
export const descriptorService = {
  // Get descriptors for an object in an image
  async getDescriptors(imageId, objectId) {
    try {
      const response = await apiClient.get(`/images/${imageId}/objects/${objectId}/descriptors`);
      return {
        imageId,
        objectId,
        descriptors: response.descriptors
      };
    } catch (error) {
      console.error('Failed to get descriptors:', error);
      throw error;
    }
  },

  // Compute descriptors (if not already computed)
  async computeDescriptors(imageId, objectId) {
    try {
      const response = await apiClient.post(`/images/${imageId}/objects/${objectId}/descriptors`, {});
      return {
        imageId,
        objectId,
        descriptors: response.descriptors,
        success: response.success
      };
    } catch (error) {
      console.error('Failed to compute descriptors:', error);
      throw error;
    }
  },
};

export default descriptorService;
