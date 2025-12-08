import apiClient, { MOCK_MODE } from './api';

const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// Mock descriptors data
const mockDescriptors = {
  color: {
    histogram: [0.15, 0.22, 0.18, 0.12, 0.08, 0.10, 0.15],
    dominantColors: [
      { color: '#3B82F6', percentage: 35, name: 'Blue' },
      { color: '#10B981', percentage: 28, name: 'Green' },
      { color: '#F59E0B', percentage: 20, name: 'Orange' },
      { color: '#6B7280', percentage: 17, name: 'Gray' },
    ],
  },
  texture: {
    tamura: {
      coarseness: 0.67,
      contrast: 0.82,
      directionality: 0.45,
    },
    gabor: [0.23, 0.45, 0.67, 0.12, 0.89, 0.34, 0.56, 0.78],
  },
  shape: {
    huMoments: [0.12, 0.34, 0.56, 0.78, 0.23, 0.45, 0.67],
    orientationHistogram: [0.1, 0.15, 0.2, 0.25, 0.15, 0.1, 0.05],
  },
};

// Descriptor Service
export const descriptorService = {
  // Get descriptors for an object in an image
  async getDescriptors(imageId, objectId) {
    if (MOCK_MODE) {
      await delay(800);
      return {
        imageId,
        objectId,
        descriptors: mockDescriptors,
      };
    }
    
    return apiClient.get(`/images/${imageId}/objects/${objectId}/descriptors`);
  },

  // Compute descriptors (if not already computed)
  async computeDescriptors(imageId, objectId) {
    if (MOCK_MODE) {
      await delay(1500);
      return {
        imageId,
        objectId,
        descriptors: mockDescriptors,
        success: true,
      };
    }
    
    return apiClient.post(`/images/${imageId}/objects/${objectId}/descriptors`, {});
  },
};

export default descriptorService;
