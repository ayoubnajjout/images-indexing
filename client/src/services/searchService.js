import apiClient, { MOCK_MODE } from './api';

const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// Mock search results
const mockSearchResults = [
  {
    rank: 1,
    imageId: 2,
    objectId: 4,
    similarity: 0.95,
    image: {
      id: 2,
      filename: 'beach_scene.jpg',
      url: 'https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=400',
    },
    object: {
      id: 4,
      label: 'person',
      confidence: 0.91,
      bbox: [200, 180, 280, 420],
    },
  },
  {
    rank: 2,
    imageId: 1,
    objectId: 2,
    similarity: 0.89,
    image: {
      id: 1,
      filename: 'city_street.jpg',
      url: 'https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=400',
    },
    object: {
      id: 2,
      label: 'person',
      confidence: 0.92,
      bbox: [50, 100, 150, 350],
    },
  },
  {
    rank: 3,
    imageId: 1,
    objectId: 3,
    similarity: 0.82,
    image: {
      id: 1,
      filename: 'city_street.jpg',
      url: 'https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=400',
    },
    object: {
      id: 3,
      label: 'person',
      confidence: 0.88,
      bbox: [400, 120, 480, 380],
    },
  },
];

// Search Service
export const searchService = {
  // Perform similarity search
  async searchSimilar(queryImageId, queryObjectId, options = {}) {
    if (MOCK_MODE) {
      await delay(1200);
      
      const { topK = 10, threshold = 0.0 } = options;
      
      let results = [...mockSearchResults];
      
      // Filter by threshold
      results = results.filter(r => r.similarity >= threshold);
      
      // Limit to topK
      results = results.slice(0, topK);
      
      return {
        query: { imageId: queryImageId, objectId: queryObjectId },
        results,
        total: results.length,
      };
    }
    
    return apiClient.post('/search/similar', {
      queryImageId,
      queryObjectId,
      ...options,
    });
  },

  // Get search history (optional)
  async getSearchHistory() {
    if (MOCK_MODE) {
      await delay(500);
      return {
        history: [
          {
            id: 1,
            queryImageId: 1,
            queryObjectId: 2,
            timestamp: '2025-12-08T10:30:00',
            resultCount: 3,
          },
        ],
      };
    }
    
    return apiClient.get('/search/history');
  },
};

export default searchService;
