import apiClient from './api';

// Search Service
export const searchService = {
  // Perform similarity search
  async searchSimilar(queryImageId, queryObjectId, options = {}) {
    try {
      const { topK = 10, threshold = 0.0, weights } = options;
      
      const requestBody = {
        queryImageId,
        queryObjectId,
        topK,
        threshold
      };
      
      // Add descriptor weights if specified
      if (weights) {
        requestBody.weights = weights;
      }
      
      const response = await apiClient.post('/search/similar', requestBody);
      
      // Filter by threshold if needed
      let results = response.results || [];
      if (threshold > 0) {
        results = results.filter(r => r.similarity >= threshold);
      }
      
      return {
        query: response.query,
        results,
        total: results.length
      };
    } catch (error) {
      console.error('Search failed:', error);
      throw error;
    }
  },

  // Get search history (optional - placeholder)
  async getSearchHistory() {
    return { history: [] };
  },
};

export default searchService;
