/**
 * 3D Model Search Service
 * Handles all API interactions for 3D model indexing and search
 */

import apiClient from './api';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api';

export const model3DService = {
  /**
   * Get all 3D model categories
   */
  async getCategories() {
    try {
      return await apiClient.get('/3d/categories');
    } catch (error) {
      console.error('Failed to get 3D categories:', error);
      throw error;
    }
  },

  /**
   * Get 3D models with optional filtering
   * @param {Object} params - Query parameters
   * @param {string} params.category - Filter by category
   * @param {number} params.page - Page number
   * @param {number} params.limit - Items per page
   * @param {boolean} params.indexed - Only return indexed models
   */
  async getModels(params = {}) {
    try {
      const queryParams = new URLSearchParams();
      if (params.category) queryParams.set('category', params.category);
      if (params.page) queryParams.set('page', params.page);
      if (params.limit) queryParams.set('limit', params.limit);
      if (params.indexed !== undefined) queryParams.set('indexed', params.indexed);
      
      const query = queryParams.toString();
      return await apiClient.get(`/3d/models${query ? `?${query}` : ''}`);
    } catch (error) {
      console.error('Failed to get 3D models:', error);
      throw error;
    }
  },

  /**
   * Get a single 3D model by ID
   * @param {string} id - Model ID
   */
  async getModel(id) {
    try {
      return await apiClient.get(`/3d/models/${id}`);
    } catch (error) {
      console.error('Failed to get 3D model:', error);
      throw error;
    }
  },

  /**
   * Extract descriptors from an uploaded 3D model file
   * @param {File} file - The .obj file
   */
  async extractDescriptors(file) {
    try {
      const formData = new FormData();
      formData.append('model', file);
      
      return await apiClient.upload('/3d/descriptors', formData);
    } catch (error) {
      console.error('Failed to extract 3D descriptors:', error);
      throw error;
    }
  },

  /**
   * Search for similar 3D models by uploading a query model
   * @param {File} file - The .obj file to use as query
   * @param {Object} options - Search options
   * @param {string} options.category - Filter by category
   * @param {number} options.limit - Maximum results
   */
  async searchByUpload(file, options = {}) {
    try {
      const formData = new FormData();
      formData.append('model', file);
      if (options.category) formData.append('category', options.category);
      if (options.limit) formData.append('limit', options.limit);
      
      return await apiClient.upload('/3d/search', formData);
    } catch (error) {
      console.error('Failed to search 3D models:', error);
      throw error;
    }
  },

  /**
   * Search for similar 3D models using an existing model's path
   * @param {string} modelPath - Path to the query model
   * @param {Object} options - Search options
   */
  async searchByPath(modelPath, options = {}) {
    try {
      const response = await fetch(`${API_BASE_URL}/3d/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          modelPath,
          category: options.category,
          limit: options.limit || 10
        })
      });
      
      if (!response.ok) {
        throw new Error(`Search failed: ${response.statusText}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Failed to search 3D models:', error);
      throw error;
    }
  },

  /**
   * Search for similar models using database (pre-indexed models)
   * @param {string} modelId - ID of the query model
   * @param {Object} options - Search options
   */
  async searchByModelId(modelId, options = {}) {
    try {
      return await apiClient.post('/3d/search/db', {
        modelId,
        category: options.category,
        limit: options.limit || 10,
        weights: options.weights
      });
    } catch (error) {
      console.error('Failed to search 3D models:', error);
      throw error;
    }
  },

  /**
   * Index a 3D model (compute and store descriptors)
   * @param {Object} params - Model info
   * @param {string} params.category - Model category
   * @param {string} params.filename - Model filename
   * @param {string} params.filepath - Full file path (optional)
   */
  async indexModel(params) {
    try {
      return await apiClient.post('/3d/index', params);
    } catch (error) {
      console.error('Failed to index 3D model:', error);
      throw error;
    }
  },

  /**
   * Index all models in a category
   * @param {string} category - Category name
   */
  async indexCategory(category, batchSize = 3) {
    try {
      return await apiClient.post('/3d/index/category', { category, batchSize });
    } catch (error) {
      console.error('Failed to index category:', error);
      throw error;
    }
  },

  /**
   * Index all models from all categories with SSE progress streaming
   * @param {number} batchSize - Number of models to process in parallel
   * @param {function} onProgress - Callback for progress updates
   * @returns {Promise} Resolves when indexing is complete
   */
  indexAllCategoriesWithProgress(batchSize = 5, onProgress) {
    return new Promise((resolve, reject) => {
      const baseUrl = import.meta.env.VITE_API_URL || 'http://localhost:3000/api';
      const url = `${baseUrl}/3d/index/all/stream?batchSize=${batchSize}`;
      
      const eventSource = new EventSource(url);
      let finalResult = null;
      
      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          switch (data.type) {
            case 'status':
              if (onProgress) onProgress({ status: data.message, current: 0, total: 0 });
              break;
            case 'init':
              if (onProgress) onProgress({ 
                status: `Found ${data.total} models (${data.skipped} already indexed)`,
                current: data.skipped,
                total: data.total,
                toIndex: data.toIndex
              });
              break;
            case 'progress':
              if (onProgress) onProgress({
                current: data.current,
                total: data.total,
                processed: data.processed,
                toIndex: data.toIndex,
                newlyIndexed: data.newlyIndexed,
                failed: data.failed,
                currentFile: data.currentFile,
                category: data.category,
                status: `Indexing: ${data.currentFile}`
              });
              break;
            case 'complete':
              finalResult = data;
              eventSource.close();
              resolve({ success: true, ...data });
              break;
            case 'error':
              eventSource.close();
              reject(new Error(data.message));
              break;
          }
        } catch (err) {
          console.error('Error parsing SSE data:', err);
        }
      };
      
      eventSource.onerror = (err) => {
        eventSource.close();
        if (!finalResult) {
          reject(new Error('Connection lost during indexing'));
        }
      };
    });
  },

  /**
   * Index all models (non-streaming fallback)
   * @param {number} batchSize - Number of models to process in parallel (default: 3)
   */
  async indexAllCategories(batchSize = 3) {
    try {
      return await apiClient.post('/3d/index/all', { batchSize });
    } catch (error) {
      console.error('Failed to index all categories:', error);
      throw error;
    }
  },

  /**
   * Get information about available 3D shape descriptors
   */
  async getDescriptorInfo() {
    try {
      return await apiClient.get('/3d/descriptor-info');
    } catch (error) {
      console.error('Failed to get descriptor info:', error);
      throw error;
    }
  },

  /**
   * Get full descriptors for a specific model (for visualization)
   * @param {string} modelId - Model ID
   */
  async getModelDescriptors(modelId) {
    try {
      return await apiClient.get(`/3d/models/${modelId}/descriptors`);
    } catch (error) {
      console.error('Failed to get model descriptors:', error);
      throw error;
    }
  },

  /**
   * Compare descriptors between two models
   * @param {string} modelId1 - First model ID
   * @param {string} modelId2 - Second model ID
   */
  async compareDescriptors(modelId1, modelId2) {
    try {
      return await apiClient.post('/3d/compare-descriptors', { modelId1, modelId2 });
    } catch (error) {
      console.error('Failed to compare descriptors:', error);
      throw error;
    }
  },

  /**
   * Get indexing statistics
   */
  async getStats() {
    try {
      return await apiClient.get('/3d/stats');
    } catch (error) {
      console.error('Failed to get 3D stats:', error);
      throw error;
    }
  },

  /**
   * Build full URL for model file
   * @param {string} modelUrl - Relative model URL
   */
  getModelFileUrl(modelUrl) {
    if (!modelUrl) return null;
    const baseUrl = API_BASE_URL.replace('/api', '');
    return `${baseUrl}${modelUrl}`;
  },

  /**
   * Build full URL for thumbnail
   * @param {string} thumbnailUrl - Relative thumbnail URL
   */
  getThumbnailUrl(thumbnailUrl) {
    if (!thumbnailUrl) return null;
    const baseUrl = API_BASE_URL.replace('/api', '');
    return `${baseUrl}${thumbnailUrl}`;
  }
};

export default model3DService;
