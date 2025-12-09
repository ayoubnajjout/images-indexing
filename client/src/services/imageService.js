import apiClient from './api';

// Image Service
export const imageService = {
  // Get all images with optional category filter and pagination
  async getImages(filters = {}) {
    try {
      const queryParams = new URLSearchParams();
      if (filters.category) {
        queryParams.append('category', filters.category);
      }
      if (filters.page) {
        queryParams.append('page', filters.page);
      }
      if (filters.limit) {
        queryParams.append('limit', filters.limit);
      }
      
      const response = await apiClient.get(`/images?${queryParams}`);
      
      // Map response to include proper URLs from server
      const images = response.images.map(img => ({
        ...img,
        id: img._id,
        url: img.url,
        filename: img.name || img.filename,
        uploadDate: img.uploadedAt ? new Date(img.uploadedAt).toISOString().split('T')[0] : null,
        objectCount: 0
      }));
      
      return { 
        images, 
        total: response.total,
        page: response.page,
        totalPages: response.totalPages,
        hasMore: response.hasMore
      };
    } catch (error) {
      console.error('Failed to fetch images:', error);
      return { images: [], total: 0, hasMore: false };
    }
  },

  // Get single image
  async getImage(id) {
    try {
      const response = await apiClient.get(`/images/${id}`);
      return {
        ...response,
        id: response._id,
        url: response.url,
        filename: response.name || response.filename,
        uploadDate: response.uploadedAt ? new Date(response.uploadedAt).toISOString().split('T')[0] : null,
        objectCount: 0
      };
    } catch (error) {
      console.error('Failed to fetch image:', error);
      throw error;
    }
  },

  // Upload single image
  async uploadImage(file) {
    try {
      const formData = new FormData();
      formData.append('image', file);
      
      const response = await apiClient.upload('/images/upload', formData);
      
      return {
        ...response.image,
        id: response.image._id,
        url: response.image.url,
        filename: response.image.name || response.image.filename
      };
    } catch (error) {
      console.error('Failed to upload image:', error);
      throw error;
    }
  },

  // Delete image
  async deleteImage(id) {
    try {
      const response = await apiClient.delete(`/images/${id}`);
      return response;
    } catch (error) {
      console.error('Failed to delete image:', error);
      throw error;
    }
  },

  // Get category images (from images folder)
  async getCategoryImages(page = 1, limit = 50) {
    try {
      const queryParams = new URLSearchParams({ page, limit });
      const response = await apiClient.get(`/images/categories/list?${queryParams}`);
      
      const images = response.images.map(img => ({
        ...img,
        url: img.url,
        filename: img.name,
        id: img._id || img.path,
        objectCount: 0
      }));
      
      return {
        images,
        total: response.total,
        hasMore: response.hasMore
      };
    } catch (error) {
      console.error('Failed to fetch category images:', error);
      return { images: [], total: 0, hasMore: false };
    }
  },

  // Run object detection
  async detectObjects(imageId) {
    if (MOCK_MODE) {
      await delay(2000);
      const image = mockImages.find(img => img.id === imageId);
      if (!image) throw new Error('Image not found');
      
      // Simulate detection results
      const detections = [
        { id: Date.now(), label: 'car', confidence: 0.95, bbox: [100, 150, 300, 400] },
        { id: Date.now() + 1, label: 'person', confidence: 0.92, bbox: [50, 100, 150, 350] },
      ];
      
      image.detections = detections;
      image.objectCount = detections.length;
      
      return { detections, success: true };
    }
    
    return apiClient.post(`/images/${imageId}/detect`, {});
  },
};

export default imageService;
