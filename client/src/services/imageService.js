import apiClient, { MOCK_MODE } from './api';

// Mock data
const mockImages = [
  {
    id: 1,
    filename: 'city_street.jpg',
    url: 'https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=400',
    uploadDate: '2025-12-01',
    objectCount: 5,
    detections: [
      { id: 1, label: 'car', confidence: 0.95, bbox: [100, 150, 300, 400] },
      { id: 2, label: 'person', confidence: 0.92, bbox: [50, 100, 150, 350] },
      { id: 3, label: 'person', confidence: 0.88, bbox: [400, 120, 480, 380] },
    ],
  },
  {
    id: 2,
    filename: 'beach_scene.jpg',
    url: 'https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=400',
    uploadDate: '2025-12-02',
    objectCount: 3,
    detections: [
      { id: 4, label: 'person', confidence: 0.91, bbox: [200, 180, 280, 420] },
      { id: 5, label: 'umbrella', confidence: 0.87, bbox: [150, 100, 250, 200] },
    ],
  },
  {
    id: 3,
    filename: 'office_desk.jpg',
    url: 'https://images.unsplash.com/photo-1484480974693-6ca0a78fb36b?w=400',
    uploadDate: '2025-12-03',
    objectCount: 4,
    detections: [
      { id: 6, label: 'laptop', confidence: 0.94, bbox: [150, 200, 450, 500] },
      { id: 7, label: 'keyboard', confidence: 0.89, bbox: [180, 320, 420, 480] },
    ],
  },
  {
    id: 4,
    filename: 'park_dogs.jpg',
    url: 'https://images.unsplash.com/photo-1548199973-03cce0bbc87b?w=400',
    uploadDate: '2025-12-04',
    objectCount: 2,
    detections: [
      { id: 8, label: 'dog', confidence: 0.96, bbox: [100, 200, 300, 500] },
      { id: 9, label: 'dog', confidence: 0.93, bbox: [350, 220, 550, 520] },
    ],
  },
];

const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// Image Service
export const imageService = {
  // Get all images
  async getImages(filters = {}) {
    if (MOCK_MODE) {
      await delay(500);
      let filtered = [...mockImages];
      
      if (filters.search) {
        filtered = filtered.filter(img => 
          img.filename.toLowerCase().includes(filters.search.toLowerCase())
        );
      }
      
      if (filters.category) {
        filtered = filtered.filter(img =>
          img.detections?.some(d => d.label === filters.category)
        );
      }
      
      if (filters.hasDetections) {
        filtered = filtered.filter(img => img.objectCount > 0);
      }
      
      return { images: filtered, total: filtered.length };
    }
    
    const queryParams = new URLSearchParams(filters).toString();
    return apiClient.get(`/images?${queryParams}`);
  },

  // Get single image
  async getImage(id) {
    if (MOCK_MODE) {
      await delay(300);
      const image = mockImages.find(img => img.id === parseInt(id));
      if (!image) throw new Error('Image not found');
      return image;
    }
    
    return apiClient.get(`/images/${id}`);
  },

  // Upload images
  async uploadImages(files, onProgress) {
    if (MOCK_MODE) {
      await delay(1500);
      const newImages = Array.from(files).map((file, index) => ({
        id: mockImages.length + index + 1,
        filename: file.name,
        url: URL.createObjectURL(file),
        uploadDate: new Date().toISOString().split('T')[0],
        objectCount: 0,
        detections: [],
      }));
      mockImages.push(...newImages);
      return { images: newImages, success: true };
    }
    
    const formData = new FormData();
    Array.from(files).forEach(file => {
      formData.append('images', file);
    });
    
    return apiClient.upload('/images/upload', formData);
  },

  // Delete image
  async deleteImage(id) {
    if (MOCK_MODE) {
      await delay(500);
      const index = mockImages.findIndex(img => img.id === id);
      if (index > -1) {
        mockImages.splice(index, 1);
      }
      return { success: true };
    }
    
    return apiClient.delete(`/images/${id}`);
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
