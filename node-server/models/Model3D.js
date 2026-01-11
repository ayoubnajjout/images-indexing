import mongoose from 'mongoose';

/**
 * Schema for storing 3D shape descriptors
 * Based on Shape Distribution descriptors (Osada et al.) and geometric features
 */
const descriptorsSchema = new mongoose.Schema({
  // Shape Distribution descriptors (histograms)
  d1: [Number],  // Distance from centroid to random surface points
  d2: [Number],  // Distance between pairs of random surface points (most discriminative)
  d3: [Number],  // Sqrt(area) of triangles from 3 random points
  d4: [Number],  // Cbrt(volume) of tetrahedra from 4 random points
  a3: [Number],  // Angles between 3 random surface points
  
  // Geometric descriptors
  bbox: [Number],       // Bounding box features
  moments: [Number],    // 3D moment invariants
  mesh_stats: [Number], // Mesh statistics
  
  // Multi-view descriptors (OpenGL rendered)
  multiview_histogram: [Number],
  multiview_silhouette: [Number],
  multiview_fourier: [Number]
}, { _id: false });

/**
 * Schema for 3D models with their computed descriptors
 */
const model3DSchema = new mongoose.Schema({
  // File information
  name: {
    type: String,
    required: true,
    index: true
  },
  filename: {
    type: String,
    required: true
  },
  filepath: {
    type: String,
    required: true,
    unique: true
  },
  category: {
    type: String,
    required: true,
    index: true
  },
  
  // Thumbnail path (if available)
  thumbnailPath: {
    type: String
  },
  
  // Model statistics
  numVertices: {
    type: Number
  },
  numFaces: {
    type: Number
  },
  surfaceArea: {
    type: Number
  },
  
  // Shape descriptors
  descriptors: descriptorsSchema,
  
  // Combined descriptor for fast similarity search (flattened array)
  combinedDescriptor: [Number],
  
  // Metadata
  processedAt: {
    type: Date,
    default: Date.now
  },
  processingError: {
    type: String
  },
  isIndexed: {
    type: Boolean,
    default: false,
    index: true
  }
}, {
  timestamps: true
});

// Indexes for efficient querying
model3DSchema.index({ category: 1, isIndexed: 1 });
model3DSchema.index({ name: 'text', filename: 'text' });

/**
 * Static method to find similar models using pre-computed descriptors
 * @param {Array} queryDescriptor - The combined descriptor to search for
 * @param {Object} options - Search options
 * @returns {Promise<Array>} Array of similar models with similarity scores
 */
model3DSchema.statics.findSimilar = async function(queryDescriptor, options = {}) {
  const {
    limit = 10,
    category = null,
    excludeId = null,
    weights = null
  } = options;
  
  // Build query
  const query = { isIndexed: true };
  if (category) {
    query.category = category;
  }
  if (excludeId) {
    query._id = { $ne: excludeId };
  }
  
  // Get all indexed models
  const models = await this.find(query).lean();
  
  if (models.length === 0) {
    return [];
  }
  
  // Default weights for different descriptor types
  const defaultWeights = {
    d1: 0.1,
    d2: 0.25,  // D2 is most discriminative
    d3: 0.15,
    d4: 0.1,
    a3: 0.15,
    bbox: 0.05,
    moments: 0.1,
    mesh_stats: 0.05,
    multiview_histogram: 0.025,
    multiview_silhouette: 0.025,
    multiview_fourier: 0.0
  };
  
  const w = weights || defaultWeights;
  
  // Compute similarities
  const similarities = models.map(model => {
    let totalSim = 0;
    let totalWeight = 0;
    
    // Compare each descriptor type
    for (const [descName, weight] of Object.entries(w)) {
      if (weight <= 0) continue;
      
      const arr1 = queryDescriptor[descName];
      const arr2 = model.descriptors ? model.descriptors[descName] : null;
      
      if (!arr1 || !arr2 || arr1.length === 0 || arr2.length === 0) continue;
      if (arr1.length !== arr2.length) continue;
      
      let sim;
      if (['d1', 'd2', 'd3', 'd4', 'a3', 'multiview_histogram'].includes(descName)) {
        // Histogram intersection for distribution descriptors
        sim = arr1.reduce((sum, val, i) => sum + Math.min(val, arr2[i]), 0);
      } else {
        // Cosine similarity for other descriptors
        const dot = arr1.reduce((sum, val, i) => sum + val * arr2[i], 0);
        const norm1 = Math.sqrt(arr1.reduce((sum, val) => sum + val * val, 0));
        const norm2 = Math.sqrt(arr2.reduce((sum, val) => sum + val * val, 0));
        
        if (norm1 > 1e-10 && norm2 > 1e-10) {
          sim = (dot / (norm1 * norm2) + 1) / 2;  // Map from [-1,1] to [0,1]
        } else {
          sim = 0;
        }
      }
      
      totalSim += weight * sim;
      totalWeight += weight;
    }
    
    const similarity = totalWeight > 0 ? totalSim / totalWeight : 0;
    
    return {
      ...model,
      similarity
    };
  });
  
  // Sort by similarity (descending) and limit
  similarities.sort((a, b) => b.similarity - a.similarity);
  
  return similarities.slice(0, limit);
};

/**
 * Static method to compute Euclidean distance between combined descriptors
 */
model3DSchema.statics.euclideanDistance = function(desc1, desc2) {
  if (!desc1 || !desc2 || desc1.length !== desc2.length) {
    return Infinity;
  }
  
  let sum = 0;
  for (let i = 0; i < desc1.length; i++) {
    const diff = desc1[i] - desc2[i];
    sum += diff * diff;
  }
  
  return Math.sqrt(sum);
};

const Model3D = mongoose.model('Model3D', model3DSchema);

export default Model3D;
