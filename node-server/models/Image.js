import mongoose from 'mongoose';

const detectionSchema = new mongoose.Schema({
  id: Number,
  label: String,           // ImageNet synset ID (e.g., 'n01440764')
  label_readable: String,  // Human-readable name (e.g., 'tench')
  class_id: Number,        // Numeric class ID (0-14)
  confidence: Number,
  bbox: [Number], // [x1, y1, x2, y2]
  descriptors: {
    color: {
      histogram: [Number],
      hsHistogram: [Number],        // 2D Hue-Saturation histogram
      dominantColors: [{
        color: String,
        percentage: Number,
        name: String
      }],
      dominantColorsHS: [[Number]], // K-means clusters in HS space
      labColorMoments: [Number]     // LAB color moments (mean, var, skew)
    },
    texture: {
      tamura: {
        coarseness: Number,
        contrast: Number,
        directionality: Number
      },
      gabor: [Number]
    },
    shape: {
      huMoments: [Number],
      contourHuMoments: [Number],           // Hu moments from contour
      orientationHistogram: [Number],
      orientationHistogramContour: [Number], // Contour-based orientation histogram
      shapeMetrics: {
        solidity: Number,
        aspectRatio: Number,
        compactness: Number
      },
      fourierDescriptors: [Number]          // Fourier descriptors
    }
  }
}, { _id: false });

const imageSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true
  },
  originalName: {
    type: String,
    required: true
  },
  filename: {
    type: String,
    required: true
  },
  path: {
    type: String,
    required: true
  },
  category: {
    type: String,
    enum: ['uploaded', 'edited', 'categories'],
    default: 'uploaded',
    index: true
  },
  size: {
    type: Number
  },
  mimetype: {
    type: String
  },
  uploadedAt: {
    type: Date,
    default: Date.now
  },
  detections: [detectionSchema],
  objectCount: {
    type: Number,
    default: 0
  }
});

export default mongoose.model('Image', imageSchema);
