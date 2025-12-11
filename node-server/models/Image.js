import mongoose from 'mongoose';

const detectionSchema = new mongoose.Schema({
  id: Number,
  label: String,
  confidence: Number,
  bbox: [Number], // [x1, y1, x2, y2]
  descriptors: {
    color: {
      histogram: [Number],
      dominantColors: [{
        color: String,
        percentage: Number,
        name: String
      }]
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
      orientationHistogram: [Number]
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
