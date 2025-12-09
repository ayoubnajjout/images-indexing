import mongoose from 'mongoose';

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
  }
});

export default mongoose.model('Image', imageSchema);
