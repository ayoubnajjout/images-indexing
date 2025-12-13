import express from 'express';
import mongoose from 'mongoose';
import cors from 'cors';
import dotenv from 'dotenv';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import Image from './models/Image.js';

dotenv.config();

// Python API URL for YOLO detection and descriptors
const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:8000';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors({
  origin: ['http://localhost:5173', 'http://localhost:3000'],
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization']
}));
// Increase body size limit for base64 image data (50MB)
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// Static file serving with CORS headers
const staticOptions = {
  setHeaders: (res, path) => {
    res.set('Access-Control-Allow-Origin', '*');
    res.set('Access-Control-Allow-Methods', 'GET');
    res.set('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
    res.set('Cross-Origin-Resource-Policy', 'cross-origin');
    res.set('Timing-Allow-Origin', '*');
    res.set('Cache-Control', 'no-cache');
  }
};

app.use('/images', express.static(path.join(__dirname, 'S3', 'images'), staticOptions));
app.use('/upload', express.static(path.join(__dirname, 'S3', 'upload'), staticOptions));
app.use('/edited', express.static(path.join(__dirname, 'S3', 'edited'), staticOptions));

// Ensure S3 directories exist
const s3BaseDir = path.join(__dirname, 'S3');
const dirs = ['images', 'upload', 'edited'];
if (!fs.existsSync(s3BaseDir)) {
  fs.mkdirSync(s3BaseDir, { recursive: true });
}
dirs.forEach(dir => {
  const dirPath = path.join(s3BaseDir, dir);
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }
});

// MongoDB connection
mongoose.connect(process.env.MONGODB_URI)
  .then(() => console.log('MongoDB connected'))
  .catch(err => console.error('MongoDB connection error:', err));

// Multer configuration for uploaded images
const uploadStorage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, path.join(__dirname, 'S3', 'upload'));
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, uniqueSuffix + path.extname(file.originalname));
  }
});

const upload = multer({
  storage: uploadStorage,
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB limit
  fileFilter: (req, file, cb) => {
    const allowedTypes = /jpeg|jpg|png|gif|webp/;
    const extname = allowedTypes.test(path.extname(file.originalname).toLowerCase());
    const mimetype = allowedTypes.test(file.mimetype);
    
    if (mimetype && extname) {
      return cb(null, true);
    } else {
      cb(new Error('Only image files are allowed!'));
    }
  }
});

// Routes

// Upload single image
app.post('/api/images/upload', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image file provided' });
    }

    const imageName = req.body.name || req.file.originalname;
    
    const newImage = new Image({
      name: imageName,
      originalName: req.file.originalname,
      filename: req.file.filename,
      path: `/upload/${req.file.filename}`,
      category: 'uploaded',
      size: req.file.size,
      mimetype: req.file.mimetype
    });

    await newImage.save();

    res.status(201).json({
      message: 'Image uploaded successfully',
      image: {
        ...newImage.toObject(),
        url: `http://localhost:${PORT}${newImage.path}`
      }
    });
  } catch (error) {
    console.error('Upload error:', error);
    res.status(500).json({ error: 'Failed to upload image' });
  }
});

// Get all images with optional filter - returns metadata with file paths (with pagination)
app.get('/api/images', async (req, res) => {
  try {
    const { category, page = 1, limit = 20 } = req.query;
    const filter = category ? { category } : {};
    
    const skip = (parseInt(page) - 1) * parseInt(limit);
    
    const [images, total] = await Promise.all([
      Image.find(filter)
        .sort({ uploadedAt: -1 })
        .skip(skip)
        .limit(parseInt(limit))
        .lean(),
      Image.countDocuments(filter)
    ]);
    
    // Return images with HTTP URLs for browser access
    const imagesWithPaths = images.map(img => ({
      ...img,
      url: `http://localhost:${PORT}${img.path}`
    }));
    
    res.json({
      images: imagesWithPaths,
      total,
      page: parseInt(page),
      totalPages: Math.ceil(total / parseInt(limit)),
      hasMore: skip + images.length < total
    });
  } catch (error) {
    console.error('Fetch error:', error);
    res.status(500).json({ error: 'Failed to fetch images' });
  }
});

// Get single image by ID
app.get('/api/images/:id', async (req, res) => {
  try {
    const image = await Image.findById(req.params.id);
    if (!image) {
      return res.status(404).json({ error: 'Image not found' });
    }
    
    // Add URL for image access
    const imageWithUrl = {
      ...image.toObject(),
      url: `http://localhost:${PORT}${image.path}`
    };
    
    res.json(imageWithUrl);
  } catch (error) {
    console.error('Fetch error:', error);
    res.status(500).json({ error: 'Failed to fetch image' });
  }
});

// Save transformed/edited image
app.post('/api/images/save-edited', async (req, res) => {
  try {
    const { imageData, name, originalId } = req.body;
    
    if (!imageData || !name) {
      return res.status(400).json({ error: 'Image data and name are required' });
    }

    // Remove data URL prefix
    const base64Data = imageData.replace(/^data:image\/\w+;base64,/, '');
    const buffer = Buffer.from(base64Data, 'base64');
    
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    const filename = `${name.replace(/\s+/g, '_')}-${uniqueSuffix}.png`;
    const filepath = path.join(__dirname, 'S3', 'edited', filename);
    
    fs.writeFileSync(filepath, buffer);

    const newImage = new Image({
      name: name,
      originalName: filename,
      filename: filename,
      path: `/edited/${filename}`,
      category: 'edited',
      size: buffer.length,
      mimetype: 'image/png'
    });

    await newImage.save();

    res.status(201).json({
      message: 'Edited image saved successfully',
      image: {
        ...newImage.toObject(),
        url: `http://localhost:${PORT}/edited/${filename}`
      }
    });
  } catch (error) {
    console.error('Save edited error:', error);
    res.status(500).json({ error: 'Failed to save edited image' });
  }
});

// Delete image
app.delete('/api/images/:id', async (req, res) => {
  try {
    const image = await Image.findById(req.params.id);
    
    if (!image) {
      return res.status(404).json({ error: 'Image not found' });
    }

    // Delete file from filesystem
    const filePath = path.join(__dirname, image.path);
    if (fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
    }

    await Image.findByIdAndDelete(req.params.id);

    res.json({ message: 'Image deleted successfully' });
  } catch (error) {
    console.error('Delete error:', error);
    res.status(500).json({ error: 'Failed to delete image' });
  }
});

// Get images from categories folder (existing images from S3) - returns from MongoDB
app.get('/api/images/categories/list', async (req, res) => {
  try {
    const { page = 1, limit = 50 } = req.query;
    const skip = (parseInt(page) - 1) * parseInt(limit);
    
    const [images, total] = await Promise.all([
      Image.find({ category: 'categories' })
        .skip(skip)
        .limit(parseInt(limit))
        .lean(),
      Image.countDocuments({ category: 'categories' })
    ]);
    
    const imagesWithUrls = images.map(img => ({
      ...img,
      url: `http://localhost:${PORT}${img.path}`
    }));

    res.json({
      images: imagesWithUrls,
      total,
      page: parseInt(page),
      totalPages: Math.ceil(total / parseInt(limit)),
      hasMore: skip + images.length < total
    });
  } catch (error) {
    console.error('Category list error:', error);
    res.status(500).json({ error: 'Failed to fetch category images' });
  }
});

// Debug endpoint to test Python API descriptor extraction
app.get('/api/debug/test-descriptors/:id', async (req, res) => {
  try {
    const image = await Image.findById(req.params.id);
    if (!image) {
      return res.status(404).json({ error: 'Image not found' });
    }

    const imageUrl = `http://localhost:${PORT}${image.path}`;
    console.log(`Testing descriptors for: ${imageUrl}`);

    // Call Python API directly
    const response = await fetch(`${PYTHON_API_URL}/detect-and-describe/url`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: imageUrl })
    });

    if (!response.ok) {
      const errorText = await response.text();
      return res.status(500).json({ error: `Python API error: ${errorText}` });
    }

    const result = await response.json();
    
    // Return raw result from Python API for debugging
    res.json({
      imageId: req.params.id,
      imagePath: image.path,
      pythonApiResponse: result,
      currentDbDetections: image.detections
    });
  } catch (error) {
    console.error('Debug test error:', error);
    res.status(500).json({ error: error.message });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`Python API URL: ${PYTHON_API_URL}`);
});

// ==================== DETECTION ROUTES ====================

// Get model info
app.get('/api/model/info', async (req, res) => {
  try {
    const response = await fetch(`${PYTHON_API_URL}/model-info`);
    if (!response.ok) {
      throw new Error('Failed to get model info');
    }
    const result = await response.json();
    res.json(result);
  } catch (error) {
    console.error('Model info error:', error);
    res.status(500).json({ error: 'Failed to get model info' });
  }
});

// Run object detection on an image
app.post('/api/images/:id/detect', async (req, res) => {
  try {
    const image = await Image.findById(req.params.id);
    if (!image) {
      return res.status(404).json({ error: 'Image not found' });
    }

    const imageUrl = `http://localhost:${PORT}${image.path}`;

    // Call Python API for detection
    const response = await fetch(`${PYTHON_API_URL}/detect-and-describe/url`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: imageUrl })
    });

    if (!response.ok) {
      throw new Error('Detection API failed');
    }

    const result = await response.json();

    // Update image with detections (now includes label_readable and class_id)
    image.detections = result.detections;
    image.objectCount = result.count;
    await image.save();

    res.json({
      success: true,
      detections: result.detections,
      count: result.count
    });
  } catch (error) {
    console.error('Detection error:', error);
    res.status(500).json({ error: 'Failed to run detection' });
  }
});

// Get descriptors for a specific object
app.get('/api/images/:imageId/objects/:objectId/descriptors', async (req, res) => {
  try {
    const { imageId, objectId } = req.params;
    
    const image = await Image.findById(imageId);
    if (!image) {
      return res.status(404).json({ error: 'Image not found' });
    }

    // Find the detection
    const detection = image.detections?.find(d => d.id === parseInt(objectId));
    if (!detection) {
      return res.status(404).json({ error: 'Object not found' });
    }

    // If descriptors are already stored, return them
    if (detection.descriptors) {
      return res.json({
        imageId,
        objectId,
        descriptors: detection.descriptors
      });
    }

    // Otherwise, compute descriptors via Python API
    const imageUrl = `http://localhost:${PORT}${image.path}`;
    
    const response = await fetch(`${PYTHON_API_URL}/descriptors/url`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        url: imageUrl,
        bbox: detection.bbox
      })
    });

    if (!response.ok) {
      throw new Error('Descriptor API failed');
    }

    const result = await response.json();

    res.json({
      imageId,
      objectId,
      descriptors: result.descriptors
    });
  } catch (error) {
    console.error('Descriptor error:', error);
    res.status(500).json({ error: 'Failed to get descriptors' });
  }
});

// Compute descriptors for an object
app.post('/api/images/:imageId/objects/:objectId/descriptors', async (req, res) => {
  try {
    const { imageId, objectId } = req.params;
    
    const image = await Image.findById(imageId);
    if (!image) {
      return res.status(404).json({ error: 'Image not found' });
    }

    const detectionIndex = image.detections?.findIndex(d => d.id === parseInt(objectId));
    if (detectionIndex === -1 || detectionIndex === undefined) {
      return res.status(404).json({ error: 'Object not found' });
    }

    const detection = image.detections[detectionIndex];
    const imageUrl = `http://localhost:${PORT}${image.path}`;

    const response = await fetch(`${PYTHON_API_URL}/descriptors/url`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        url: imageUrl,
        bbox: detection.bbox
      })
    });

    if (!response.ok) {
      throw new Error('Descriptor API failed');
    }

    const result = await response.json();

    // Store descriptors in the detection
    image.detections[detectionIndex].descriptors = result.descriptors;
    await image.save();

    res.json({
      imageId,
      objectId,
      descriptors: result.descriptors,
      success: true
    });
  } catch (error) {
    console.error('Compute descriptor error:', error);
    res.status(500).json({ error: 'Failed to compute descriptors' });
  }
});

// ==================== SEARCH ROUTES ====================

// Search for similar objects
app.post('/api/search/similar', async (req, res) => {
  try {
    const { queryImageId, queryObjectId, topK = 10, weights } = req.body;

    // Get query image and object
    const queryImage = await Image.findById(queryImageId);
    if (!queryImage) {
      return res.status(404).json({ error: 'Query image not found' });
    }

    const queryDetection = queryImage.detections?.find(d => d.id === parseInt(queryObjectId));
    if (!queryDetection) {
      return res.status(404).json({ error: 'Query object not found' });
    }

    // Get or compute query descriptors
    let queryDescriptors = queryDetection.descriptors;
    if (!queryDescriptors) {
      const imageUrl = `http://localhost:${PORT}${queryImage.path}`;
      const descResponse = await fetch(`${PYTHON_API_URL}/descriptors/url`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: imageUrl, bbox: queryDetection.bbox })
      });
      
      if (!descResponse.ok) {
        throw new Error('Failed to get query descriptors');
      }
      
      const descResult = await descResponse.json();
      queryDescriptors = descResult.descriptors;
    }

    // Get all images with detections (exclude query image)
    const images = await Image.find({ 
      'detections.0': { $exists: true },
      _id: { $ne: queryImageId }
    });

    // Build database of objects with descriptors - NO label filtering
    // Similarity is based purely on descriptor values (color, texture, shape)
    const database = [];
    for (const img of images) {
      for (const det of img.detections || []) {
        // Include all objects that have descriptors (no label filtering)
        if (det.descriptors) {
          database.push({
            id: `${img._id}_${det.id}`,
            imageId: img._id.toString(),
            objectId: det.id,
            label: det.label,
            label_readable: det.label_readable,
            confidence: det.confidence,
            bbox: det.bbox,
            descriptors: det.descriptors,
            image: {
              id: img._id,
              filename: img.name,
              url: `http://localhost:${PORT}${img.path}`
            }
          });
        }
      }
    }

    // Call Python API to compute similarities (pass weights if provided)
    const searchPayload = {
      query_descriptors: queryDescriptors,
      database,
      top_k: topK
    };
    
    // Add descriptor weights if specified
    if (weights) {
      searchPayload.weights = weights;
    }
    
    const searchResponse = await fetch(`${PYTHON_API_URL}/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(searchPayload)
    });

    if (!searchResponse.ok) {
      throw new Error('Search API failed');
    }

    const searchResult = await searchResponse.json();

    console.log('Python API search result (first item):', JSON.stringify(searchResult.results?.[0], null, 2));

    // Format results with detailed similarity scores
    const results = searchResult.results.map((r, index) => {
      const dbItem = database.find(d => d.id === r.id);
      console.log(`Result ${index + 1} scores:`, r.scores);
      return {
        rank: index + 1,
        imageId: r.imageId,
        objectId: r.objectId,
        similarity: r.similarity,
        scores: r.scores, // Include detailed scores (color, texture, shape)
        image: dbItem?.image,
        object: {
          id: r.objectId,
          label: r.label,
          label_readable: dbItem?.label_readable,
          confidence: dbItem?.confidence,
          bbox: dbItem?.bbox,
          descriptors: dbItem?.descriptors // Include descriptors for display
        }
      };
    });

    res.json({
      query: { imageId: queryImageId, objectId: queryObjectId },
      results,
      total: results.length
    });
  } catch (error) {
    console.error('Search error:', error);
    res.status(500).json({ error: 'Search failed' });
  }
});
