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
app.use(express.json());

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

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
