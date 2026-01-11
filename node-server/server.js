import express from 'express';
import mongoose from 'mongoose';
import cors from 'cors';
import dotenv from 'dotenv';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import Image from './models/Image.js';
import Model3D from './models/Model3D.js';

dotenv.config();

// Python API URL for YOLO detection and descriptors
const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:8000';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 5000;

// Helper to build URLs for client access and for inter-container API calls
function buildImageUrls(imagePath) {
  const clientUrl = `http://localhost:${PORT}${imagePath}`;
  // When running under docker-compose the Python API is addressed by service name
  const internalHost = process.env.INTERNAL_HOST || (PYTHON_API_URL.includes('python-api') ? 'node-server' : 'localhost');
  const apiUrl = `http://${internalHost}:${PORT}${imagePath}`;
  return { clientUrl, apiUrl };
}

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
    // Accept based on mimetype first (recommended) and fall back to file extension
    try {
      if (file && file.mimetype && file.mimetype.startsWith('image/')) {
        return cb(null, true);
      }

      const ext = (file?.originalname && path.extname(file.originalname).toLowerCase()) || '';
      const allowedExts = ['.jpg', '.jpeg', '.png', '.gif', '.webp'];
      if (allowedExts.includes(ext)) {
        return cb(null, true);
      }

      return cb(new Error('Only image files are allowed!'));
    } catch (err) {
      return cb(new Error('Only image files are allowed!'));
    }
  }
});

// Routes

// Upload single image (wrap multer to handle errors and return JSON)
app.post('/api/images/upload', (req, res) => {
  upload.single('image')(req, res, async (err) => {
    if (err) {
      console.error('Multer upload error:', err);
      return res.status(400).json({ error: err.message });
    }

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

    const { clientUrl, apiUrl } = buildImageUrls(image.path);
    console.log(`Testing descriptors for: ${clientUrl} (python-api will use ${apiUrl})`);

    // Call Python API directly using an internal URL that python-api can reach
    const response = await fetch(`${PYTHON_API_URL}/detect-and-describe/url`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: apiUrl })
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Python API debug error:', response.status, errorText);
      return res.status(502).json({ error: `Python API error: ${response.status} - ${errorText}` });
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

// ==================== 3D MODEL ROUTES ====================

// Path to 3D data
const MODELS_3D_PATH = process.env.MODELS_3D_PATH || path.join(__dirname, '..', '3d-data', '3D Models');
const THUMBNAILS_3D_PATH = process.env.THUMBNAILS_3D_PATH || path.join(__dirname, '..', '3d-data', 'Thumbnails');

// Helper function to find thumbnail with various extensions
function findThumbnail(thumbnailsBase, category, objFilename) {
  const baseName = objFilename.replace('.obj', '');
  const extensions = ['.jpg', '.png', '.JPG', '.PNG', '.jpeg', '.JPEG'];
  
  for (const ext of extensions) {
    // Try exact match
    let testPath = path.join(thumbnailsBase, category, baseName + ext);
    if (fs.existsSync(testPath)) {
      return `/3d-thumbnails/${encodeURIComponent(category)}/${encodeURIComponent(baseName + ext)}`;
    }
    
    // Try capitalized first letter
    const capName = baseName.charAt(0).toUpperCase() + baseName.slice(1);
    testPath = path.join(thumbnailsBase, category, capName + ext);
    if (fs.existsSync(testPath)) {
      return `/3d-thumbnails/${encodeURIComponent(category)}/${encodeURIComponent(capName + ext)}`;
    }
    
    // Try lowercase
    testPath = path.join(thumbnailsBase, category, baseName.toLowerCase() + ext);
    if (fs.existsSync(testPath)) {
      return `/3d-thumbnails/${encodeURIComponent(category)}/${encodeURIComponent(baseName.toLowerCase() + ext)}`;
    }
  }
  
  return null;
}

// Serve 3D model files and thumbnails
app.use('/3d-models', express.static(MODELS_3D_PATH, staticOptions));
app.use('/3d-thumbnails', express.static(THUMBNAILS_3D_PATH, staticOptions));

// Multer configuration for 3D model uploads
const upload3D = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024 }, // 50MB limit for 3D models
  fileFilter: (req, file, cb) => {
    if (file.originalname.endsWith('.obj')) {
      cb(null, true);
    } else {
      cb(new Error('Only .obj files are allowed!'));
    }
  }
});

// Get all 3D model categories
app.get('/api/3d/categories', async (req, res) => {
  try {
    // First try to get from Python API (which has direct file access)
    const response = await fetch(`${PYTHON_API_URL}/3d/categories`);
    if (response.ok) {
      const data = await response.json();
      return res.json(data);
    }
    
    // Fallback to local file system
    const categories = [];
    if (fs.existsSync(MODELS_3D_PATH)) {
      const items = fs.readdirSync(MODELS_3D_PATH);
      for (const item of items) {
        const itemPath = path.join(MODELS_3D_PATH, item);
        if (fs.statSync(itemPath).isDirectory()) {
          const models = fs.readdirSync(itemPath).filter(f => f.endsWith('.obj'));
          categories.push({
            name: item,
            count: models.length
          });
        }
      }
    }
    
    categories.sort((a, b) => a.name.localeCompare(b.name));
    
    res.json({
      success: true,
      categories,
      total: categories.length
    });
  } catch (error) {
    console.error('Get 3D categories error:', error);
    res.status(500).json({ error: 'Failed to fetch categories' });
  }
});

// Get all 3D models with optional filtering
app.get('/api/3d/models', async (req, res) => {
  try {
    const { category, page = 1, limit = 20, indexed } = req.query;
    const skip = (parseInt(page) - 1) * parseInt(limit);
    
    // First check if we have indexed models in the database
    const dbQuery = {};
    if (category) dbQuery.category = category;
    if (indexed === 'true') dbQuery.isIndexed = true;
    
    const [dbModels, dbTotal] = await Promise.all([
      Model3D.find(dbQuery)
        .sort({ category: 1, name: 1 })
        .skip(skip)
        .limit(parseInt(limit))
        .select('-descriptors -combinedDescriptor')
        .lean(),
      Model3D.countDocuments(dbQuery)
    ]);
    
    if (dbModels.length > 0 || indexed === 'true') {
      // Return models from database
      const modelsWithUrls = dbModels.map(model => ({
        ...model,
        modelUrl: `/3d-models/${encodeURIComponent(model.category)}/${encodeURIComponent(model.filename)}`,
        thumbnailUrl: model.thumbnailPath ? 
          `/3d-thumbnails/${encodeURIComponent(model.category)}/${encodeURIComponent(path.basename(model.thumbnailPath))}` : 
          null
      }));
      
      return res.json({
        success: true,
        models: modelsWithUrls,
        total: dbTotal,
        page: parseInt(page),
        totalPages: Math.ceil(dbTotal / parseInt(limit)),
        source: 'database'
      });
    }
    
    // Fallback to file system scan
    const models = [];
    const searchCategories = category ? [category] : 
      (fs.existsSync(MODELS_3D_PATH) ? fs.readdirSync(MODELS_3D_PATH).filter(f => 
        fs.statSync(path.join(MODELS_3D_PATH, f)).isDirectory()) : []);
    
    for (const cat of searchCategories) {
      const catPath = path.join(MODELS_3D_PATH, cat);
      if (!fs.existsSync(catPath)) continue;
      
      const files = fs.readdirSync(catPath).filter(f => f.endsWith('.obj'));
      for (const filename of files) {
        models.push({
          filename,
          name: filename.replace('.obj', ''),
          category: cat,
          modelUrl: `/3d-models/${encodeURIComponent(cat)}/${encodeURIComponent(filename)}`,
          thumbnailUrl: findThumbnail(THUMBNAILS_3D_PATH, cat, filename),
          isIndexed: false
        });
      }
    }
    
    const total = models.length;
    const paginatedModels = models.slice(skip, skip + parseInt(limit));
    
    res.json({
      success: true,
      models: paginatedModels,
      total,
      page: parseInt(page),
      totalPages: Math.ceil(total / parseInt(limit)),
      source: 'filesystem'
    });
  } catch (error) {
    console.error('Get 3D models error:', error);
    res.status(500).json({ error: 'Failed to fetch models' });
  }
});

// Get single 3D model by ID or path
app.get('/api/3d/models/:id', async (req, res) => {
  try {
    const model = await Model3D.findById(req.params.id).lean();
    
    if (!model) {
      return res.status(404).json({ error: '3D model not found' });
    }
    
    res.json({
      success: true,
      model: {
        ...model,
        modelUrl: `/3d-models/${encodeURIComponent(model.category)}/${encodeURIComponent(model.filename)}`,
        thumbnailUrl: model.thumbnailPath ? 
          `/3d-thumbnails/${encodeURIComponent(model.category)}/${encodeURIComponent(path.basename(model.thumbnailPath))}` : 
          null
      }
    });
  } catch (error) {
    console.error('Get 3D model error:', error);
    res.status(500).json({ error: 'Failed to fetch model' });
  }
});

// Extract descriptors from uploaded 3D model
app.post('/api/3d/descriptors', upload3D.single('model'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No model file provided' });
    }
    
    // Send to Python API for descriptor extraction
    const formData = new FormData();
    const blob = new Blob([req.file.buffer], { type: 'application/octet-stream' });
    formData.append('file', blob, req.file.originalname);
    
    const response = await fetch(`${PYTHON_API_URL}/3d/descriptors`, {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Python API error: ${error}`);
    }
    
    const result = await response.json();
    res.json(result);
  } catch (error) {
    console.error('Extract 3D descriptors error:', error);
    res.status(500).json({ error: 'Failed to extract descriptors' });
  }
});

// Search for similar 3D models
app.post('/api/3d/search', upload3D.single('model'), async (req, res) => {
  try {
    const { category, limit = 10, modelPath } = req.body;
    
    // If modelPath is provided, search by path
    if (modelPath) {
      const response = await fetch(`${PYTHON_API_URL}/3d/search/path`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: modelPath, limit: parseInt(limit), category })
      });
      
      if (!response.ok) {
        throw new Error('Search failed');
      }
      
      const result = await response.json();
      
      // Enhance results with URLs
      if (result.results) {
        result.results = result.results.map(r => ({
          ...r,
          modelUrl: `/3d-models/${encodeURIComponent(r.category)}/${encodeURIComponent(r.filename)}`,
          thumbnailUrl: r.thumbnail ? 
            `/3d-thumbnails/${encodeURIComponent(r.category)}/${encodeURIComponent(path.basename(r.thumbnail))}` : 
            null
        }));
      }
      
      return res.json(result);
    }
    
    // Search by uploaded file
    if (!req.file) {
      return res.status(400).json({ error: 'No model file provided' });
    }
    
    // Create a temporary file and send to Python API
    const formData = new FormData();
    const blob = new Blob([req.file.buffer], { type: 'application/octet-stream' });
    formData.append('file', blob, req.file.originalname);
    
    const searchUrl = new URL(`${PYTHON_API_URL}/3d/search`);
    searchUrl.searchParams.set('limit', limit);
    if (category) searchUrl.searchParams.set('category', category);
    
    const response = await fetch(searchUrl.toString(), {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      throw new Error('Search failed');
    }
    
    const result = await response.json();
    
    // Enhance results with URLs
    if (result.results) {
      result.results = result.results.map(r => ({
        ...r,
        modelUrl: `/3d-models/${encodeURIComponent(r.category)}/${encodeURIComponent(r.filename)}`,
        thumbnailUrl: r.thumbnail ? 
          `/3d-thumbnails/${encodeURIComponent(r.category)}/${encodeURIComponent(path.basename(r.thumbnail))}` : 
          null
      }));
    }
    
    res.json(result);
  } catch (error) {
    console.error('Search 3D models error:', error);
    res.status(500).json({ error: 'Search failed' });
  }
});

// Index a 3D model (compute and store descriptors)
app.post('/api/3d/index', async (req, res) => {
  try {
    const { category, filename, filepath } = req.body;
    
    if (!filepath && (!category || !filename)) {
      return res.status(400).json({ error: 'Must provide filepath or category and filename' });
    }
    
    const modelPath = filepath || path.join(MODELS_3D_PATH, category, filename);
    
    // Get descriptors from Python API
    const response = await fetch(`${PYTHON_API_URL}/3d/descriptors/path`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: modelPath })
    });
    
    if (!response.ok) {
      throw new Error('Failed to extract descriptors');
    }
    
    const result = await response.json();
    const desc = result.descriptors;
    
    // Determine thumbnail path
    const cat = category || path.basename(path.dirname(modelPath));
    const fname = filename || path.basename(modelPath);
    const thumbnailUrl = findThumbnail(THUMBNAILS_3D_PATH, cat, fname);
    
    // Save or update in database
    const modelData = {
      name: fname.replace('.obj', ''),
      filename: fname,
      filepath: modelPath,
      category: cat,
      thumbnailPath: thumbnailUrl ? path.join(THUMBNAILS_3D_PATH, cat, path.basename(thumbnailUrl.split('/').pop())) : null,
      numVertices: desc.num_vertices,
      numFaces: desc.num_faces,
      surfaceArea: desc.surface_area,
      descriptors: desc.descriptors,
      combinedDescriptor: desc.combined_descriptor,
      isIndexed: true,
      processedAt: new Date()
    };
    
    const model = await Model3D.findOneAndUpdate(
      { filepath: modelPath },
      modelData,
      { upsert: true, new: true }
    );
    
    res.json({
      success: true,
      model: {
        ...model.toObject(),
        modelUrl: `/3d-models/${encodeURIComponent(cat)}/${encodeURIComponent(fname)}`,
        thumbnailUrl: thumbnailUrl
      }
    });
  } catch (error) {
    console.error('Index 3D model error:', error);
    res.status(500).json({ error: 'Failed to index model' });
  }
});

// Bulk index all models in a category
app.post('/api/3d/index/category', async (req, res) => {
  try {
    const { category } = req.body;
    
    if (!category) {
      return res.status(400).json({ error: 'Category is required' });
    }
    
    const catPath = path.join(MODELS_3D_PATH, category);
    if (!fs.existsSync(catPath)) {
      return res.status(404).json({ error: 'Category not found' });
    }
    
    const files = fs.readdirSync(catPath).filter(f => f.endsWith('.obj'));
    const results = { success: 0, failed: 0, errors: [] };
    
    for (const filename of files) {
      try {
        const modelPath = path.join(catPath, filename);
        
        // Check if already indexed
        const existing = await Model3D.findOne({ filepath: modelPath, isIndexed: true });
        if (existing) {
          results.success++;
          continue;
        }
        
        // Get descriptors
        const response = await fetch(`${PYTHON_API_URL}/3d/descriptors/path`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: modelPath })
        });
        
        if (!response.ok) {
          results.failed++;
          results.errors.push(`${filename}: Failed to extract descriptors`);
          continue;
        }
        
        const result = await response.json();
        const desc = result.descriptors;
        
        const thumbnailUrl = findThumbnail(THUMBNAILS_3D_PATH, category, filename);
        
        await Model3D.findOneAndUpdate(
          { filepath: modelPath },
          {
            name: filename.replace('.obj', ''),
            filename,
            filepath: modelPath,
            category,
            thumbnailPath: thumbnailUrl ? path.join(THUMBNAILS_3D_PATH, category, decodeURIComponent(thumbnailUrl.split('/').pop())) : null,
            numVertices: desc.num_vertices,
            numFaces: desc.num_faces,
            surfaceArea: desc.surface_area,
            descriptors: desc.descriptors,
            combinedDescriptor: desc.combined_descriptor,
            isIndexed: true,
            processedAt: new Date()
          },
          { upsert: true }
        );
        
        results.success++;
      } catch (err) {
        results.failed++;
        results.errors.push(`${filename}: ${err.message}`);
      }
    }
    
    res.json({
      success: true,
      category,
      total: files.length,
      indexed: results.success,
      failed: results.failed,
      errors: results.errors.slice(0, 10) // Limit errors shown
    });
  } catch (error) {
    console.error('Bulk index error:', error);
    res.status(500).json({ error: 'Bulk indexing failed' });
  }
});

// Bulk index ALL models from ALL categories
app.post('/api/3d/index/all', async (req, res) => {
  try {
    if (!fs.existsSync(MODELS_3D_PATH)) {
      return res.status(404).json({ error: '3D models path not found' });
    }
    
    const categories = fs.readdirSync(MODELS_3D_PATH)
      .filter(item => fs.statSync(path.join(MODELS_3D_PATH, item)).isDirectory());
    
    const overallResults = {
      success: 0,
      failed: 0,
      totalFiles: 0,
      categories: {},
      errors: []
    };
    
    for (const category of categories) {
      const catPath = path.join(MODELS_3D_PATH, category);
      const files = fs.readdirSync(catPath).filter(f => f.endsWith('.obj'));
      overallResults.totalFiles += files.length;
      overallResults.categories[category] = { success: 0, failed: 0, total: files.length };
      
      for (const filename of files) {
        try {
          const modelPath = path.join(catPath, filename);
          
          // Check if already indexed
          const existing = await Model3D.findOne({ filepath: modelPath, isIndexed: true });
          if (existing) {
            overallResults.success++;
            overallResults.categories[category].success++;
            continue;
          }
          
          // Get descriptors
          const response = await fetch(`${PYTHON_API_URL}/3d/descriptors/path`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: modelPath })
          });
          
          if (!response.ok) {
            overallResults.failed++;
            overallResults.categories[category].failed++;
            overallResults.errors.push(`${category}/${filename}: Failed to extract descriptors`);
            continue;
          }
          
          const result = await response.json();
          const desc = result.descriptors;
          
          const thumbnailUrl = findThumbnail(THUMBNAILS_3D_PATH, category, filename);
          
          await Model3D.findOneAndUpdate(
            { filepath: modelPath },
            {
              name: filename.replace('.obj', ''),
              filename,
              filepath: modelPath,
              category,
              thumbnailPath: thumbnailUrl ? path.join(THUMBNAILS_3D_PATH, category, decodeURIComponent(thumbnailUrl.split('/').pop())) : null,
              numVertices: desc.num_vertices,
              numFaces: desc.num_faces,
              surfaceArea: desc.surface_area,
              descriptors: desc.descriptors,
              combinedDescriptor: desc.combined_descriptor,
              isIndexed: true,
              processedAt: new Date()
            },
            { upsert: true }
          );
          
          overallResults.success++;
          overallResults.categories[category].success++;
        } catch (err) {
          overallResults.failed++;
          overallResults.categories[category].failed++;
          overallResults.errors.push(`${category}/${filename}: ${err.message}`);
        }
      }
    }
    
    res.json({
      success: true,
      totalCategories: categories.length,
      totalFiles: overallResults.totalFiles,
      indexed: overallResults.success,
      failed: overallResults.failed,
      categories: overallResults.categories,
      errors: overallResults.errors.slice(0, 20) // Limit errors shown
    });
  } catch (error) {
    console.error('Bulk index all error:', error);
    res.status(500).json({ error: 'Bulk indexing all categories failed' });
  }
});

// Get 3D descriptor info
app.get('/api/3d/descriptor-info', async (req, res) => {
  try {
    const response = await fetch(`${PYTHON_API_URL}/3d/descriptor-info`);
    if (!response.ok) {
      throw new Error('Failed to get descriptor info');
    }
    const data = await response.json();
    res.json(data);
  } catch (error) {
    console.error('Get descriptor info error:', error);
    res.status(500).json({ error: 'Failed to get descriptor info' });
  }
});

// Search similar models using database
app.post('/api/3d/search/db', async (req, res) => {
  try {
    const { modelId, category, limit = 10, weights } = req.body;
    
    if (!modelId) {
      return res.status(400).json({ error: 'modelId is required' });
    }
    
    // Get query model
    const queryModel = await Model3D.findById(modelId);
    if (!queryModel || !queryModel.isIndexed) {
      return res.status(404).json({ error: 'Model not found or not indexed' });
    }
    
    // Find similar models
    const results = await Model3D.findSimilar(queryModel.descriptors, {
      limit: parseInt(limit),
      category,
      excludeId: queryModel._id,
      weights
    });
    
    // Add URLs
    const resultsWithUrls = results.map(r => ({
      ...r,
      modelUrl: `/3d-models/${encodeURIComponent(r.category)}/${encodeURIComponent(r.filename)}`,
      thumbnailUrl: r.thumbnailPath ? 
        `/3d-thumbnails/${encodeURIComponent(r.category)}/${encodeURIComponent(path.basename(r.thumbnailPath))}` : 
        null
    }));
    
    res.json({
      success: true,
      query: {
        id: queryModel._id,
        name: queryModel.name,
        category: queryModel.category
      },
      results: resultsWithUrls,
      count: resultsWithUrls.length
    });
  } catch (error) {
    console.error('DB search error:', error);
    res.status(500).json({ error: 'Search failed' });
  }
});

// Get indexing statistics
app.get('/api/3d/stats', async (req, res) => {
  try {
    const [totalModels, indexedModels, categoryCounts] = await Promise.all([
      Model3D.countDocuments(),
      Model3D.countDocuments({ isIndexed: true }),
      Model3D.aggregate([
        { $group: { _id: '$category', count: { $sum: 1 }, indexed: { $sum: { $cond: ['$isIndexed', 1, 0] } } } },
        { $sort: { _id: 1 } }
      ])
    ]);
    
    res.json({
      success: true,
      stats: {
        totalModels,
        indexedModels,
        categories: categoryCounts.map(c => ({
          name: c._id,
          total: c.count,
          indexed: c.indexed
        }))
      }
    });
  } catch (error) {
    console.error('Get stats error:', error);
    res.status(500).json({ error: 'Failed to get stats' });
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

    const { clientUrl, apiUrl } = buildImageUrls(image.path);
    console.log('Image path:', image.path);
    console.log('API URL for Python:', apiUrl);

    // Call Python API for detection (use internal URL so python-api can fetch it)
    const response = await fetch(`${PYTHON_API_URL}/detect-and-describe/url`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: apiUrl })
    });

    if (!response.ok) {
      const text = await response.text();
      console.error('Detection API error:', response.status, text);
      return res.status(502).json({ error: `Detection API failed: ${response.status} - ${text}` });
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
    const { clientUrl, apiUrl } = buildImageUrls(image.path);

    const response = await fetch(`${PYTHON_API_URL}/descriptors/url`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        url: apiUrl,
        bbox: detection.bbox
      })
    });

    if (!response.ok) {
      const text = await response.text();
      console.error('Descriptor API error:', response.status, text);
      return res.status(502).json({ error: `Descriptor API failed: ${response.status} - ${text}` });
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
    const { clientUrl, apiUrl } = buildImageUrls(image.path);

    const response = await fetch(`${PYTHON_API_URL}/descriptors/url`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        url: apiUrl,
        bbox: detection.bbox
      })
    });

    if (!response.ok) {
      const text = await response.text();
      console.error('Descriptor API error:', response.status, text);
      return res.status(502).json({ error: `Descriptor API failed: ${response.status} - ${text}` });
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
      const { clientUrl, apiUrl } = buildImageUrls(queryImage.path);
      const descResponse = await fetch(`${PYTHON_API_URL}/descriptors/url`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: apiUrl, bbox: queryDetection.bbox })
      });
      
      if (!descResponse.ok) {
        const text = await descResponse.text();
        console.error('Query descriptor API error:', descResponse.status, text);
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
