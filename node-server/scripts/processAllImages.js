/**
 * Batch Processing Script
 * 
 * This script processes all images in the database:
 * 1. Runs YOLO detection to find objects in each image
 * 2. Calculates descriptors (color, texture, shape) for each detected object
 * 3. Stores the results in MongoDB
 * 
 * Usage: node scripts/processAllImages.js
 */

import mongoose from 'mongoose';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load .env from parent directory (node-server)
dotenv.config({ path: path.join(__dirname, '..', '.env') });

// MongoDB connection
const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/image-indexing';
const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:8000';
const SERVER_PORT = process.env.PORT || 5000;

// Image Schema - matching the new Python API output format
const detectionSchema = new mongoose.Schema({
  id: Number,
  label: String,           // ImageNet synset ID (e.g., 'n01440764')
  label_readable: String,  // Human-readable name (e.g., 'tench')
  class_id: Number,        // Numeric class ID (0-14)
  confidence: Number,
  bbox: [Number],
  descriptors: {
    color: {
      histogram: [Number],
      hsHistogram: [Number],
      dominantColors: [{
        color: String,
        percentage: Number,
        name: String
      }],
      dominantColorsHS: [[Number]],
      labColorMoments: [Number]
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
      contourHuMoments: [Number],
      orientationHistogram: [Number],
      orientationHistogramContour: [Number],
      shapeMetrics: {
        solidity: Number,
        aspectRatio: Number,
        compactness: Number
      },
      fourierDescriptors: [Number]
    }
  }
}, { _id: false });

const imageSchema = new mongoose.Schema({
  name: { type: String, required: true },
  path: { type: String, required: true },
  category: { type: String, default: 'uncategorized' },
  size: Number,
  width: Number,
  height: Number,
  format: String,
  detections: [detectionSchema],
  objectCount: { type: Number, default: 0 },
  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now }
});

const Image = mongoose.model('Image', imageSchema);

// Progress tracking
let processedCount = 0;
let totalCount = 0;
let successCount = 0;
let errorCount = 0;

/**
 * Process a single image - run detection and extract descriptors
 */
async function processImage(image) {
  const imageUrl = `http://localhost:${SERVER_PORT}${image.path}`;
  
  console.log(`\n[${processedCount + 1}/${totalCount}] Processing: ${image.name}`);
  console.log(`  URL: ${imageUrl}`);
  
  try {
    // Call Python API to detect objects and extract descriptors
    const response = await fetch(`${PYTHON_API_URL}/detect-and-describe/url`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: imageUrl })
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API returned ${response.status}: ${errorText}`);
    }
    
    const result = await response.json();
    
    if (!result.success) {
      throw new Error(result.error || 'Detection failed');
    }
    
    const detections = result.detections || [];
    
    // Format detections for storage - keep the exact format from Python API
    const formattedDetections = detections.map((det, index) => ({
      id: det.id || index,
      label: det.label,
      label_readable: det.label_readable,
      class_id: det.class_id,
      confidence: det.confidence,
      bbox: det.bbox,
      descriptors: det.descriptors || null
    }));
    
    // Update image in database
    await Image.findByIdAndUpdate(image._id, {
      detections: formattedDetections,
      objectCount: formattedDetections.length,
      updatedAt: new Date()
    });
    
    console.log(`  ✓ Found ${formattedDetections.length} objects`);
    
    // Log descriptor summary for each object
    formattedDetections.forEach((det, idx) => {
      const hasColor = det.descriptors?.color ? '✓' : '✗';
      const hasTexture = det.descriptors?.texture ? '✓' : '✗';
      const hasShape = det.descriptors?.shape ? '✓' : '✗';
      const label = det.label_readable || det.label;
      console.log(`    Object ${idx}: ${label} (${(det.confidence * 100).toFixed(1)}%) - Color:${hasColor} Texture:${hasTexture} Shape:${hasShape}`);
      
      // Show some descriptor details
      if (det.descriptors?.color) {
        const colorDesc = det.descriptors.color;
        const histLen = colorDesc.histogram?.length || 0;
        const domColorsLen = colorDesc.dominantColors?.length || 0;
        const labMomentsLen = colorDesc.labColorMoments?.length || 0;
        console.log(`      Color: histogram[${histLen}], dominantColors[${domColorsLen}], labMoments[${labMomentsLen}]`);
      }
      if (det.descriptors?.shape) {
        const shapeDesc = det.descriptors.shape;
        const huLen = shapeDesc.huMoments?.length || shapeDesc.contourHuMoments?.length || 0;
        const fourierLen = shapeDesc.fourierDescriptors?.length || 0;
        const hasMetrics = shapeDesc.shapeMetrics ? '✓' : '✗';
        console.log(`      Shape: huMoments[${huLen}], fourier[${fourierLen}], metrics:${hasMetrics}`);
      }
    });
    
    successCount++;
    
  } catch (error) {
    console.error(`  ✗ Error: ${error.message}`);
    errorCount++;
  }
  
  processedCount++;
}

/**
 * Main function - process all images
 */
async function main() {
  console.log('='.repeat(60));
  console.log('Image Batch Processing Script');
  console.log('='.repeat(60));
  console.log(`\nMongoDB: ${MONGODB_URI}`);
  console.log(`Python API: ${PYTHON_API_URL}`);
  console.log(`Server Port: ${SERVER_PORT}`);
  
  try {
    // Connect to MongoDB
    console.log('\nConnecting to MongoDB...');
    await mongoose.connect(MONGODB_URI);
    console.log('Connected!\n');
    
    // Get all images from database
    const images = await Image.find({});
    totalCount = images.length;
    
    console.log(`Found ${totalCount} images in database\n`);
    
    if (totalCount === 0) {
      console.log('No images to process. Run sync-db first to add images.');
      return;
    }
    
    // Check if Python API is running
    console.log('Checking Python API...');
    try {
      const healthCheck = await fetch(`${PYTHON_API_URL}/`);
      if (!healthCheck.ok) {
        throw new Error('API not responding');
      }
      console.log('Python API is running!\n');
    } catch (error) {
      console.error('ERROR: Python API is not running!');
      console.error('Please start the Python API first: cd python-api && python main.py');
      process.exit(1);
    }
    
    // Process each image
    console.log('Starting batch processing...');
    console.log('-'.repeat(60));
    
    for (const image of images) {
      await processImage(image);
    }
    
    // Summary
    console.log('\n' + '='.repeat(60));
    console.log('PROCESSING COMPLETE');
    console.log('='.repeat(60));
    console.log(`Total images:     ${totalCount}`);
    console.log(`Successful:       ${successCount}`);
    console.log(`Errors:           ${errorCount}`);
    console.log('='.repeat(60));
    
    // Show statistics
    const imagesWithDetections = await Image.countDocuments({ objectCount: { $gt: 0 } });
    const totalObjects = await Image.aggregate([
      { $group: { _id: null, total: { $sum: '$objectCount' } } }
    ]);
    
    console.log(`\nDatabase Statistics:`);
    console.log(`  Images with detections: ${imagesWithDetections}/${totalCount}`);
    console.log(`  Total objects detected: ${totalObjects[0]?.total || 0}`);
    
  } catch (error) {
    console.error('Fatal error:', error);
    process.exit(1);
  } finally {
    await mongoose.disconnect();
    console.log('\nDisconnected from MongoDB');
  }
}

// Run the script
main();
