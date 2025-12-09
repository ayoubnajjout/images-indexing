import mongoose from 'mongoose';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import Image from '../models/Image.js';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Connect to MongoDB
mongoose.connect(process.env.MONGODB_URI)
  .then(() => console.log('✓ MongoDB connected'))
  .catch(err => {
    console.error('MongoDB connection error:', err);
    process.exit(1);
  });

async function syncCategoriesToDB() {
  try {
    console.log('Starting sync of category images to database...\n');
    
    const categoriesPath = path.join(__dirname, '..', 'S3', 'images');
    
    if (!fs.existsSync(categoriesPath)) {
      console.error('Images folder not found!');
      process.exit(1);
    }
    
    // Get all category folders
    const categories = fs.readdirSync(categoriesPath, { withFileTypes: true })
      .filter(dirent => dirent.isDirectory())
      .map(dirent => dirent.name);
    
    console.log(`Found ${categories.length} categories: ${categories.join(', ')}\n`);
    
    let totalAdded = 0;
    let totalSkipped = 0;
    
    for (const category of categories) {
      const categoryPath = path.join(categoriesPath, category);
      const files = fs.readdirSync(categoryPath);
      
      console.log(`Processing category: ${category} (${files.length} files)`);
      
      for (const file of files) {
        const filePath = path.join(categoryPath, file);
        const stats = fs.statSync(filePath);
        
        // Check if image already exists in database
        const existingImage = await Image.findOne({
          category: 'categories',
          path: `/images/${category}/${file}`
        });
        
        if (existingImage) {
          totalSkipped++;
          continue;
        }
        
        // Add image to database
        const newImage = new Image({
          name: file,
          originalName: file,
          filename: file,
          path: `/images/${category}/${file}`,
          category: 'categories',
          size: stats.size,
          mimetype: getMimeType(file)
        });
        
        await newImage.save();
        totalAdded++;
      }
    }
    
    console.log(`\n✓ Sync completed!`);
    console.log(`  - Added: ${totalAdded} images`);
    console.log(`  - Skipped: ${totalSkipped} images (already in database)`);
    
    // Show database statistics
    const counts = await Image.aggregate([
      { $group: { _id: '$category', count: { $sum: 1 } } }
    ]);
    
    console.log('\nDatabase Statistics:');
    counts.forEach(({ _id, count }) => {
      console.log(`  - ${_id}: ${count} images`);
    });
    
    process.exit(0);
  } catch (error) {
    console.error('Error syncing images:', error);
    process.exit(1);
  }
}

function getMimeType(filename) {
  const ext = path.extname(filename).toLowerCase();
  const mimeTypes = {
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.gif': 'image/gif',
    '.webp': 'image/webp',
    '.bmp': 'image/bmp'
  };
  return mimeTypes[ext] || 'image/jpeg';
}

// Run the sync
syncCategoriesToDB();
