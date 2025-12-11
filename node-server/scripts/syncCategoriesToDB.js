import mongoose from 'mongoose';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import Image from '../models/Image.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load .env from parent directory (node-server)
dotenv.config({ path: path.join(__dirname, '..', '.env') });

// Connect to MongoDB
mongoose.connect(process.env.MONGODB_URI)
  .then(() => console.log('✓ MongoDB connected'))
  .catch(err => {
    console.error('MongoDB connection error:', err);
    process.exit(1);
  });

async function syncCategoriesToDB() {
  try {
    console.log('Starting sync of images to database...\n');
    
    const imagesPath = path.join(__dirname, '..', 'S3', 'images');
    
    if (!fs.existsSync(imagesPath)) {
      console.error('Images folder not found!');
      process.exit(1);
    }
    
    let totalAdded = 0;
    let totalSkipped = 0;
    
    // Get all items in the images folder
    const items = fs.readdirSync(imagesPath, { withFileTypes: true });
    
    // Get direct image files (not in subfolders)
    const directFiles = items.filter(item => item.isFile() && isImageFile(item.name));
    
    // Get category folders
    const categories = items.filter(item => item.isDirectory()).map(item => item.name);
    
    // Process direct images first
    if (directFiles.length > 0) {
      console.log(`Processing ${directFiles.length} images in root folder...`);
      
      for (const file of directFiles) {
        const filePath = path.join(imagesPath, file.name);
        const stats = fs.statSync(filePath);
        
        // Check if image already exists in database
        const existingImage = await Image.findOne({
          category: 'categories',
          path: `/images/${file.name}`
        });
        
        if (existingImage) {
          totalSkipped++;
          continue;
        }
        
        // Add image to database
        const newImage = new Image({
          name: file.name,
          originalName: file.name,
          filename: file.name,
          path: `/images/${file.name}`,
          category: 'categories',
          size: stats.size,
          mimetype: getMimeType(file.name)
        });
        
        await newImage.save();
        totalAdded++;
      }
    }
    
    // Process category subfolders
    if (categories.length > 0) {
      console.log(`Found ${categories.length} category folders: ${categories.join(', ')}\n`);
      
      for (const category of categories) {
        const categoryPath = path.join(imagesPath, category);
        const files = fs.readdirSync(categoryPath).filter(f => isImageFile(f));
        
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

function isImageFile(filename) {
  const ext = path.extname(filename).toLowerCase();
  return ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'].includes(ext);
}

// Run the sync
syncCategoriesToDB();
