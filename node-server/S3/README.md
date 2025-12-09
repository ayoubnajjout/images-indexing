# S3 Storage Service

This folder simulates an independent S3-like storage service for the Image Indexing Application.

## Overview

The `S3/` folder serves as a centralized, independent storage location for all image files, mimicking the architecture of cloud storage services like Amazon S3. This design provides:

- **Separation of Concerns**: Storage is independent from application logic
- **Scalability**: Easy to migrate to actual AWS S3 or similar services
- **Organization**: Clear separation between different image categories
- **Portability**: Can be easily backed up, replicated, or moved

## Folder Structure

```
S3/
├── images/          # Category images (pre-loaded dataset)
│   ├── acoustic_guitar/
│   ├── airliner/
│   ├── ant/
│   ├── banana/
│   └── ... (more categories)
├── upload/          # User-uploaded images
└── edited/          # Transformed/edited images
```

## Storage Categories

### 1. `images/` - Category Images
- **Purpose**: Pre-existing image dataset organized by categories
- **Source**: Synced to MongoDB via `npm run sync-db`
- **Access**: Read-only via `/images/*` endpoint
- **Use Case**: Browse pre-loaded image collections

### 2. `upload/` - Uploaded Images
- **Purpose**: User-uploaded images via API
- **Source**: Created via `POST /api/images/upload`
- **Access**: Read/write via `/upload/*` endpoint
- **Naming**: Timestamp-based unique filenames
- **Limits**: 10MB max file size, images only

### 3. `edited/` - Edited Images
- **Purpose**: Images created through transformations
- **Source**: Created via `POST /api/images/save-edited`
- **Access**: Read/write via `/edited/*` endpoint
- **Format**: Saved as PNG with unique names
- **Operations**: Scale, rotate, flip transformations

## HTTP Endpoints

All files in this folder are served via Express static file serving:

```
http://localhost:5000/images/{category}/{filename}
http://localhost:5000/upload/{filename}
http://localhost:5000/edited/{filename}
```

## MongoDB Integration

While files are stored in `S3/`, metadata is stored in MongoDB:

```javascript
{
  name: "image_name",
  filename: "unique_filename.jpg",
  path: "/upload/unique_filename.jpg",  // Relative path
  category: "uploaded",                  // uploaded | edited | categories
  size: 1024000,
  mimetype: "image/jpeg",
  uploadedAt: "2025-12-09T..."
}
```

## Migration to Real S3

To migrate to actual AWS S3 or similar services:

1. **Update Storage Logic**:
   - Replace file system operations with S3 SDK calls
   - Update paths from local to S3 URLs

2. **Update Multer Configuration**:
   - Use `multer-s3` instead of disk storage
   - Configure S3 bucket credentials

3. **Update Image URLs**:
   - Change from `http://localhost:5000/...` to S3 URLs
   - Update CORS configuration for S3 bucket

4. **Sync Existing Files**:
   - Use AWS CLI or SDK to upload existing files
   - Update database paths to S3 URLs

## Backup & Maintenance

### Backup
```bash
# Backup entire S3 folder
tar -czf s3-backup-$(date +%Y%m%d).tar.gz S3/

# Or copy to cloud storage
rclone sync S3/ remote:backup/s3/
```

### Clean Up Orphaned Files
```bash
# Find files not in database
node scripts/cleanOrphanedFiles.js
```

### Storage Statistics
```bash
# Check storage usage
du -sh S3/*

# Count files per category
find S3/images -type f | wc -l
find S3/upload -type f | wc -l
find S3/edited -type f | wc -l
```

## Security Considerations

1. **File Validation**: Only image types allowed
2. **Size Limits**: 10MB max per upload
3. **Unique Naming**: Timestamp-based to prevent collisions
4. **Path Sanitization**: Prevent directory traversal attacks
5. **CORS**: Configured for local development

## Performance

- **Static File Serving**: Express serves files directly
- **Caching**: Browser caching enabled for static assets
- **Lazy Loading**: Frontend loads images on demand
- **Pagination**: Reduces load by limiting queries

## Future Enhancements

- [ ] CDN integration for faster delivery
- [ ] Image optimization (compression, resizing)
- [ ] Automatic thumbnail generation
- [ ] Storage quota management per user
- [ ] File versioning system
- [ ] Automatic backup to cloud storage
