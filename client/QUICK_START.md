# Quick Start Guide

## Prerequisites Installation

Before starting, ensure you have:
- Node.js (v18 or higher)
- npm (comes with Node.js)

## Installation Steps

### 1. Install Dependencies

```bash
cd client
npm install
```

### 2. Install Additional Required Packages

```bash
npm install react-router-dom react-dropzone autoprefixer
```

### 3. Start Development Server

```bash
npm run dev
```

The app will be available at `http://localhost:5173`

## First Time Setup

### Environment Configuration (Optional)

Create a `.env` file in the `client` directory:

```env
# API Configuration
VITE_API_URL=http://localhost:5000/api

# Set to true to use mock data (no backend required)
VITE_MOCK_API=true
```

## Using the Application

### 1. Collection Page (Home)
**Upload Images**
- Drag and drop images into the upload area, or
- Click "Browse Files" to select images
- Watch the progress bars for each upload

**Browse Gallery**
- View all uploaded images in a grid
- Filter by filename or object category
- Toggle to show only images with detections

**Image Actions**
- Hover over an image card to see action buttons
- Click "Detect Objects" to run object detection
- Click "View Details" to see detection results
- Download or delete images

### 2. Image Detail Page
**Run Detection**
- Click "Run Detection" to detect objects with YOLOv8n
- Wait for processing (mock: ~2 seconds)
- View bounding boxes overlaid on the image

**Interact with Objects**
- Click on bounding boxes to select objects
- Click on objects in the sidebar to highlight them
- Use "View Descriptors" to see visual features
- Use "Use as Query" to search for similar objects

### 3. Descriptors Page
**Select Image and Object**
- Use the dropdowns to select an image
- Choose an object from that image
- Wait for descriptors to load

**Explore Descriptors**
- **Color**: View color histograms and dominant colors
- **Texture**: See Tamura features and Gabor filter responses
- **Shape**: Examine Hu moments and orientation histograms

### 4. Search Page
**Build Your Query**
1. Select an image from the grid (Step 1)
2. Choose an object from that image (Step 2)
3. Click "Search Similar Objects"

**View Results**
- Results are ranked by similarity score
- Click on any result to view that image
- See rank badges (#1, #2, etc.)
- View similarity percentages

### 5. Create Page
**Apply Transformations**
1. Select an image from the dropdown
2. Adjust transformations:
   - **Scale**: Drag slider (25% - 200%)
   - **Rotate**: Click preset angles or free rotate
   - **Flip**: Toggle horizontal/vertical flip
   - **Crop**: Select aspect ratio (coming soon)
3. Preview changes in real-time
4. Click "Apply & Save as New"

## Mock Data vs Real API

### Currently in Mock Mode
The application comes pre-configured with mock data and simulated API calls. This allows you to:
- Explore the UI without a backend
- Test all features with sample images
- See simulated loading states

### Switching to Real API

1. **Set up your Flask backend** with the required endpoints (see API Integration section)

2. **Update `.env` file:**
   ```env
   VITE_API_URL=http://localhost:5000/api
   VITE_MOCK_API=false
   ```

3. **Restart the dev server:**
   ```bash
   npm run dev
   ```

4. **Configure CORS on your backend** to allow requests from `http://localhost:5173`

## Common Tasks

### Adding New Images
1. Go to Collection page
2. Drag images into upload area
3. Wait for upload to complete
4. Images appear in gallery

### Running Object Detection
1. From Collection page, click "Detect Objects" on an image, or
2. Go to Image Detail page and click "Run Detection"
3. Wait for processing
4. View detected objects with bounding boxes

### Finding Similar Objects
1. Ensure you have images with detections
2. Go to Search page
3. Select an image and object as query
4. Click "Search Similar Objects"
5. Browse ranked results

### Creating Transformed Images
1. Go to Create page
2. Select an image
3. Apply desired transformations
4. Click "Apply & Save as New"
5. New image is created in your collection

## Keyboard Shortcuts (Future Enhancement)

```
Coming soon:
- Ctrl/Cmd + U: Upload images
- Ctrl/Cmd + F: Focus search
- Esc: Close modals
- Arrow keys: Navigate gallery
```

## Tips & Best Practices

### Performance
- Upload reasonable image sizes (< 5MB recommended)
- Use filters to narrow down large collections
- Images are lazy-loaded in the gallery

### Organization
- Use descriptive filenames for easy searching
- Run detection on all images for full functionality
- Use filters to find specific object types

### Workflow
1. **Upload** → Upload your image collection
2. **Detect** → Run object detection on images
3. **Explore** → View descriptors for interesting objects
4. **Search** → Find similar objects across collection
5. **Transform** → Create variations of images

## Troubleshooting

### Images not appearing
- Check browser console for errors
- Verify images are in supported formats (JPG, PNG, GIF, WebP)
- Ensure MOCK_MODE is enabled in development

### Detection not working
- In mock mode, detection is simulated (works automatically)
- With real API, ensure Flask backend is running
- Check API_URL in .env file

### Styles not loading
- Clear browser cache
- Restart dev server: `npm run dev`
- Check console for Tailwind CSS errors

### Routing issues
- Use browser navigation (not page refresh)
- Check that React Router is configured
- Verify all routes in App.jsx

## Development Mode Features

### Hot Module Replacement
- Changes to components update instantly
- No page refresh needed
- State is preserved when possible

### Developer Tools
- React DevTools extension recommended
- Browser console shows API calls in mock mode
- Network tab shows all requests

### Error Handling
- Toast notifications for errors
- Console logging for debugging
- Detailed error messages in development

## Building for Production

### Create Production Build
```bash
npm run build
```

### Preview Production Build
```bash
npm run preview
```

### Deploy
- Output is in `dist/` directory
- Upload to hosting service (Vercel, Netlify, etc.)
- Configure environment variables on hosting platform

## Next Steps

1. **Customize Styling**: Edit `tailwind.config.js` for custom colors
2. **Add Backend**: Set up Flask API with YOLOv8n
3. **Extend Features**: Add new transformations or descriptors
4. **Optimize**: Implement pagination, lazy loading
5. **Test**: Add unit tests with Vitest or Jest

## Need Help?

- Check `CLIENT_README.md` for detailed documentation
- Review `COMPONENT_HIERARCHY.md` for architecture
- Inspect component files for implementation details
- Open an issue on GitHub for bugs or questions

## Available Scripts Reference

```bash
# Development
npm run dev          # Start dev server (port 5173)
npm run build        # Build for production
npm run preview      # Preview production build
npm run lint         # Run ESLint

# Package Management
npm install          # Install all dependencies
npm update           # Update dependencies
npm audit fix        # Fix security vulnerabilities
```

Enjoy exploring your images with ImageExplorer! 🎨🔍
