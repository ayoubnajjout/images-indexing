# ImageExplorer - Content-Based Image Exploration Frontend

A minimalistic, modern React frontend for content-based image exploration with YOLOv8n object detection and descriptor-based similarity search.

## Features

### 🖼️ **Image Collection Management**
- Drag-and-drop image upload with progress tracking
- Grid-based image gallery with filtering and search
- Filter by filename, object category, and detection status
- Download and delete images
- Real-time upload progress indicators

### 🎯 **Object Detection**
- Run YOLOv8n object detection on images
- Visual bounding boxes with confidence scores
- Interactive object selection
- Detailed object information panel
- Re-detection capability

### 📊 **Visual Descriptors**
- **Color Descriptors**: Color histograms and dominant colors with percentages
- **Texture Descriptors**: Tamura features (coarseness, contrast, directionality) and Gabor filters
- **Shape Descriptors**: Hu moments and orientation histograms
- Interactive visualizations with charts and progress bars

### 🔍 **Similarity Search**
- Query builder for selecting image and object
- Visual query summary with key descriptors
- Ranked search results by similarity score
- Top-K results with adjustable parameters
- Direct navigation to similar images

### ✨ **Image Transformations**
- Scale: 25% to 200%
- Rotate: Free rotation with preset angles
- Flip: Horizontal and vertical
- Real-time preview with canvas rendering
- Save transformations as new images

## Tech Stack

- **React 18+** - Modern functional components with hooks
- **React Router** - Client-side routing
- **Tailwind CSS** - Utility-first styling
- **Vite** - Fast build tool and dev server

## Project Structure

```
client/
├── src/
│   ├── components/
│   │   ├── ui/              # Reusable UI components
│   │   │   ├── Button.jsx
│   │   │   ├── Card.jsx
│   │   │   ├── Badge.jsx
│   │   │   ├── Input.jsx
│   │   │   ├── Modal.jsx
│   │   │   ├── Toast.jsx
│   │   │   ├── Dropdown.jsx
│   │   │   ├── Tabs.jsx
│   │   │   ├── Spinner.jsx
│   │   │   └── SkeletonLoader.jsx
│   │   ├── layout/          # Layout components
│   │   │   ├── Layout.jsx
│   │   │   └── Navigation.jsx
│   │   ├── collection/      # Collection page components
│   │   │   ├── UploadPanel.jsx
│   │   │   ├── FilterBar.jsx
│   │   │   ├── ImageCard.jsx
│   │   │   └── ImageGrid.jsx
│   │   ├── detection/       # Detection page components
│   │   │   ├── ImageViewer.jsx
│   │   │   └── ObjectList.jsx
│   │   ├── descriptors/     # Descriptors page components
│   │   │   ├── ColorDescriptors.jsx
│   │   │   ├── TextureDescriptors.jsx
│   │   │   ├── ShapeDescriptors.jsx
│   │   │   └── ObjectInfoPanel.jsx
│   │   ├── search/          # Search page components
│   │   │   ├── QueryBuilder.jsx
│   │   │   ├── QuerySummary.jsx
│   │   │   └── SearchResults.jsx
│   │   └── transform/       # Transform page components
│   │       ├── ImageTransformer.jsx
│   │       └── TransformControls.jsx
│   ├── pages/               # Page components
│   │   ├── CollectionPage.jsx
│   │   ├── ImageDetailPage.jsx
│   │   ├── DescriptorsPage.jsx
│   │   ├── SearchPage.jsx
│   │   └── CreatePage.jsx
│   ├── services/            # API service layer
│   │   ├── api.js
│   │   ├── imageService.js
│   │   ├── descriptorService.js
│   │   ├── searchService.js
│   │   └── transformService.js
│   ├── App.jsx              # Main app component
│   ├── main.jsx             # Entry point
│   └── index.css            # Global styles
├── tailwind.config.js       # Tailwind configuration
├── postcss.config.js        # PostCSS configuration
├── vite.config.js           # Vite configuration
└── package.json             # Dependencies
```

## Installation

### Prerequisites
- Node.js 18+ and npm

### Setup

1. **Navigate to the client directory:**
   ```bash
   cd client
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Install additional required packages:**
   ```bash
   npm install react-router-dom react-dropzone autoprefixer
   ```

4. **Configure environment variables (optional):**
   Create a `.env` file:
   ```env
   VITE_API_URL=http://localhost:5000/api
   VITE_MOCK_API=true
   ```

5. **Start the development server:**
   ```bash
   npm run dev
   ```

6. **Open your browser:**
   Navigate to `http://localhost:5173`

## User Flow

### 1. Upload Images → 2. Run Detection → 3. View Descriptors → 4. Search Similar → 5. Explore Results

```
┌─────────────────┐
│   Collection    │  Upload images, manage gallery
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Image Detail    │  Run object detection, view bounding boxes
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Descriptors    │  Explore color, texture, shape features
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Search      │  Build query, find similar objects
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Results      │  View ranked similar images/objects
└─────────────────┘
```

## API Integration

The app is designed to integrate with a Flask REST API. Currently running in **mock mode** with sample data.

### Required API Endpoints

#### Images
- `GET /api/images` - List images with filters
- `GET /api/images/:id` - Get single image
- `POST /api/images/upload` - Upload images
- `DELETE /api/images/:id` - Delete image
- `POST /api/images/:id/detect` - Run object detection

#### Descriptors
- `GET /api/images/:imageId/objects/:objectId/descriptors` - Get descriptors
- `POST /api/images/:imageId/objects/:objectId/descriptors` - Compute descriptors

#### Search
- `POST /api/search/similar` - Similarity search
- `GET /api/search/history` - Search history

#### Transform
- `POST /api/images/:id/transform` - Apply transformations

### Switching from Mock to Real API

1. Set `VITE_MOCK_API=false` in `.env`
2. Set `VITE_API_URL` to your Flask API URL
3. Ensure CORS is configured on the backend

## Design System

### Color Palette
- **Primary (Accent)**: Indigo (`#4f46e5`)
- **Success**: Green
- **Warning**: Yellow
- **Danger**: Red
- **Neutral**: Slate grays
- **Background**: `slate-50`

### Typography
- **Font**: System sans-serif
- **Headings**: Semibold, clear hierarchy
- **Body**: Regular weight, comfortable line-height

### Components
All components follow a consistent design language:
- **Cards**: White background, subtle borders, shadow on hover
- **Buttons**: Primary (indigo), Secondary (white), Ghost, Outline variants
- **Inputs**: Border focus with ring effect
- **Badges**: Rounded, color-coded by variant
- **Spacing**: Generous whitespace, 4px/8px grid

### Responsive Breakpoints
- **Mobile**: < 640px
- **Tablet**: 640px - 1024px
- **Desktop**: > 1024px

## Key Features Explained

### 1. Collection Page
- **Upload**: Drag-and-drop with multi-file support and progress tracking
- **Gallery**: Responsive grid with hover actions
- **Filters**: Text search, category filter, detection status toggle
- **Actions**: View, Download, Delete, Detect

### 2. Image Detail Page
- **Viewer**: Interactive image with overlay bounding boxes
- **Detection**: Run/re-run YOLOv8n detection
- **Object List**: Sidebar with detected objects, click to highlight
- **Quick Actions**: View descriptors, use as search query

### 3. Descriptors Page
- **Selection**: Dropdowns for image and object
- **Visualizations**:
  - Color: Histogram bars, dominant color swatches
  - Texture: Progress bars for Tamura, grid for Gabor
  - Shape: List of Hu moments, orientation histogram

### 4. Search Page
- **Query Builder**: Step-by-step image and object selection
- **Query Summary**: Preview with key descriptors
- **Results**: Grid of ranked similar objects with similarity scores

### 5. Create Page
- **Transformations**: Scale, rotate, flip controls
- **Preview**: Real-time canvas rendering
- **Save**: Creates new image with transformations applied

## Development

### Available Scripts
- `npm run dev` - Start dev server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

### Code Style
- **Components**: Functional components with hooks
- **Props**: Destructured with defaults
- **State**: useState for local, consider Context for global
- **Styling**: Tailwind utility classes
- **Files**: PascalCase for components, camelCase for utilities

### Adding New Features

1. **Create component** in appropriate directory
2. **Export** from index.js
3. **Import** in page component
4. **Add API call** in service layer if needed
5. **Update routing** if new page

## Performance Optimizations

- **Lazy Loading**: Consider React.lazy for pages
- **Image Optimization**: Use appropriate image sizes
- **Memoization**: Use React.memo for expensive components
- **Debouncing**: Debounce search inputs
- **Pagination**: Implement for large image collections

## Deployment

### Build for Production
```bash
npm run build
```

Output will be in `dist/` directory.

### Deploy Options
- **Vercel**: `vercel deploy`
- **Netlify**: Drag-and-drop `dist/` folder
- **GitHub Pages**: Use `gh-pages` package
- **Docker**: Create Dockerfile with nginx

## Troubleshooting

### Common Issues

1. **Tailwind styles not working**
   - Check `tailwind.config.js` content paths
   - Verify `@import "tailwindcss"` in index.css

2. **Images not loading**
   - Check CORS settings on image URLs
   - Verify API URL in `.env`

3. **Routes not working**
   - Ensure `BrowserRouter` is wrapping app
   - Check server configuration for SPA routing

## Future Enhancements

- [ ] Batch operations on images
- [ ] Advanced crop tool with draggable regions
- [ ] Export search results
- [ ] Image comparison side-by-side
- [ ] Keyboard shortcuts
- [ ] Dark mode support
- [ ] Accessibility improvements (ARIA labels)
- [ ] PWA support
- [ ] Real-time collaboration

## Contributing

1. Follow the existing code style
2. Write clear commit messages
3. Test on multiple screen sizes
4. Update documentation

## License

MIT

## Contact

For questions or support, please open an issue on GitHub.
