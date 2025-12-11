# Image Indexing System - Setup Guide

A content-based image retrieval system with YOLO object detection and visual descriptor extraction.

## System Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   React Client  │────▶│  Node.js Server  │────▶│  Python FastAPI │
│   (Port 5173)   │     │   (Port 5000)    │     │   (Port 8000)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │   MongoDB    │
                        │ (Port 27017) │
                        └──────────────┘
```

## Prerequisites

- **Node.js** v18+ 
- **Python** 3.10+
- **MongoDB** (running locally on port 27017)

## Quick Start

### 1. Start MongoDB

Make sure MongoDB is running locally. If you don't have it installed:
- Download from: https://www.mongodb.com/try/download/community
- Or use MongoDB Atlas (cloud) and update the connection string in `.env`

### 2. Setup Python API (Terminal 1)

```bash
cd python-api

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the API server
python main.py
```

The Python API will download the YOLOv8n model on first run (~6MB).

### 3. Setup Node.js Server (Terminal 2)

```bash
cd node-server

# Install dependencies
npm install

# Start the server
npm run dev
```

### 4. Setup React Client (Terminal 3)

```bash
cd client

# Install dependencies
npm install

# Start the development server
npm run dev
```

### 5. Access the Application

Open your browser and go to: **http://localhost:5173**

## Features

### 1. Image Collection Management
- Upload images (drag & drop or click to select)
- View all uploaded images in a grid
- Delete images
- Filter by category (uploaded, edited, categories)

### 2. Object Detection
- Click on any image to view details
- Click "Run Detection" to detect objects using YOLOv8
- View detected objects with bounding boxes
- Supports 15 categories: person, car, dog, cat, bird, bicycle, motorcycle, bus, truck, boat, horse, sheep, cow, elephant, bear

### 3. Visual Descriptors
- Go to "Descriptors" page
- Select an image with detected objects
- View computed descriptors:
  - **Color**: HSV histogram, dominant colors
  - **Texture**: Tamura features (coarseness, contrast, directionality), Gabor filter responses
  - **Shape**: Hu moments, orientation histogram

### 4. Similarity Search
- Go to "Search" page
- Select an image and a detected object
- Click "Search" to find similar objects
- Results are ranked by visual similarity

### 5. Image Transformations
- Go to "Create" page
- Select an image
- Apply transformations: scale, rotate, flip
- Save as a new image

## API Endpoints

### Node.js Server (Port 5000)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/images` | List all images |
| GET | `/api/images/:id` | Get single image |
| POST | `/api/images/upload` | Upload image |
| DELETE | `/api/images/:id` | Delete image |
| POST | `/api/images/:id/detect` | Run object detection |
| GET | `/api/images/:imageId/objects/:objectId/descriptors` | Get descriptors |
| POST | `/api/search/similar` | Search similar objects |

### Python API (Port 8000)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/detect` | Detect objects (file upload) |
| POST | `/detect/url` | Detect objects (URL) |
| POST | `/descriptors` | Extract descriptors (file upload) |
| POST | `/descriptors/url` | Extract descriptors (URL) |
| POST | `/detect-and-describe/url` | Detect + extract descriptors |
| POST | `/similarity` | Compute similarity between descriptors |
| POST | `/search` | Search similar objects in database |

## Adding Sample Images

To populate the database with sample images, place images in `node-server/S3/images/` folder, then run:

```bash
cd node-server
npm run sync-db
```

## Troubleshooting

### MongoDB connection failed
- Ensure MongoDB is running: `mongod --dbpath /your/data/path`
- Check connection string in `node-server/.env`

### Python API errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- If OpenCV fails, try: `pip install opencv-python-headless`

### CORS errors
- Ensure all servers are running on correct ports
- Check that CORS origins match in both Node.js and Python servers

### YOLO model download issues
- The model downloads automatically on first run
- If it fails, manually download from: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
- Place it in the `python-api` folder

## Project Structure

```
images-indexing/
├── client/                 # React frontend
│   ├── src/
│   │   ├── components/     # UI components
│   │   ├── pages/          # Page components
│   │   └── services/       # API services
│   └── package.json
├── node-server/            # Express backend
│   ├── models/             # MongoDB models
│   ├── S3/                 # Image storage
│   │   ├── images/         # Category images
│   │   ├── upload/         # Uploaded images
│   │   └── edited/         # Edited images
│   ├── server.js
│   └── package.json
└── python-api/             # FastAPI + YOLO
    ├── main.py             # API endpoints
    ├── detection.py        # YOLO detector
    ├── descriptors.py      # Visual descriptors
    └── requirements.txt
```
