"""
3D Shape Descriptors Module using OpenGL for Multi-View Rendering

This module implements Global Feature-based similarity descriptors for 3D models:
1. D1 - Distance between a fixed point and a random point on the surface
2. D2 - Distance between two random points on the surface  
3. D3 - Square root of area of triangle formed by 3 random points
4. D4 - Cube root of volume of tetrahedron formed by 4 random points
5. A3 - Angle between 3 random points on the surface
6. Multi-view 2D projections using OpenGL rendering

Based on: "Shape Distributions" by Osada et al. and
"A search engine for 3D models" by Funkhouser et al.
"""

import numpy as np
import os
from typing import Tuple, List, Dict, Optional
import math
from collections import defaultdict

# OpenGL imports
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
    import glfw
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    print("Warning: OpenGL not available. Some features will be disabled.")

# For image processing of rendered views
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class OBJLoader:
    """Load and parse .obj files"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.vertices = []
        self.faces = []
        self.normals = []
        self.face_normals = []
        self.centroid = np.array([0.0, 0.0, 0.0])
        self.bounding_box = None
        self.scale_factor = 1.0
        
    def load(self) -> bool:
        """Load the OBJ file and parse vertices and faces"""
        try:
            with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if not parts:
                        continue
                        
                    if parts[0] == 'v' and len(parts) >= 4:
                        # Vertex
                        try:
                            vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                            self.vertices.append(vertex)
                        except (ValueError, IndexError):
                            continue
                            
                    elif parts[0] == 'vn' and len(parts) >= 4:
                        # Vertex normal
                        try:
                            normal = [float(parts[1]), float(parts[2]), float(parts[3])]
                            self.normals.append(normal)
                        except (ValueError, IndexError):
                            continue
                            
                    elif parts[0] == 'f':
                        # Face - can have format: v, v/vt, v/vt/vn, v//vn
                        face_vertices = []
                        for part in parts[1:]:
                            indices = part.split('/')
                            try:
                                v_idx = int(indices[0]) - 1  # OBJ indices are 1-based
                                if v_idx >= 0:
                                    face_vertices.append(v_idx)
                            except (ValueError, IndexError):
                                continue
                        
                        if len(face_vertices) >= 3:
                            # Triangulate if needed (simple fan triangulation)
                            for i in range(1, len(face_vertices) - 1):
                                self.faces.append([
                                    face_vertices[0],
                                    face_vertices[i],
                                    face_vertices[i + 1]
                                ])
            
            if len(self.vertices) == 0:
                return False
                
            self.vertices = np.array(self.vertices, dtype=np.float32)
            self.faces = np.array(self.faces, dtype=np.int32)
            
            # Calculate centroid and normalize
            self._normalize_model()
            self._compute_face_normals()
            
            return True
            
        except Exception as e:
            print(f"Error loading OBJ file: {e}")
            return False
    
    def _normalize_model(self):
        """Center and scale the model to fit in a unit sphere"""
        # Calculate centroid
        self.centroid = np.mean(self.vertices, axis=0)
        
        # Center the model
        self.vertices = self.vertices - self.centroid
        
        # Calculate bounding box
        min_coords = np.min(self.vertices, axis=0)
        max_coords = np.max(self.vertices, axis=0)
        self.bounding_box = (min_coords, max_coords)
        
        # Scale to fit in unit sphere
        max_dist = np.max(np.linalg.norm(self.vertices, axis=1))
        if max_dist > 0:
            self.scale_factor = 1.0 / max_dist
            self.vertices = self.vertices * self.scale_factor
    
    def _compute_face_normals(self):
        """Compute face normals for rendering"""
        self.face_normals = []
        for face in self.faces:
            if len(face) >= 3:
                v0 = self.vertices[face[0]]
                v1 = self.vertices[face[1]]
                v2 = self.vertices[face[2]]
                
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                norm = np.linalg.norm(normal)
                if norm > 0:
                    normal = normal / norm
                self.face_normals.append(normal)
            else:
                self.face_normals.append(np.array([0, 0, 1]))
        
        self.face_normals = np.array(self.face_normals, dtype=np.float32)
    
    def get_surface_area(self) -> float:
        """Calculate total surface area of the mesh"""
        total_area = 0.0
        for face in self.faces:
            if len(face) >= 3:
                v0 = self.vertices[face[0]]
                v1 = self.vertices[face[1]]
                v2 = self.vertices[face[2]]
                
                edge1 = v1 - v0
                edge2 = v2 - v0
                cross = np.cross(edge1, edge2)
                area = 0.5 * np.linalg.norm(cross)
                total_area += area
        
        return total_area
    
    def sample_points_on_surface(self, num_points: int) -> np.ndarray:
        """Sample random points on the mesh surface using area-weighted sampling"""
        if len(self.faces) == 0:
            return np.array([])
        
        # Calculate face areas for weighted sampling
        areas = []
        for face in self.faces:
            v0 = self.vertices[face[0]]
            v1 = self.vertices[face[1]]
            v2 = self.vertices[face[2]]
            
            edge1 = v1 - v0
            edge2 = v2 - v0
            area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
            areas.append(area)
        
        areas = np.array(areas)
        total_area = np.sum(areas)
        
        if total_area == 0:
            return np.array([])
        
        # Normalize to get probabilities
        probabilities = areas / total_area
        
        # Sample faces based on area
        face_indices = np.random.choice(len(self.faces), size=num_points, p=probabilities)
        
        # Sample points within each selected face using barycentric coordinates
        points = []
        for face_idx in face_indices:
            face = self.faces[face_idx]
            v0 = self.vertices[face[0]]
            v1 = self.vertices[face[1]]
            v2 = self.vertices[face[2]]
            
            # Random barycentric coordinates
            r1, r2 = np.random.random(2)
            sqrt_r1 = np.sqrt(r1)
            
            # Point on triangle
            point = (1 - sqrt_r1) * v0 + sqrt_r1 * (1 - r2) * v1 + sqrt_r1 * r2 * v2
            points.append(point)
        
        return np.array(points)


class ShapeDistributionDescriptors:
    """
    Compute Shape Distribution descriptors (Osada et al.)
    
    These are global shape descriptors based on the distribution of 
    geometric measurements sampled from the shape's surface.
    """
    
    def __init__(self, num_samples: int = 10000, num_bins: int = 64):
        self.num_samples = num_samples
        self.num_bins = num_bins
    
    def compute_d1(self, obj: OBJLoader) -> np.ndarray:
        """
        D1: Distribution of distances from centroid to random surface points
        Measures the overall size/extent of the shape
        """
        points = obj.sample_points_on_surface(self.num_samples)
        if len(points) == 0:
            return np.zeros(self.num_bins)
        
        # Distance from origin (model is already centered)
        distances = np.linalg.norm(points, axis=1)
        
        # Create histogram
        hist, _ = np.histogram(distances, bins=self.num_bins, range=(0, 1.5), density=True)
        hist = hist / (np.sum(hist) + 1e-10)  # Normalize
        
        return hist.astype(np.float32)
    
    def compute_d2(self, obj: OBJLoader) -> np.ndarray:
        """
        D2: Distribution of distances between pairs of random surface points
        Most discriminative descriptor - captures overall shape structure
        """
        points = obj.sample_points_on_surface(self.num_samples * 2)
        if len(points) < 2:
            return np.zeros(self.num_bins)
        
        # Pair up points and compute distances
        half = len(points) // 2
        points1 = points[:half]
        points2 = points[half:half*2]
        
        distances = np.linalg.norm(points1 - points2, axis=1)
        
        # Create histogram
        hist, _ = np.histogram(distances, bins=self.num_bins, range=(0, 2.0), density=True)
        hist = hist / (np.sum(hist) + 1e-10)
        
        return hist.astype(np.float32)
    
    def compute_d3(self, obj: OBJLoader) -> np.ndarray:
        """
        D3: Distribution of sqrt(area) of triangles formed by 3 random points
        Captures local surface structure
        """
        points = obj.sample_points_on_surface(self.num_samples * 3)
        if len(points) < 3:
            return np.zeros(self.num_bins)
        
        # Group into triplets
        third = len(points) // 3
        p1 = points[:third]
        p2 = points[third:third*2]
        p3 = points[third*2:third*3]
        
        # Calculate triangle areas using cross product
        edge1 = p2 - p1
        edge2 = p3 - p1
        crosses = np.cross(edge1, edge2)
        areas = 0.5 * np.linalg.norm(crosses, axis=1)
        sqrt_areas = np.sqrt(areas)
        
        # Create histogram
        hist, _ = np.histogram(sqrt_areas, bins=self.num_bins, range=(0, 1.5), density=True)
        hist = hist / (np.sum(hist) + 1e-10)
        
        return hist.astype(np.float32)
    
    def compute_d4(self, obj: OBJLoader) -> np.ndarray:
        """
        D4: Distribution of cbrt(volume) of tetrahedra formed by 4 random points
        Captures volumetric structure
        """
        points = obj.sample_points_on_surface(self.num_samples * 4)
        if len(points) < 4:
            return np.zeros(self.num_bins)
        
        # Group into quadruplets
        quarter = len(points) // 4
        p1 = points[:quarter]
        p2 = points[quarter:quarter*2]
        p3 = points[quarter*2:quarter*3]
        p4 = points[quarter*3:quarter*4]
        
        # Calculate tetrahedron volumes using scalar triple product
        edge1 = p2 - p1
        edge2 = p3 - p1
        edge3 = p4 - p1
        
        # Volume = |det([edge1, edge2, edge3])| / 6
        volumes = np.abs(np.einsum('ij,ij->i', edge1, np.cross(edge2, edge3))) / 6.0
        cbrt_volumes = np.cbrt(volumes)
        
        # Create histogram
        hist, _ = np.histogram(cbrt_volumes, bins=self.num_bins, range=(0, 0.8), density=True)
        hist = hist / (np.sum(hist) + 1e-10)
        
        return hist.astype(np.float32)
    
    def compute_a3(self, obj: OBJLoader) -> np.ndarray:
        """
        A3: Distribution of angles between 3 random surface points
        Captures angular structure of the shape
        """
        points = obj.sample_points_on_surface(self.num_samples * 3)
        if len(points) < 3:
            return np.zeros(self.num_bins)
        
        # Group into triplets
        third = len(points) // 3
        p1 = points[:third]
        p2 = points[third:third*2]  # This is the vertex of the angle
        p3 = points[third*2:third*3]
        
        # Calculate angles at p2
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Normalize
        v1_norm = np.linalg.norm(v1, axis=1, keepdims=True)
        v2_norm = np.linalg.norm(v2, axis=1, keepdims=True)
        
        # Avoid division by zero
        v1_norm = np.maximum(v1_norm, 1e-10)
        v2_norm = np.maximum(v2_norm, 1e-10)
        
        v1 = v1 / v1_norm
        v2 = v2 / v2_norm
        
        # Dot product and angle
        dots = np.clip(np.einsum('ij,ij->i', v1, v2), -1.0, 1.0)
        angles = np.arccos(dots)
        
        # Create histogram (angles from 0 to pi)
        hist, _ = np.histogram(angles, bins=self.num_bins, range=(0, np.pi), density=True)
        hist = hist / (np.sum(hist) + 1e-10)
        
        return hist.astype(np.float32)
    
    def compute_all(self, obj: OBJLoader) -> Dict[str, np.ndarray]:
        """Compute all shape distribution descriptors"""
        return {
            'd1': self.compute_d1(obj),
            'd2': self.compute_d2(obj),
            'd3': self.compute_d3(obj),
            'd4': self.compute_d4(obj),
            'a3': self.compute_a3(obj)
        }


class OpenGLRenderer:
    """
    OpenGL-based renderer for multi-view 3D model rendering
    Creates 2D projections from multiple viewpoints for view-based descriptors
    """
    
    def __init__(self, width: int = 256, height: int = 256):
        self.width = width
        self.height = height
        self.window = None
        self.initialized = False
        
    def initialize(self) -> bool:
        """Initialize OpenGL context using GLFW"""
        if not OPENGL_AVAILABLE:
            print("OpenGL not available")
            return False
            
        try:
            if not glfw.init():
                print("Failed to initialize GLFW")
                return False
            
            # Create invisible window for offscreen rendering
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
            
            self.window = glfw.create_window(self.width, self.height, "3D Renderer", None, None)
            
            if not self.window:
                print("Failed to create GLFW window")
                glfw.terminate()
                return False
            
            glfw.make_context_current(self.window)
            
            # Setup OpenGL
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_LIGHTING)
            glEnable(GL_LIGHT0)
            glEnable(GL_COLOR_MATERIAL)
            glEnable(GL_NORMALIZE)
            
            # Light setup
            glLightfv(GL_LIGHT0, GL_POSITION, [1.0, 1.0, 1.0, 0.0])
            glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
            glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
            
            glClearColor(1.0, 1.0, 1.0, 1.0)
            
            self.initialized = True
            return True
            
        except Exception as e:
            print(f"Error initializing OpenGL: {e}")
            return False
    
    def cleanup(self):
        """Clean up OpenGL resources"""
        if self.window:
            glfw.destroy_window(self.window)
        glfw.terminate()
        self.initialized = False
    
    def render_view(self, obj: OBJLoader, azimuth: float, elevation: float) -> Optional[np.ndarray]:
        """
        Render the model from a specific viewpoint
        
        Args:
            obj: Loaded OBJ model
            azimuth: Horizontal angle in degrees (0-360)
            elevation: Vertical angle in degrees (-90 to 90)
            
        Returns:
            Rendered image as numpy array or None on error
        """
        if not self.initialized:
            return None
            
        try:
            glfw.make_context_current(self.window)
            
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            # Setup projection
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(45.0, self.width / self.height, 0.1, 100.0)
            
            # Setup modelview
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            
            # Camera position based on angles
            distance = 3.0
            az_rad = math.radians(azimuth)
            el_rad = math.radians(elevation)
            
            eye_x = distance * math.cos(el_rad) * math.sin(az_rad)
            eye_y = distance * math.sin(el_rad)
            eye_z = distance * math.cos(el_rad) * math.cos(az_rad)
            
            gluLookAt(eye_x, eye_y, eye_z, 0, 0, 0, 0, 1, 0)
            
            # Render the model
            glColor3f(0.7, 0.7, 0.8)
            glBegin(GL_TRIANGLES)
            
            for i, face in enumerate(obj.faces):
                if i < len(obj.face_normals):
                    glNormal3fv(obj.face_normals[i])
                
                for vertex_idx in face:
                    if vertex_idx < len(obj.vertices):
                        glVertex3fv(obj.vertices[vertex_idx])
            
            glEnd()
            
            glFlush()
            
            # Read pixels
            pixels = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
            image = np.frombuffer(pixels, dtype=np.uint8).reshape(self.height, self.width, 3)
            
            # Flip vertically (OpenGL origin is bottom-left)
            image = np.flipud(image)
            
            return image
            
        except Exception as e:
            print(f"Error rendering view: {e}")
            return None
    
    def render_multi_view(self, obj: OBJLoader, num_views: int = 20) -> List[np.ndarray]:
        """
        Render model from multiple viewpoints distributed on a sphere
        
        Uses icosahedron-based distribution for even coverage
        """
        views = []
        
        # Generate viewpoints distributed on sphere
        # Using golden spiral for uniform distribution
        golden_ratio = (1 + math.sqrt(5)) / 2
        
        for i in range(num_views):
            theta = 2 * math.pi * i / golden_ratio
            phi = math.acos(1 - 2 * (i + 0.5) / num_views)
            
            azimuth = math.degrees(theta) % 360
            elevation = 90 - math.degrees(phi)
            
            view = self.render_view(obj, azimuth, elevation)
            if view is not None:
                views.append(view)
        
        return views


class MultiViewDescriptor:
    """
    Compute multi-view based shape descriptors using OpenGL rendering
    
    Extracts 2D features from rendered views and combines them into
    a global descriptor for 3D shape matching.
    """
    
    def __init__(self, num_views: int = 12, image_size: int = 64):
        self.num_views = num_views
        self.image_size = image_size
        self.renderer = None
        
    def initialize(self) -> bool:
        """Initialize the OpenGL renderer"""
        if not OPENGL_AVAILABLE:
            return False
            
        self.renderer = OpenGLRenderer(self.image_size, self.image_size)
        return self.renderer.initialize()
    
    def cleanup(self):
        """Clean up resources"""
        if self.renderer:
            self.renderer.cleanup()
    
    def compute_view_histogram(self, image: np.ndarray, num_bins: int = 16) -> np.ndarray:
        """
        Compute histogram-based descriptor from a single view
        Uses grayscale intensity histogram
        """
        # Convert to grayscale
        gray = np.mean(image, axis=2).astype(np.uint8)
        
        # Compute histogram
        hist, _ = np.histogram(gray, bins=num_bins, range=(0, 255), density=True)
        hist = hist / (np.sum(hist) + 1e-10)
        
        return hist.astype(np.float32)
    
    def compute_silhouette_descriptor(self, image: np.ndarray) -> np.ndarray:
        """
        Compute silhouette-based descriptor from a rendered view
        Captures the shape outline from each viewpoint
        """
        # Convert to grayscale and threshold to get silhouette
        gray = np.mean(image, axis=2)
        
        # Background is white (255), object is darker
        silhouette = (gray < 250).astype(np.float32)
        
        # Compute basic statistics of silhouette
        features = []
        
        # 1. Silhouette area ratio
        area_ratio = np.sum(silhouette) / silhouette.size
        features.append(area_ratio)
        
        # 2. Centroid
        if np.sum(silhouette) > 0:
            y_coords, x_coords = np.where(silhouette > 0)
            centroid_x = np.mean(x_coords) / self.image_size
            centroid_y = np.mean(y_coords) / self.image_size
        else:
            centroid_x, centroid_y = 0.5, 0.5
        features.extend([centroid_x, centroid_y])
        
        # 3. Bounding box aspect ratio
        if np.sum(silhouette) > 0:
            y_coords, x_coords = np.where(silhouette > 0)
            width = (np.max(x_coords) - np.min(x_coords) + 1) / self.image_size
            height = (np.max(y_coords) - np.min(y_coords) + 1) / self.image_size
            aspect = width / (height + 1e-10)
        else:
            aspect = 1.0
        features.append(aspect)
        
        # 4. Compactness (area / perimeter^2)
        # Approximate perimeter by edge detection
        edges_h = np.abs(np.diff(silhouette, axis=0))
        edges_v = np.abs(np.diff(silhouette, axis=1))
        perimeter = np.sum(edges_h) + np.sum(edges_v)
        compactness = np.sum(silhouette) / (perimeter ** 2 + 1e-10)
        features.append(compactness)
        
        # 5. Moments (Hu moments approximation - first 4)
        if np.sum(silhouette) > 0:
            y_coords, x_coords = np.where(silhouette > 0)
            cx, cy = np.mean(x_coords), np.mean(y_coords)
            
            # Central moments
            mu20 = np.mean((x_coords - cx) ** 2)
            mu02 = np.mean((y_coords - cy) ** 2)
            mu11 = np.mean((x_coords - cx) * (y_coords - cy))
            mu30 = np.mean((x_coords - cx) ** 3)
            
            # Normalize
            n = len(x_coords)
            mu20 /= n
            mu02 /= n
            mu11 /= n
            mu30 /= n
            
            features.extend([mu20, mu02, mu11, mu30])
        else:
            features.extend([0, 0, 0, 0])
        
        return np.array(features, dtype=np.float32)
    
    def compute_fourier_descriptor(self, image: np.ndarray, num_coeffs: int = 16) -> np.ndarray:
        """
        Compute Fourier descriptor from silhouette contour
        """
        # Get silhouette
        gray = np.mean(image, axis=2)
        silhouette = (gray < 250).astype(np.uint8)
        
        # Find contour points (simple edge detection)
        edges_h = np.abs(np.diff(silhouette.astype(float), axis=0))
        edges_v = np.abs(np.diff(silhouette.astype(float), axis=1))
        
        # Get edge points
        edge_points = []
        ey, ex = np.where(edges_h > 0)
        for y, x in zip(ey, ex):
            edge_points.append((x, y))
        ey, ex = np.where(edges_v > 0)
        for y, x in zip(ey, ex):
            edge_points.append((x, y))
        
        if len(edge_points) < num_coeffs * 2:
            return np.zeros(num_coeffs * 2, dtype=np.float32)
        
        # Convert to complex numbers (x + iy)
        edge_points = np.array(edge_points)
        
        # Sort by angle from centroid for better contour representation
        centroid = np.mean(edge_points, axis=0)
        angles = np.arctan2(edge_points[:, 1] - centroid[1], 
                          edge_points[:, 0] - centroid[0])
        sorted_indices = np.argsort(angles)
        edge_points = edge_points[sorted_indices]
        
        # Create complex representation
        z = edge_points[:, 0] + 1j * edge_points[:, 1]
        
        # Compute FFT
        fft = np.fft.fft(z)
        
        # Take magnitude of first N coefficients (excluding DC)
        coeffs = np.abs(fft[1:num_coeffs + 1])
        
        # Normalize by first coefficient for scale invariance
        if coeffs[0] > 1e-10:
            coeffs = coeffs / coeffs[0]
        
        # Also include phase information
        phases = np.angle(fft[1:num_coeffs + 1])
        
        descriptor = np.concatenate([coeffs, phases])
        return descriptor.astype(np.float32)
    
    def compute(self, obj: OBJLoader) -> Dict[str, np.ndarray]:
        """
        Compute multi-view descriptor for the 3D model
        
        Returns concatenated features from all views
        """
        if not self.renderer or not self.renderer.initialized:
            # Return empty descriptor if renderer not available
            return {
                'multiview_histogram': np.zeros(self.num_views * 16, dtype=np.float32),
                'multiview_silhouette': np.zeros(self.num_views * 9, dtype=np.float32),
                'multiview_fourier': np.zeros(self.num_views * 32, dtype=np.float32)
            }
        
        views = self.renderer.render_multi_view(obj, self.num_views)
        
        if len(views) == 0:
            return {
                'multiview_histogram': np.zeros(self.num_views * 16, dtype=np.float32),
                'multiview_silhouette': np.zeros(self.num_views * 9, dtype=np.float32),
                'multiview_fourier': np.zeros(self.num_views * 32, dtype=np.float32)
            }
        
        # Compute descriptors for each view
        histograms = []
        silhouettes = []
        fouriers = []
        
        for view in views:
            histograms.append(self.compute_view_histogram(view))
            silhouettes.append(self.compute_silhouette_descriptor(view))
            fouriers.append(self.compute_fourier_descriptor(view))
        
        return {
            'multiview_histogram': np.concatenate(histograms).astype(np.float32),
            'multiview_silhouette': np.concatenate(silhouettes).astype(np.float32),
            'multiview_fourier': np.concatenate(fouriers).astype(np.float32)
        }


class GeometricDescriptors:
    """
    Compute geometric global descriptors for 3D models
    These capture overall geometric properties of the shape
    """
    
    @staticmethod
    def compute_bounding_box_features(obj: OBJLoader) -> np.ndarray:
        """
        Compute features from the oriented bounding box
        """
        if obj.bounding_box is None:
            return np.zeros(6, dtype=np.float32)
        
        min_coords, max_coords = obj.bounding_box
        
        # After normalization, compute original ratios
        dimensions = (max_coords - min_coords) / obj.scale_factor
        
        # Sort dimensions
        dims_sorted = np.sort(dimensions)
        
        features = [
            dims_sorted[2] / (dims_sorted[0] + 1e-10),  # Max/Min ratio
            dims_sorted[1] / (dims_sorted[0] + 1e-10),  # Mid/Min ratio
            dims_sorted[2] / (dims_sorted[1] + 1e-10),  # Max/Mid ratio
            np.prod(dims_sorted),  # Volume
            2 * (dims_sorted[0] * dims_sorted[1] + 
                 dims_sorted[1] * dims_sorted[2] + 
                 dims_sorted[0] * dims_sorted[2]),  # Surface area
            np.sum(dims_sorted)  # Sum of dimensions
        ]
        
        return np.array(features, dtype=np.float32)
    
    @staticmethod
    def compute_moment_descriptors(obj: OBJLoader) -> np.ndarray:
        """
        Compute 3D moment-based descriptors
        """
        vertices = obj.vertices
        
        if len(vertices) == 0:
            return np.zeros(9, dtype=np.float32)
        
        # Central moments (already centered)
        features = []
        
        # Second order moments (inertia tensor components)
        Ixx = np.mean(vertices[:, 1]**2 + vertices[:, 2]**2)
        Iyy = np.mean(vertices[:, 0]**2 + vertices[:, 2]**2)
        Izz = np.mean(vertices[:, 0]**2 + vertices[:, 1]**2)
        Ixy = -np.mean(vertices[:, 0] * vertices[:, 1])
        Ixz = -np.mean(vertices[:, 0] * vertices[:, 2])
        Iyz = -np.mean(vertices[:, 1] * vertices[:, 2])
        
        features.extend([Ixx, Iyy, Izz, Ixy, Ixz, Iyz])
        
        # Third order moments
        m300 = np.mean(vertices[:, 0]**3)
        m030 = np.mean(vertices[:, 1]**3)
        m003 = np.mean(vertices[:, 2]**3)
        
        features.extend([m300, m030, m003])
        
        return np.array(features, dtype=np.float32)
    
    @staticmethod
    def compute_mesh_statistics(obj: OBJLoader) -> np.ndarray:
        """
        Compute mesh-based statistical features
        """
        features = []
        
        # Number of vertices and faces (normalized)
        num_verts = len(obj.vertices)
        num_faces = len(obj.faces)
        
        features.append(np.log10(num_verts + 1))
        features.append(np.log10(num_faces + 1))
        
        # Vertex density
        surface_area = obj.get_surface_area()
        vertex_density = num_verts / (surface_area + 1e-10)
        features.append(np.log10(vertex_density + 1))
        
        # Face area statistics
        face_areas = []
        for face in obj.faces:
            v0 = obj.vertices[face[0]]
            v1 = obj.vertices[face[1]]
            v2 = obj.vertices[face[2]]
            
            edge1 = v1 - v0
            edge2 = v2 - v0
            area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
            face_areas.append(area)
        
        if len(face_areas) > 0:
            face_areas = np.array(face_areas)
            features.append(np.mean(face_areas))
            features.append(np.std(face_areas))
            features.append(np.median(face_areas))
        else:
            features.extend([0, 0, 0])
        
        return np.array(features, dtype=np.float32)


class Shape3DDescriptorExtractor:
    """
    Main class for extracting all 3D shape descriptors
    """
    
    def __init__(self, use_opengl: bool = True):
        self.use_opengl = use_opengl and OPENGL_AVAILABLE
        self.shape_dist = ShapeDistributionDescriptors(num_samples=5000, num_bins=64)
        self.multiview = None
        
        if self.use_opengl:
            self.multiview = MultiViewDescriptor(num_views=12, image_size=64)
            if not self.multiview.initialize():
                print("Warning: Could not initialize OpenGL renderer")
                self.use_opengl = False
    
    def extract(self, obj_path: str) -> Optional[Dict]:
        """
        Extract all descriptors from an OBJ file
        
        Args:
            obj_path: Path to the .obj file
            
        Returns:
            Dictionary containing all descriptors and metadata
        """
        # Load the model
        obj = OBJLoader(obj_path)
        if not obj.load():
            print(f"Failed to load OBJ file: {obj_path}")
            return None
        
        # Extract shape distribution descriptors
        shape_dist_desc = self.shape_dist.compute_all(obj)
        
        # Extract geometric descriptors
        bbox_features = GeometricDescriptors.compute_bounding_box_features(obj)
        moment_features = GeometricDescriptors.compute_moment_descriptors(obj)
        mesh_stats = GeometricDescriptors.compute_mesh_statistics(obj)
        
        # Extract multi-view descriptors if OpenGL is available
        if self.use_opengl and self.multiview:
            multiview_desc = self.multiview.compute(obj)
        else:
            multiview_desc = {
                'multiview_histogram': np.array([], dtype=np.float32),
                'multiview_silhouette': np.array([], dtype=np.float32),
                'multiview_fourier': np.array([], dtype=np.float32)
            }
        
        # Combine all descriptors
        combined_descriptor = np.concatenate([
            shape_dist_desc['d1'],
            shape_dist_desc['d2'],
            shape_dist_desc['d3'],
            shape_dist_desc['d4'],
            shape_dist_desc['a3'],
            bbox_features,
            moment_features,
            mesh_stats
        ])
        
        # Create result dictionary
        result = {
            'filepath': obj_path,
            'filename': os.path.basename(obj_path),
            'num_vertices': len(obj.vertices),
            'num_faces': len(obj.faces),
            'surface_area': float(obj.get_surface_area()),
            
            # Individual descriptors (for flexible similarity computation)
            'descriptors': {
                'd1': shape_dist_desc['d1'].tolist(),
                'd2': shape_dist_desc['d2'].tolist(),
                'd3': shape_dist_desc['d3'].tolist(),
                'd4': shape_dist_desc['d4'].tolist(),
                'a3': shape_dist_desc['a3'].tolist(),
                'bbox': bbox_features.tolist(),
                'moments': moment_features.tolist(),
                'mesh_stats': mesh_stats.tolist(),
                'multiview_histogram': multiview_desc['multiview_histogram'].tolist(),
                'multiview_silhouette': multiview_desc['multiview_silhouette'].tolist(),
                'multiview_fourier': multiview_desc['multiview_fourier'].tolist()
            },
            
            # Combined descriptor for fast similarity search
            'combined_descriptor': combined_descriptor.tolist()
        }
        
        return result
    
    def cleanup(self):
        """Clean up resources"""
        if self.multiview:
            self.multiview.cleanup()


def compute_similarity(desc1: Dict, desc2: Dict, weights: Dict = None) -> float:
    """
    Compute similarity between two 3D models based on their descriptors
    
    Args:
        desc1: Descriptor dictionary for first model
        desc2: Descriptor dictionary for second model
        weights: Optional weights for different descriptor types
        
    Returns:
        Similarity score (0-1, higher is more similar)
    """
    if weights is None:
        weights = {
            'd1': 0.1,
            'd2': 0.25,  # D2 is most discriminative
            'd3': 0.15,
            'd4': 0.1,
            'a3': 0.15,
            'bbox': 0.05,
            'moments': 0.1,
            'mesh_stats': 0.05,
            'multiview_histogram': 0.025,
            'multiview_silhouette': 0.025,
            'multiview_fourier': 0.0
        }
    
    total_similarity = 0.0
    total_weight = 0.0
    
    for desc_name, weight in weights.items():
        if weight <= 0:
            continue
            
        arr1 = np.array(desc1['descriptors'].get(desc_name, []))
        arr2 = np.array(desc2['descriptors'].get(desc_name, []))
        
        if len(arr1) == 0 or len(arr2) == 0 or len(arr1) != len(arr2):
            continue
        
        # Use histogram intersection for histogram-type descriptors
        if desc_name in ['d1', 'd2', 'd3', 'd4', 'a3', 'multiview_histogram']:
            # Histogram intersection
            similarity = np.sum(np.minimum(arr1, arr2))
        else:
            # Cosine similarity for other descriptors
            norm1 = np.linalg.norm(arr1)
            norm2 = np.linalg.norm(arr2)
            if norm1 > 1e-10 and norm2 > 1e-10:
                similarity = np.dot(arr1, arr2) / (norm1 * norm2)
                similarity = (similarity + 1) / 2  # Map from [-1,1] to [0,1]
            else:
                similarity = 0.0
        
        total_similarity += weight * similarity
        total_weight += weight
    
    if total_weight > 0:
        return total_similarity / total_weight
    return 0.0


def euclidean_distance(desc1: List[float], desc2: List[float]) -> float:
    """Compute Euclidean distance between combined descriptors"""
    arr1 = np.array(desc1)
    arr2 = np.array(desc2)
    
    if len(arr1) != len(arr2):
        return float('inf')
    
    return float(np.linalg.norm(arr1 - arr2))


def chi_square_distance(hist1: List[float], hist2: List[float]) -> float:
    """Compute Chi-square distance between histograms"""
    h1 = np.array(hist1)
    h2 = np.array(hist2)
    
    if len(h1) != len(h2):
        return float('inf')
    
    # Chi-square distance
    denominator = h1 + h2
    with np.errstate(divide='ignore', invalid='ignore'):
        chi_sq = np.where(denominator > 0, 
                         (h1 - h2) ** 2 / denominator, 
                         0)
    
    return float(0.5 * np.sum(chi_sq))


# Main entry point for testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python shape_descriptors_3d.py <obj_file>")
        sys.exit(1)
    
    obj_path = sys.argv[1]
    
    print(f"Extracting descriptors from: {obj_path}")
    
    extractor = Shape3DDescriptorExtractor(use_opengl=True)
    
    try:
        result = extractor.extract(obj_path)
        
        if result:
            print(f"\nModel: {result['filename']}")
            print(f"Vertices: {result['num_vertices']}")
            print(f"Faces: {result['num_faces']}")
            print(f"Surface Area: {result['surface_area']:.4f}")
            print(f"\nDescriptor sizes:")
            for name, desc in result['descriptors'].items():
                print(f"  {name}: {len(desc)} values")
            print(f"\nCombined descriptor: {len(result['combined_descriptor'])} values")
        else:
            print("Failed to extract descriptors")
    finally:
        extractor.cleanup()
