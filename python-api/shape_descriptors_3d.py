"""
3D Shape Descriptors Module using OpenGL for Multi-View Rendering

This module implements content-based 3D shape retrieval methods as described in:
"A survey of content based 3D shape retrieval methods" by Tangelder & Veltkamp

=== GLOBAL FEATURES BASED SIMILARITY (Section 3.1.1) ===
Global features characterize the global shape of a 3D model:
- Volume, surface area, volume-to-surface ratio
- Statistical moments (moment invariants)
- Bounding box features and aspect ratios
- Convex-hull based indices:
  * Hull crumpliness: ratio of object surface area to convex hull surface area
  * Hull packing: percentage of convex hull volume not occupied by object
  * Hull compactness: ratio of cubed hull surface area to squared hull volume

=== GLOBAL FEATURE DISTRIBUTION BASED SIMILARITY (Section 3.1.2) ===
Shape distributions measuring properties from random surface points:
- D1: Distance from centroid to random surface point
- D2: Distance between pairs of random surface points (most discriminative)
- D3: Square root of area of triangle from 3 random points
- D4: Cube root of volume of tetrahedron from 4 random points  
- A3: Angle between 3 random surface points

=== VIEW BASED SIMILARITY (Section 3.3.1) ===
Multi-view 2D projections using OpenGL rendering with:
- Silhouette-based descriptors
- Fourier descriptors of contours
- Histogram-based view features

References:
- Tangelder & Veltkamp, "A survey of content based 3D shape retrieval methods"
- Osada et al., "Shape Distributions"
- Corney et al., convex-hull based indices
- Funkhouser et al., "A search engine for 3D models"
"""

import numpy as np
import os
from typing import Tuple, List, Dict, Optional
import math
from collections import defaultdict

# OpenGL imports (required) - Core rendering engine for all 3D operations
from OpenGL.GL import *
from OpenGL import arrays
# Note: GLU is NOT used - we have manual implementations for gluPerspective and gluLookAt

# For image processing of rendered views
from PIL import Image


# =============================================================================
# OpenGL Helper Functions - Manual implementations to avoid GLU dependency
# =============================================================================

def _set_perspective_matrix(fov_y: float, aspect: float, near: float, far: float):
    """Set perspective projection matrix without gluPerspective (GLU may not be available)"""
    f = 1.0 / math.tan(math.radians(fov_y) / 2.0)
    matrix = np.zeros(16, dtype=np.float32)
    matrix[0] = f / aspect
    matrix[5] = f
    matrix[10] = (far + near) / (near - far)
    matrix[11] = -1.0
    matrix[14] = (2.0 * far * near) / (near - far)
    glLoadMatrixf(matrix)


def _set_look_at_matrix(eye_x: float, eye_y: float, eye_z: float,
                        center_x: float, center_y: float, center_z: float,
                        up_x: float, up_y: float, up_z: float):
    """Set view matrix without gluLookAt (GLU may not be available)"""
    # Forward vector (from eye to center)
    fx = center_x - eye_x
    fy = center_y - eye_y
    fz = center_z - eye_z
    
    # Normalize forward
    f_len = math.sqrt(fx*fx + fy*fy + fz*fz)
    if f_len > 0:
        fx /= f_len
        fy /= f_len
        fz /= f_len
    
    # Side vector = forward x up
    sx = fy * up_z - fz * up_y
    sy = fz * up_x - fx * up_z
    sz = fx * up_y - fy * up_x
    
    # Normalize side
    s_len = math.sqrt(sx*sx + sy*sy + sz*sz)
    if s_len > 0:
        sx /= s_len
        sy /= s_len
        sz /= s_len
    
    # Recompute up = side x forward
    ux = sy * fz - sz * fy
    uy = sz * fx - sx * fz
    uz = sx * fy - sy * fx
    
    # Build rotation matrix (column-major for OpenGL)
    matrix = np.array([
        sx, ux, -fx, 0.0,
        sy, uy, -fy, 0.0,
        sz, uz, -fz, 0.0,
        0.0, 0.0, 0.0, 1.0
    ], dtype=np.float32)
    
    glMultMatrixf(matrix)
    glTranslatef(-eye_x, -eye_y, -eye_z)


# Try to import GLFW for windowed rendering
_GLFW_AVAILABLE = False
_EGL_AVAILABLE = False

try:
    import glfw
    _GLFW_AVAILABLE = True
except ImportError:
    pass

try:
    from OpenGL import EGL
    from OpenGL.EGL import *
    _EGL_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# OpenGL Context Manager - Singleton for managing OpenGL state
# Supports both GLFW (with display) and EGL (headless/Docker)
# =============================================================================

class OpenGLContext:
    """
    Singleton class to manage OpenGL context and state.
    Automatically selects between GLFW (display) and EGL (headless).
    """
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if OpenGLContext._initialized:
            return
        self.window = None
        self.egl_display = None
        self.egl_context = None
        self.egl_surface = None
        self.width = 256
        self.height = 256
        self.use_egl = False
        self.use_software = False
        self._init_opengl()
        OpenGLContext._initialized = True
    
    def _init_opengl(self) -> bool:
        """Initialize OpenGL context - try GLFW first, fall back to EGL or software"""
        # Try GLFW first (for systems with display)
        if _GLFW_AVAILABLE:
            try:
                if glfw.init():
                    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
                    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
                    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
                    
                    self.window = glfw.create_window(self.width, self.height, "3D Engine", None, None)
                    
                    if self.window:
                        glfw.make_context_current(self.window)
                        self._setup_gl_state()
                        print("OpenGL initialized with GLFW")
                        return True
                    else:
                        glfw.terminate()
            except Exception as e:
                print(f"GLFW initialization failed: {e}")
                if self.window:
                    glfw.terminate()
                self.window = None
        
        # Fall back to EGL for headless rendering (Docker)
        if _EGL_AVAILABLE:
            try:
                self.egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY)
                if self.egl_display != EGL_NO_DISPLAY:
                    major, minor = EGLint(), EGLint()
                    if eglInitialize(self.egl_display, major, minor):
                        # Configure EGL for offscreen rendering
                        config_attribs = arrays.GLintArray.asArray([
                            EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                            EGL_BLUE_SIZE, 8,
                            EGL_GREEN_SIZE, 8,
                            EGL_RED_SIZE, 8,
                            EGL_DEPTH_SIZE, 24,
                            EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
                            EGL_NONE
                        ])
                        
                        configs = (EGLConfig * 1)()
                        num_configs = EGLint()
                        
                        if eglChooseConfig(self.egl_display, config_attribs, configs, 1, num_configs):
                            if num_configs.value > 0:
                                eglBindAPI(EGL_OPENGL_API)
                                
                                # Create pbuffer surface
                                pbuffer_attribs = arrays.GLintArray.asArray([
                                    EGL_WIDTH, self.width,
                                    EGL_HEIGHT, self.height,
                                    EGL_NONE
                                ])
                                
                                self.egl_surface = eglCreatePbufferSurface(
                                    self.egl_display, configs[0], pbuffer_attribs
                                )
                                
                                if self.egl_surface != EGL_NO_SURFACE:
                                    context_attribs = arrays.GLintArray.asArray([EGL_NONE])
                                    self.egl_context = eglCreateContext(
                                        self.egl_display, configs[0], EGL_NO_CONTEXT, context_attribs
                                    )
                                    
                                    if self.egl_context != EGL_NO_CONTEXT:
                                        eglMakeCurrent(
                                            self.egl_display, self.egl_surface, 
                                            self.egl_surface, self.egl_context
                                        )
                                        self.use_egl = True
                                        self._setup_gl_state()
                                        print("OpenGL initialized with EGL (headless mode)")
                                        return True
            except Exception as e:
                print(f"EGL initialization failed: {e}")
        
        # Last resort: Use software rendering flag
        print("Warning: No hardware OpenGL available. Using software rendering mode.")
        self.use_software = True
        return True
    
    def _setup_gl_state(self):
        """Setup common OpenGL state"""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_NORMALIZE)
        
        glLightfv(GL_LIGHT0, GL_POSITION, [1.0, 1.0, 1.0, 0.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        
        glClearColor(1.0, 1.0, 1.0, 1.0)
    
    def make_current(self):
        """Make this context current for OpenGL operations"""
        if self.use_software:
            return
        elif self.use_egl and self.egl_context:
            eglMakeCurrent(self.egl_display, self.egl_surface, self.egl_surface, self.egl_context)
        elif self.window:
            glfw.make_context_current(self.window)
    
    def resize(self, width: int, height: int):
        """Resize the rendering viewport"""
        if self.use_software:
            self.width = width
            self.height = height
            return
            
        if width == self.width and height == self.height:
            return
            
        self.width = width
        self.height = height
        
        if self.use_egl:
            # For EGL, we may need to recreate the pbuffer surface
            # For now, just update viewport
            pass
        elif self.window:
            glfw.set_window_size(self.window, width, height)
        
        glViewport(0, 0, width, height)
    
    def is_available(self) -> bool:
        """Check if OpenGL rendering is available"""
        return not self.use_software
    
    def cleanup(self):
        """Clean up OpenGL resources"""
        if self.use_egl:
            if self.egl_context:
                eglDestroyContext(self.egl_display, self.egl_context)
            if self.egl_surface:
                eglDestroySurface(self.egl_display, self.egl_surface)
            if self.egl_display:
                eglTerminate(self.egl_display)
            self.egl_context = None
            self.egl_surface = None
            self.egl_display = None
        elif self.window:
            glfw.destroy_window(self.window)
            self.window = None
            glfw.terminate()
        
        OpenGLContext._initialized = False
        OpenGLContext._instance = None


def get_opengl_context() -> OpenGLContext:
    """Get or create the global OpenGL context"""
    return OpenGLContext()


# =============================================================================
# OpenGL 3D Model - Unified model representation with OpenGL integration
# =============================================================================

class Model3D:
    """
    OpenGL-integrated 3D model class.
    Handles loading, normalization, and provides OpenGL display list for rendering.
    All 3D operations go through this class.
    """
    
    def __init__(self):
        self.filepath = ""
        self.vertices = np.array([], dtype=np.float32)
        self.faces = np.array([], dtype=np.int32)
        self.normals = np.array([], dtype=np.float32)
        self.face_normals = np.array([], dtype=np.float32)
        self.centroid = np.array([0.0, 0.0, 0.0])
        self.bounding_box = None
        self.scale_factor = 1.0
        
        # OpenGL resources
        self._display_list = None
        self._is_loaded = False
        
        # Cached computed values
        self._surface_area = None
        self._volume = None
        self._convex_hull = None
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'Model3D':
        """
        Load a 3D model from an OBJ file using OpenGL context.
        
        Args:
            filepath: Path to the .obj file
            
        Returns:
            Model3D instance or None if loading failed
        """
        model = cls()
        model.filepath = filepath
        
        # Ensure OpenGL context is available
        ctx = get_opengl_context()
        ctx.make_current()
        
        if not model._parse_obj_file(filepath):
            return None
        
        model._normalize()
        model._compute_normals()
        model._create_display_list()
        model._is_loaded = True
        
        return model
    
    def _parse_obj_file(self, filepath: str) -> bool:
        """Parse OBJ file and extract geometry data"""
        vertices = []
        faces = []
        normals = []
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if not parts:
                        continue
                    
                    if parts[0] == 'v' and len(parts) >= 4:
                        try:
                            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                        except (ValueError, IndexError):
                            continue
                    
                    elif parts[0] == 'vn' and len(parts) >= 4:
                        try:
                            normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
                        except (ValueError, IndexError):
                            continue
                    
                    elif parts[0] == 'f':
                        face_verts = []
                        for part in parts[1:]:
                            indices = part.split('/')
                            try:
                                v_idx = int(indices[0]) - 1
                                if v_idx >= 0:
                                    face_verts.append(v_idx)
                            except (ValueError, IndexError):
                                continue
                        
                        if len(face_verts) >= 3:
                            # Triangulate using fan method
                            for i in range(1, len(face_verts) - 1):
                                faces.append([face_verts[0], face_verts[i], face_verts[i + 1]])
            
            if len(vertices) == 0:
                return False
            
            self.vertices = np.array(vertices, dtype=np.float32)
            self.faces = np.array(faces, dtype=np.int32)
            self.normals = np.array(normals, dtype=np.float32) if normals else np.array([])
            
            return True
            
        except Exception as e:
            print(f"Error parsing OBJ file: {e}")
            return False
    
    def _normalize(self):
        """Center and scale model to fit in unit sphere"""
        self.centroid = np.mean(self.vertices, axis=0)
        self.vertices = self.vertices - self.centroid
        
        min_coords = np.min(self.vertices, axis=0)
        max_coords = np.max(self.vertices, axis=0)
        self.bounding_box = (min_coords, max_coords)
        
        max_dist = np.max(np.linalg.norm(self.vertices, axis=1))
        if max_dist > 0:
            self.scale_factor = 1.0 / max_dist
            self.vertices = self.vertices * self.scale_factor
    
    def _compute_normals(self):
        """Compute face normals for OpenGL rendering"""
        face_normals = []
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
                face_normals.append(normal)
            else:
                face_normals.append(np.array([0, 0, 1]))
        
        self.face_normals = np.array(face_normals, dtype=np.float32)
    
    def _create_display_list(self):
        """Create OpenGL display list for efficient rendering"""
        ctx = get_opengl_context()
        
        # Skip display list creation if OpenGL not available
        if not ctx.is_available():
            print("Warning: OpenGL not available, display list not created")
            return
            
        ctx.make_current()
        
        # Delete old display list if exists
        if self._display_list is not None:
            glDeleteLists(self._display_list, 1)
        
        self._display_list = glGenLists(1)
        glNewList(self._display_list, GL_COMPILE)
        
        glColor3f(0.7, 0.7, 0.8)
        glBegin(GL_TRIANGLES)
        
        for i, face in enumerate(self.faces):
            if i < len(self.face_normals):
                glNormal3fv(self.face_normals[i])
            
            for vertex_idx in face:
                if vertex_idx < len(self.vertices):
                    glVertex3fv(self.vertices[vertex_idx])
        
        glEnd()
        glEndList()
    
    def render(self):
        """Render the model using OpenGL display list"""
        if self._display_list is not None:
            glCallList(self._display_list)
    
    def get_surface_area(self) -> float:
        """Calculate total surface area (cached)"""
        if self._surface_area is not None:
            return self._surface_area
        
        total_area = 0.0
        for face in self.faces:
            if len(face) >= 3:
                v0 = self.vertices[face[0]]
                v1 = self.vertices[face[1]]
                v2 = self.vertices[face[2]]
                
                edge1 = v1 - v0
                edge2 = v2 - v0
                area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
                total_area += area
        
        self._surface_area = total_area
        return total_area
    
    def get_volume(self) -> float:
        """Calculate mesh volume using divergence theorem (cached)"""
        if self._volume is not None:
            return self._volume
        
        if len(self.faces) == 0:
            return 0.0
        
        volume = 0.0
        for face in self.faces:
            v0 = self.vertices[face[0]]
            v1 = self.vertices[face[1]]
            v2 = self.vertices[face[2]]
            volume += np.dot(v0, np.cross(v1, v2)) / 6.0
        
        self._volume = abs(volume)
        return self._volume
    
    def get_convex_hull(self) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Compute convex hull (cached)"""
        if self._convex_hull is not None:
            return self._convex_hull
        
        try:
            from scipy.spatial import ConvexHull
            
            if len(self.vertices) < 4:
                self._convex_hull = (np.array([]), np.array([]), 0.0, 0.0)
                return self._convex_hull
            
            hull = ConvexHull(self.vertices)
            self._convex_hull = (
                self.vertices[hull.vertices],
                hull.simplices,
                hull.volume,
                hull.area
            )
            return self._convex_hull
            
        except Exception as e:
            print(f"Convex hull computation failed: {e}")
            self._convex_hull = (np.array([]), np.array([]), 0.0, 0.0)
            return self._convex_hull
    
    def sample_surface_points(self, num_points: int) -> np.ndarray:
        """Sample random points on mesh surface using vectorized area-weighted sampling (fast)"""
        if len(self.faces) == 0:
            return np.array([], dtype=np.float32)
        
        # Vectorized area calculation
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        
        areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        total_area = np.sum(areas)
        
        if total_area == 0:
            return np.array([], dtype=np.float32)
        
        # Sample faces based on area
        probabilities = areas / total_area
        face_indices = np.random.choice(len(self.faces), size=num_points, p=probabilities)
        
        # Vectorized point sampling on triangles
        v0 = self.vertices[self.faces[face_indices, 0]]
        v1 = self.vertices[self.faces[face_indices, 1]]
        v2 = self.vertices[self.faces[face_indices, 2]]
        
        r1 = np.random.random(num_points)[:, np.newaxis]
        r2 = np.random.random(num_points)[:, np.newaxis]
        sqrt_r1 = np.sqrt(r1)
        
        points = (1 - sqrt_r1) * v0 + sqrt_r1 * (1 - r2) * v1 + sqrt_r1 * r2 * v2
        
        return points.astype(np.float32)
    
    # Aliases for backward compatibility
    def compute_volume(self) -> float:
        """Alias for get_volume()"""
        return self.get_volume()
    
    def compute_convex_hull(self) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Alias for get_convex_hull()"""
        return self.get_convex_hull()
    
    def cleanup(self):
        """Release OpenGL resources"""
        if self._display_list is not None:
            try:
                ctx = get_opengl_context()
                ctx.make_current()
                glDeleteLists(self._display_list, 1)
            except:
                pass
            self._display_list = None
    
    def __del__(self):
        self.cleanup()


# Legacy alias for backward compatibility
OBJLoader = Model3D


# =============================================================================
# OpenGL Depth Buffer Renderer - For GPU-based feature extraction
# =============================================================================

class OpenGLDepthRenderer:
    """
    OpenGL-based depth buffer renderer for extracting shape features.
    All computations go through OpenGL rendering pipeline.
    """
    
    def __init__(self, size: int = 128):
        self.size = size
        self.ctx = get_opengl_context()
        self.num_views = 20  # Views for depth sampling
        
    def render_depth(self, model: Model3D, azimuth: float, elevation: float) -> Optional[np.ndarray]:
        """Render depth buffer from a viewpoint using OpenGL"""
        if not self.ctx.is_available():
            return None
            
        self.ctx.make_current()
        self.ctx.resize(self.size, self.size)
        
        try:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            _set_perspective_matrix(45.0, 1.0, 0.1, 100.0)
            
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            
            distance = 3.0
            az_rad = math.radians(azimuth)
            el_rad = math.radians(elevation)
            
            eye_x = distance * math.cos(el_rad) * math.sin(az_rad)
            eye_y = distance * math.sin(el_rad)
            eye_z = distance * math.cos(el_rad) * math.cos(az_rad)
            
            _set_look_at_matrix(eye_x, eye_y, eye_z, 0, 0, 0, 0, 1, 0)
            
            model.render()
            glFlush()
            
            # Read depth buffer
            depth = glReadPixels(0, 0, self.size, self.size, GL_DEPTH_COMPONENT, GL_FLOAT)
            depth_array = np.frombuffer(depth, dtype=np.float32).reshape(self.size, self.size)
            
            return np.flipud(depth_array)
            
        except Exception as e:
            print(f"Depth render error: {e}")
            return None
    
    def get_surface_points_from_depth(self, model: Model3D, num_points: int) -> np.ndarray:
        """
        Sample 3D surface points using OpenGL depth buffer unprojection.
        This uses GPU rendering to find actual surface points.
        """
        if not self.ctx.is_available():
            return np.array([], dtype=np.float32)
        
        all_points = []
        points_per_view = num_points // self.num_views + 1
        golden_ratio = (1 + math.sqrt(5)) / 2
        
        for i in range(self.num_views):
            theta = 2 * math.pi * i / golden_ratio
            phi = math.acos(1 - 2 * (i + 0.5) / self.num_views)
            azimuth = math.degrees(theta) % 360
            elevation = 90 - math.degrees(phi)
            
            depth = self.render_depth(model, azimuth, elevation)
            if depth is None:
                continue
            
            # Find valid depth pixels (not background)
            valid_mask = depth < 0.999
            valid_indices = np.where(valid_mask)
            
            if len(valid_indices[0]) == 0:
                continue
            
            # Sample points from valid pixels
            num_valid = len(valid_indices[0])
            sample_count = min(points_per_view, num_valid)
            sample_idx = np.random.choice(num_valid, sample_count, replace=False)
            
            # Unproject depth to 3D using view parameters
            for idx in sample_idx:
                y, x = valid_indices[0][idx], valid_indices[1][idx]
                z = depth[y, x]
                
                # Convert to normalized device coordinates
                ndc_x = (2.0 * x / self.size) - 1.0
                ndc_y = (2.0 * y / self.size) - 1.0
                ndc_z = 2.0 * z - 1.0
                
                # Approximate world coordinates (simplified unprojection)
                distance = 3.0
                az_rad = math.radians(azimuth)
                el_rad = math.radians(elevation)
                
                # View direction
                view_x = math.cos(el_rad) * math.sin(az_rad)
                view_y = math.sin(el_rad)
                view_z = math.cos(el_rad) * math.cos(az_rad)
                
                # Right vector
                right_x = math.cos(az_rad)
                right_z = -math.sin(az_rad)
                
                # Up vector (cross product)
                up_x = -view_y * right_z
                up_y = view_z * right_x - view_x * (-right_z)
                up_z = view_x * 0 - 0 * right_x
                
                # Linearize depth
                near, far = 0.1, 100.0
                linear_z = (2.0 * near * far) / (far + near - ndc_z * (far - near))
                
                # World position
                scale = linear_z * math.tan(math.radians(22.5))
                world_x = ndc_x * scale * right_x + ndc_y * scale * up_x - linear_z * view_x + distance * view_x
                world_y = ndc_y * scale * up_y - linear_z * view_y + distance * view_y
                world_z = ndc_x * scale * right_z + ndc_y * scale * up_z - linear_z * view_z + distance * view_z
                
                all_points.append([world_x, world_y, world_z])
        
        if len(all_points) == 0:
            return np.array([], dtype=np.float32)
        
        points = np.array(all_points, dtype=np.float32)
        
        # Return requested number of points
        if len(points) > num_points:
            indices = np.random.choice(len(points), num_points, replace=False)
            points = points[indices]
        
        return points


# =============================================================================
# Shape Distribution Descriptors - OpenGL GPU-Accelerated
# =============================================================================

class ShapeDistributionDescriptors:
    """
    Compute Shape Distribution descriptors using OpenGL depth buffer.
    Surface points are sampled via GPU rendering and depth unprojection.
    """
    
    def __init__(self, num_samples: int = 2000, num_bins: int = 64):
        self.num_samples = num_samples
        self.num_bins = num_bins
        self.depth_renderer = OpenGLDepthRenderer(size=128)
    
    def _sample_points(self, model: Model3D, count: int) -> np.ndarray:
        """Sample surface points using OpenGL depth buffer"""
        # Try OpenGL depth-based sampling first
        points = self.depth_renderer.get_surface_points_from_depth(model, count)
        
        # Fallback to model's method if OpenGL fails
        if len(points) < count // 2:
            points = model.sample_surface_points(count)
        
        return points
    
    def compute_d1(self, model: Model3D) -> np.ndarray:
        """D1: Distance from centroid to random surface points (OpenGL sampled)"""
        points = self._sample_points(model, self.num_samples)
        if len(points) == 0:
            return np.zeros(self.num_bins, dtype=np.float32)
        
        distances = np.linalg.norm(points, axis=1)
        hist, _ = np.histogram(distances, bins=self.num_bins, range=(0, 1.5), density=True)
        hist = hist / (np.sum(hist) + 1e-10)
        
        return hist.astype(np.float32)
    
    def compute_d2(self, model: Model3D) -> np.ndarray:
        """D2: Distance between pairs of random surface points (OpenGL sampled)"""
        points = self._sample_points(model, self.num_samples * 2)
        if len(points) < 2:
            return np.zeros(self.num_bins, dtype=np.float32)
        
        half = len(points) // 2
        distances = np.linalg.norm(points[:half] - points[half:half*2], axis=1)
        
        hist, _ = np.histogram(distances, bins=self.num_bins, range=(0, 2.0), density=True)
        hist = hist / (np.sum(hist) + 1e-10)
        
        return hist.astype(np.float32)
    
    def compute_d3(self, model: Model3D) -> np.ndarray:
        """D3: sqrt(area) of triangles from 3 random points (OpenGL sampled)"""
        points = self._sample_points(model, self.num_samples * 3)
        if len(points) < 3:
            return np.zeros(self.num_bins, dtype=np.float32)
        
        third = len(points) // 3
        p1, p2, p3 = points[:third], points[third:third*2], points[third*2:third*3]
        
        crosses = np.cross(p2 - p1, p3 - p1)
        sqrt_areas = np.sqrt(0.5 * np.linalg.norm(crosses, axis=1))
        
        hist, _ = np.histogram(sqrt_areas, bins=self.num_bins, range=(0, 1.5), density=True)
        hist = hist / (np.sum(hist) + 1e-10)
        
        return hist.astype(np.float32)
    
    def compute_d4(self, model: Model3D) -> np.ndarray:
        """D4: cbrt(volume) of tetrahedra from 4 random points (OpenGL sampled)"""
        points = self._sample_points(model, self.num_samples * 4)
        if len(points) < 4:
            return np.zeros(self.num_bins, dtype=np.float32)
        
        quarter = len(points) // 4
        p1 = points[:quarter]
        p2 = points[quarter:quarter*2]
        p3 = points[quarter*2:quarter*3]
        p4 = points[quarter*3:quarter*4]
        
        volumes = np.abs(np.einsum('ij,ij->i', p2 - p1, np.cross(p3 - p1, p4 - p1))) / 6.0
        cbrt_volumes = np.cbrt(volumes)
        
        hist, _ = np.histogram(cbrt_volumes, bins=self.num_bins, range=(0, 0.8), density=True)
        hist = hist / (np.sum(hist) + 1e-10)
        
        return hist.astype(np.float32)
    
    def compute_a3(self, model: Model3D) -> np.ndarray:
        """A3: Angles between 3 random surface points (OpenGL sampled)"""
        points = self._sample_points(model, self.num_samples * 3)
        if len(points) < 3:
            return np.zeros(self.num_bins, dtype=np.float32)
        
        third = len(points) // 3
        p1, p2, p3 = points[:third], points[third:third*2], points[third*2:third*3]
        
        v1, v2 = p1 - p2, p3 - p2
        v1_norm = np.maximum(np.linalg.norm(v1, axis=1, keepdims=True), 1e-10)
        v2_norm = np.maximum(np.linalg.norm(v2, axis=1, keepdims=True), 1e-10)
        
        dots = np.clip(np.einsum('ij,ij->i', v1 / v1_norm, v2 / v2_norm), -1.0, 1.0)
        angles = np.arccos(dots)
        
        hist, _ = np.histogram(angles, bins=self.num_bins, range=(0, np.pi), density=True)
        hist = hist / (np.sum(hist) + 1e-10)
        
        return hist.astype(np.float32)
    
    def compute_all(self, model: Model3D) -> Dict[str, np.ndarray]:
        """Compute all shape distribution descriptors using OpenGL"""
        return {
            'd1': self.compute_d1(model),
            'd2': self.compute_d2(model),
            'd3': self.compute_d3(model),
            'd4': self.compute_d4(model),
            'a3': self.compute_a3(model)
        }


# =============================================================================
# OpenGL Renderer - Multi-view rendering using shared context
# =============================================================================

class OpenGLRenderer:
    """
    OpenGL-based renderer for multi-view 3D model rendering.
    Uses shared OpenGL context from OpenGLContext singleton.
    """
    
    def __init__(self, width: int = 256, height: int = 256):
        self.width = width
        self.height = height
        self.ctx = get_opengl_context()
        self.initialized = True  # Uses shared context
        
    def initialize(self) -> bool:
        """Ensure OpenGL context is ready"""
        self.ctx.make_current()
        self.ctx.resize(self.width, self.height)
        return True
    
    def cleanup(self):
        """No cleanup needed - shared context"""
        pass
    
    def render_view(self, model: Model3D, azimuth: float, elevation: float) -> Optional[np.ndarray]:
        """
        Render the model from a specific viewpoint using OpenGL.
        
        Args:
            model: Model3D instance with OpenGL display list
            azimuth: Horizontal angle in degrees (0-360)
            elevation: Vertical angle in degrees (-90 to 90)
            
        Returns:
            Rendered image as numpy array, or None if OpenGL unavailable
        """
        # Check if OpenGL is available
        if not self.ctx.is_available():
            print("Warning: OpenGL not available for rendering")
            return None
            
        self.ctx.make_current()
        self.ctx.resize(self.width, self.height)
        
        try:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            # Setup projection
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            _set_perspective_matrix(45.0, self.width / self.height, 0.1, 100.0)
            
            # Setup modelview with camera position
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            
            distance = 3.0
            az_rad = math.radians(azimuth)
            el_rad = math.radians(elevation)
            
            eye_x = distance * math.cos(el_rad) * math.sin(az_rad)
            eye_y = distance * math.sin(el_rad)
            eye_z = distance * math.cos(el_rad) * math.cos(az_rad)
            
            _set_look_at_matrix(eye_x, eye_y, eye_z, 0, 0, 0, 0, 1, 0)
            
            # Render using model's display list
            model.render()
            
            glFlush()
            
            # Read pixels from framebuffer
            pixels = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
            image = np.frombuffer(pixels, dtype=np.uint8).reshape(self.height, self.width, 3)
            
            # Flip vertically (OpenGL origin is bottom-left)
            return np.flipud(image)
            
        except Exception as e:
            print(f"Error rendering view: {e}")
            return None
    
    def render_multi_view(self, model: Model3D, num_views: int = 20) -> List[np.ndarray]:
        """
        Render model from multiple viewpoints using golden spiral distribution.
        """
        views = []
        golden_ratio = (1 + math.sqrt(5)) / 2
        
        for i in range(num_views):
            theta = 2 * math.pi * i / golden_ratio
            phi = math.acos(1 - 2 * (i + 0.5) / num_views)
            
            azimuth = math.degrees(theta) % 360
            elevation = 90 - math.degrees(phi)
            
            view = self.render_view(model, azimuth, elevation)
            if view is not None:
                views.append(view)
        
        return views


# =============================================================================
# Multi-View Descriptor - OpenGL-based view rendering and feature extraction
# =============================================================================

class MultiViewDescriptor:
    """
    Compute multi-view based shape descriptors using OpenGL rendering.
    Extracts 2D features from rendered views for 3D shape matching.
    """
    
    def __init__(self, num_views: int = 12, image_size: int = 64):
        self.num_views = num_views
        self.image_size = image_size
        self.renderer = OpenGLRenderer(image_size, image_size)
        
    def initialize(self) -> bool:
        """Initialize OpenGL renderer"""
        return self.renderer.initialize()
    
    def cleanup(self):
        """Cleanup (no-op with shared context)"""
        pass
    
    def compute_view_histogram(self, image: np.ndarray, num_bins: int = 16) -> np.ndarray:
        """Compute grayscale intensity histogram from a view"""
        gray = np.mean(image, axis=2).astype(np.uint8)
        hist, _ = np.histogram(gray, bins=num_bins, range=(0, 255), density=True)
        hist = hist / (np.sum(hist) + 1e-10)
        return hist.astype(np.float32)
    
    def compute_silhouette_descriptor(self, image: np.ndarray) -> np.ndarray:
        """Compute silhouette-based features from a rendered view"""
        gray = np.mean(image, axis=2)
        silhouette = (gray < 250).astype(np.float32)
        
        features = []
        
        # Area ratio
        area_ratio = np.sum(silhouette) / silhouette.size
        features.append(area_ratio)
        
        # Centroid
        if np.sum(silhouette) > 0:
            y_coords, x_coords = np.where(silhouette > 0)
            features.extend([np.mean(x_coords) / self.image_size, np.mean(y_coords) / self.image_size])
        else:
            features.extend([0.5, 0.5])
        
        # Aspect ratio
        if np.sum(silhouette) > 0:
            y_coords, x_coords = np.where(silhouette > 0)
            width = (np.max(x_coords) - np.min(x_coords) + 1) / self.image_size
            height = (np.max(y_coords) - np.min(y_coords) + 1) / self.image_size
            features.append(width / (height + 1e-10))
        else:
            features.append(1.0)
        
        # Compactness
        edges_h = np.abs(np.diff(silhouette, axis=0))
        edges_v = np.abs(np.diff(silhouette, axis=1))
        perimeter = np.sum(edges_h) + np.sum(edges_v)
        features.append(np.sum(silhouette) / (perimeter ** 2 + 1e-10))
        
        # Moments
        if np.sum(silhouette) > 0:
            y_coords, x_coords = np.where(silhouette > 0)
            cx, cy = np.mean(x_coords), np.mean(y_coords)
            n = len(x_coords)
            features.extend([
                np.mean((x_coords - cx) ** 2) / n,
                np.mean((y_coords - cy) ** 2) / n,
                np.mean((x_coords - cx) * (y_coords - cy)) / n,
                np.mean((x_coords - cx) ** 3) / n
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        return np.array(features, dtype=np.float32)
    
    def compute_fourier_descriptor(self, image: np.ndarray, num_coeffs: int = 16) -> np.ndarray:
        """Compute Fourier descriptor from silhouette contour"""
        gray = np.mean(image, axis=2)
        silhouette = (gray < 250).astype(np.uint8)
        
        edges_h = np.abs(np.diff(silhouette.astype(float), axis=0))
        edges_v = np.abs(np.diff(silhouette.astype(float), axis=1))
        
        edge_points = []
        ey, ex = np.where(edges_h > 0)
        edge_points.extend([(x, y) for y, x in zip(ey, ex)])
        ey, ex = np.where(edges_v > 0)
        edge_points.extend([(x, y) for y, x in zip(ey, ex)])
        
        if len(edge_points) < num_coeffs * 2:
            return np.zeros(num_coeffs * 2, dtype=np.float32)
        
        edge_points = np.array(edge_points)
        centroid = np.mean(edge_points, axis=0)
        angles = np.arctan2(edge_points[:, 1] - centroid[1], edge_points[:, 0] - centroid[0])
        edge_points = edge_points[np.argsort(angles)]
        
        z = edge_points[:, 0] + 1j * edge_points[:, 1]
        fft = np.fft.fft(z)
        
        coeffs = np.abs(fft[1:num_coeffs + 1])
        if coeffs[0] > 1e-10:
            coeffs = coeffs / coeffs[0]
        
        phases = np.angle(fft[1:num_coeffs + 1])
        return np.concatenate([coeffs, phases]).astype(np.float32)
    
    def compute(self, model: Model3D) -> Dict[str, np.ndarray]:
        """Compute multi-view descriptor for the 3D model"""
        views = self.renderer.render_multi_view(model, self.num_views)
        
        if len(views) == 0:
            return {
                'multiview_histogram': np.zeros(self.num_views * 16, dtype=np.float32),
                'multiview_silhouette': np.zeros(self.num_views * 9, dtype=np.float32),
                'multiview_fourier': np.zeros(self.num_views * 32, dtype=np.float32)
            }
        
        histograms, silhouettes, fouriers = [], [], []
        
        for view in views:
            histograms.append(self.compute_view_histogram(view))
            silhouettes.append(self.compute_silhouette_descriptor(view))
            fouriers.append(self.compute_fourier_descriptor(view))
        
        return {
            'multiview_histogram': np.concatenate(histograms).astype(np.float32),
            'multiview_silhouette': np.concatenate(silhouettes).astype(np.float32),
            'multiview_fourier': np.concatenate(fouriers).astype(np.float32)
        }


# =============================================================================
# Convex Hull Descriptors - OpenGL depth-based hull features
# =============================================================================

class ConvexHullDescriptors:
    """
    Compute convex-hull based shape descriptors using OpenGL.
    Uses depth buffer analysis for volume/surface estimation.
    Based on Corney et al.'s convex-hull based indices.
    """
    
    def __init__(self):
        self.depth_renderer = OpenGLDepthRenderer(size=64)
    
    def estimate_volume_from_depth(self, model: Model3D) -> float:
        """Estimate volume using OpenGL depth buffer integration"""
        ctx = get_opengl_context()
        if not ctx.is_available():
            return model.compute_volume()  # Fallback
        
        total_volume = 0.0
        num_views = 6  # 6 axis-aligned views
        
        views = [(0, 0), (90, 0), (180, 0), (270, 0), (0, 90), (0, -90)]
        
        for azimuth, elevation in views:
            depth = self.depth_renderer.render_depth(model, azimuth, elevation)
            if depth is None:
                continue
            
            # Valid depth pixels
            valid_mask = depth < 0.999
            valid_depths = depth[valid_mask]
            
            if len(valid_depths) > 0:
                # Estimate volume slice from depth variance
                total_volume += np.sum(1.0 - valid_depths) / len(views)
        
        return total_volume * 2.0  # Scale factor
    
    def estimate_surface_area_from_depth(self, model: Model3D) -> float:
        """Estimate surface area using OpenGL silhouette perimeters"""
        ctx = get_opengl_context()
        if not ctx.is_available():
            return model.get_surface_area()  # Fallback
        
        total_perimeter = 0.0
        num_views = 20
        golden_ratio = (1 + math.sqrt(5)) / 2
        
        for i in range(num_views):
            theta = 2 * math.pi * i / golden_ratio
            phi = math.acos(1 - 2 * (i + 0.5) / num_views)
            azimuth = math.degrees(theta) % 360
            elevation = 90 - math.degrees(phi)
            
            depth = self.depth_renderer.render_depth(model, azimuth, elevation)
            if depth is None:
                continue
            
            # Create silhouette and compute perimeter
            silhouette = (depth < 0.999).astype(np.float32)
            edges_h = np.abs(np.diff(silhouette, axis=0))
            edges_v = np.abs(np.diff(silhouette, axis=1))
            total_perimeter += np.sum(edges_h) + np.sum(edges_v)
        
        # Estimate surface area from average perimeter
        return (total_perimeter / num_views) * 0.1
    
    def compute_hull_descriptors(self, model: Model3D) -> np.ndarray:
        """Compute all convex hull based descriptors using OpenGL (6 features)"""
        features = np.zeros(6, dtype=np.float32)
        
        # Use OpenGL-based estimates
        object_area = self.estimate_surface_area_from_depth(model)
        object_volume = self.estimate_volume_from_depth(model)
        
        # Get hull from scipy (only for comparison ratios)
        _, _, hull_volume, hull_area = model.compute_convex_hull()
        
        if hull_area <= 1e-10 or hull_volume <= 1e-10:
            return features
        
        # Hull crumpliness
        hull_crumpliness = object_area / hull_area
        features[0] = min(hull_crumpliness, 10.0)
        
        # Hull packing
        volume_ratio = object_volume / hull_volume
        features[1] = 1.0 - min(volume_ratio, 1.0)
        
        # Hull compactness (normalized to sphere)
        hull_compactness = (hull_area ** 3) / (hull_volume ** 2)
        features[2] = hull_compactness / (36 * np.pi)
        
        # Volume ratio
        features[3] = min(volume_ratio, 1.0)
        
        # Area ratio
        features[4] = 1.0 / hull_crumpliness if hull_crumpliness > 0 else 0.0
        
        # Sphericity
        sphericity = (np.pi ** (1/3)) * ((6 * hull_volume) ** (2/3)) / hull_area
        features[5] = min(sphericity, 1.0)
        
        return features
    
    def compute_volume_surface_features(self, model: Model3D) -> np.ndarray:
        """Compute volume and surface features using OpenGL (4 features)"""
        features = np.zeros(4, dtype=np.float32)
        
        # Use OpenGL-based estimates
        surface_area = self.estimate_surface_area_from_depth(model)
        volume = self.estimate_volume_from_depth(model)
        
        features[0] = volume
        features[1] = surface_area
        
        if surface_area > 1e-10:
            features[2] = volume / surface_area
            iq = (36 * np.pi * volume ** 2) / (surface_area ** 3)
            features[3] = min(iq, 1.0)
        
        return features


# =============================================================================
# Geometric Descriptors - OpenGL-based bounding box, moments, mesh stats
# =============================================================================

class GeometricDescriptors:
    """
    Compute geometric global descriptors using OpenGL depth analysis.
    """
    
    def __init__(self):
        self.depth_renderer = OpenGLDepthRenderer(size=64)
    
    def compute_bounding_box_from_depth(self, model: Model3D) -> np.ndarray:
        """Compute bounding box features using OpenGL depth rendering (6 features)"""
        ctx = get_opengl_context()
        
        # Get extents from depth buffers
        extents = [0.0, 0.0, 0.0]
        views = [(0, 0), (90, 0), (0, 90)]  # Front, side, top
        
        for idx, (azimuth, elevation) in enumerate(views):
            depth = self.depth_renderer.render_depth(model, azimuth, elevation)
            if depth is None:
                continue
            
            valid_mask = depth < 0.999
            if np.sum(valid_mask) > 0:
                # Extent is proportional to silhouette size
                rows = np.any(valid_mask, axis=1)
                cols = np.any(valid_mask, axis=0)
                row_extent = (np.sum(rows) / self.depth_renderer.size)
                col_extent = (np.sum(cols) / self.depth_renderer.size)
                
                # Depth extent
                valid_depths = depth[valid_mask]
                depth_extent = np.max(valid_depths) - np.min(valid_depths)
                
                extents[idx] = max(row_extent, col_extent, depth_extent)
        
        dims_sorted = np.sort(extents)
        if dims_sorted[0] < 1e-10:
            dims_sorted[0] = 0.1  # Avoid division by zero
        
        return np.array([
            dims_sorted[2] / (dims_sorted[0] + 1e-10),  # Max/Min ratio
            dims_sorted[1] / (dims_sorted[0] + 1e-10),  # Mid/Min ratio
            dims_sorted[2] / (dims_sorted[1] + 1e-10),  # Max/Mid ratio
            np.prod(dims_sorted),                       # Volume
            2 * (dims_sorted[0] * dims_sorted[1] + dims_sorted[1] * dims_sorted[2] + dims_sorted[0] * dims_sorted[2]),
            np.sum(dims_sorted)
        ], dtype=np.float32)
    
    def compute_moment_from_depth(self, model: Model3D) -> np.ndarray:
        """Compute moment descriptors using OpenGL depth buffers (9 features)"""
        ctx = get_opengl_context()
        if not ctx.is_available():
            # Fallback to mesh-based moments
            return self._compute_mesh_moments(model)
        
        # Compute moments from depth buffer projections
        moments = []
        views = [(0, 0), (90, 0), (0, 90)]
        
        for azimuth, elevation in views:
            depth = self.depth_renderer.render_depth(model, azimuth, elevation)
            if depth is None:
                moments.extend([0, 0, 0])
                continue
            
            valid_mask = depth < 0.999
            if np.sum(valid_mask) == 0:
                moments.extend([0, 0, 0])
                continue
            
            y_coords, x_coords = np.where(valid_mask)
            cx = np.mean(x_coords) / self.depth_renderer.size
            cy = np.mean(y_coords) / self.depth_renderer.size
            
            # 2D moments from silhouette
            mu20 = np.mean((x_coords / self.depth_renderer.size - cx) ** 2)
            mu02 = np.mean((y_coords / self.depth_renderer.size - cy) ** 2)
            mu11 = np.mean((x_coords / self.depth_renderer.size - cx) * 
                          (y_coords / self.depth_renderer.size - cy))
            
            moments.extend([mu20, mu02, mu11])
        
        return np.array(moments, dtype=np.float32)
    
    def _compute_mesh_moments(self, model: Model3D) -> np.ndarray:
        """Fallback mesh-based moment computation"""
        vertices = model.vertices
        if len(vertices) == 0:
            return np.zeros(9, dtype=np.float32)
        
        Ixx = np.mean(vertices[:, 1]**2 + vertices[:, 2]**2)
        Iyy = np.mean(vertices[:, 0]**2 + vertices[:, 2]**2)
        Izz = np.mean(vertices[:, 0]**2 + vertices[:, 1]**2)
        Ixy = -np.mean(vertices[:, 0] * vertices[:, 1])
        Ixz = -np.mean(vertices[:, 0] * vertices[:, 2])
        Iyz = -np.mean(vertices[:, 1] * vertices[:, 2])
        m300 = np.mean(vertices[:, 0]**3)
        m030 = np.mean(vertices[:, 1]**3)
        m003 = np.mean(vertices[:, 2]**3)
        
        return np.array([Ixx, Iyy, Izz, Ixy, Ixz, Iyz, m300, m030, m003], dtype=np.float32)
    
    def compute_depth_statistics(self, model: Model3D) -> np.ndarray:
        """Compute statistics from OpenGL depth buffer (6 features)"""
        ctx = get_opengl_context()
        
        all_depths = []
        all_areas = []
        num_views = 12
        golden_ratio = (1 + math.sqrt(5)) / 2
        
        for i in range(num_views):
            theta = 2 * math.pi * i / golden_ratio
            phi = math.acos(1 - 2 * (i + 0.5) / num_views)
            azimuth = math.degrees(theta) % 360
            elevation = 90 - math.degrees(phi)
            
            depth = self.depth_renderer.render_depth(model, azimuth, elevation)
            if depth is None:
                continue
            
            valid_mask = depth < 0.999
            valid_depths = depth[valid_mask]
            
            if len(valid_depths) > 0:
                all_depths.extend(valid_depths.tolist())
                all_areas.append(np.sum(valid_mask) / valid_mask.size)
        
        features = []
        
        if len(all_depths) > 0:
            all_depths = np.array(all_depths)
            features.append(np.log10(len(all_depths) + 1))  # Point count
            features.append(np.mean(all_depths))             # Mean depth
            features.append(np.std(all_depths))              # Depth variance
        else:
            features.extend([0, 0, 0])
        
        if len(all_areas) > 0:
            features.append(np.mean(all_areas))              # Mean silhouette area
            features.append(np.std(all_areas))               # Area variance
            features.append(np.median(all_areas))            # Median area
        else:
            features.extend([0, 0, 0])
        
        return np.array(features, dtype=np.float32)
    
    # Static method aliases for backward compatibility
    @staticmethod
    def compute_bounding_box_features(model: Model3D) -> np.ndarray:
        return GeometricDescriptors().compute_bounding_box_from_depth(model)
    
    @staticmethod
    def compute_moment_descriptors(model: Model3D) -> np.ndarray:
        return GeometricDescriptors().compute_moment_from_depth(model)
    
    @staticmethod
    def compute_mesh_statistics(model: Model3D) -> np.ndarray:
        return GeometricDescriptors().compute_depth_statistics(model)


# =============================================================================
# Main Descriptor Extractor - Orchestrates all OpenGL-based extraction
# =============================================================================

class Shape3DDescriptorExtractor:
    """
    Main class for extracting all 3D shape descriptors using OpenGL.
    All 3D operations go through OpenGL rendering pipeline:
    - Surface point sampling via depth buffer unprojection
    - Volume/surface estimation via depth integration
    - Bounding box/moments from silhouette analysis
    - Multi-view features from rendered images
    """
    
    def __init__(self):
        # All extractors use OpenGL depth buffer rendering
        self.shape_dist = ShapeDistributionDescriptors(num_samples=2000, num_bins=64)
        self.multiview = MultiViewDescriptor(num_views=8, image_size=64)
        self.hull_desc = ConvexHullDescriptors()
        self.geo_desc = GeometricDescriptors()
        
        if not self.multiview.initialize():
            raise RuntimeError("Failed to initialize OpenGL renderer. Ensure OpenGL is properly installed.")
    
    def extract(self, obj_path: str) -> Optional[Dict]:
        """
        Extract all descriptors from an OBJ file using OpenGL-based processing.
        
        Args:
            obj_path: Path to the .obj file
            
        Returns:
            Dictionary containing all descriptors and metadata
        """
        # Load model using OpenGL-integrated Model3D class
        model = Model3D.load_from_file(obj_path)
        if model is None:
            print(f"Failed to load OBJ file: {obj_path}")
            return None
        
        # Extract shape distribution descriptors using OpenGL depth sampling
        shape_dist_desc = self.shape_dist.compute_all(model)
        
        # Extract geometric descriptors using OpenGL depth analysis
        bbox_features = self.geo_desc.compute_bounding_box_from_depth(model)
        moment_features = self.geo_desc.compute_moment_from_depth(model)
        mesh_stats = self.geo_desc.compute_depth_statistics(model)
        
        # Extract convex hull descriptors using OpenGL volume estimation
        hull_features = self.hull_desc.compute_hull_descriptors(model)
        volume_surface_features = self.hull_desc.compute_volume_surface_features(model)
        
        # Extract multi-view descriptors using OpenGL rendering (View-based - Section 3.3.1)
        multiview_desc = self.multiview.compute(model)
        
        # Combine all descriptors
        combined_descriptor = np.concatenate([
            shape_dist_desc['d1'],
            shape_dist_desc['d2'],
            shape_dist_desc['d3'],
            shape_dist_desc['d4'],
            shape_dist_desc['a3'],
            bbox_features,
            moment_features,
            mesh_stats,
            hull_features,
            volume_surface_features
        ])
        
        # Create result dictionary
        result = {
            'filepath': obj_path,
            'filename': os.path.basename(obj_path),
            'num_vertices': len(model.vertices),
            'num_faces': len(model.faces),
            'surface_area': float(model.get_surface_area()),
            'volume': float(model.compute_volume()),
            
            # Individual descriptors (for flexible similarity computation)
            'descriptors': {
                # Global Feature Distribution descriptors (Section 3.1.2)
                'd1': shape_dist_desc['d1'].tolist(),
                'd2': shape_dist_desc['d2'].tolist(),
                'd3': shape_dist_desc['d3'].tolist(),
                'd4': shape_dist_desc['d4'].tolist(),
                'a3': shape_dist_desc['a3'].tolist(),
                # Global Features descriptors (Section 3.1.1)
                'bbox': bbox_features.tolist(),
                'moments': moment_features.tolist(),
                'mesh_stats': mesh_stats.tolist(),
                'hull_features': hull_features.tolist(),
                'volume_surface': volume_surface_features.tolist(),
                # View-based descriptors (Section 3.3.1)
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
        self.multiview.cleanup()


def compute_similarity(desc1: Dict, desc2: Dict, weights: Dict = None) -> float:
    """
    Compute similarity between two 3D models based on their descriptors
    
    Weights are distributed across three categories from the survey paper:
    - Global Feature Distribution (Section 3.1.2): D1, D2, D3, D4, A3
    - Global Features (Section 3.1.1): bbox, moments, mesh_stats, hull_features, volume_surface
    - View-based (Section 3.3.1): multiview descriptors
    
    Args:
        desc1: Descriptor dictionary for first model
        desc2: Descriptor dictionary for second model
        weights: Optional weights for different descriptor types
        
    Returns:
        Similarity score (0-1, higher is more similar)
    """
    if weights is None:
        weights = {
            # Global Feature Distribution descriptors (Section 3.1.2) - 50% weight
            'd1': 0.05,
            'd2': 0.20,  # D2 is most discriminative according to Osada et al.
            'd3': 0.10,
            'd4': 0.05,
            'a3': 0.10,
            # Global Features descriptors (Section 3.1.1) - 35% weight
            'bbox': 0.05,
            'moments': 0.08,
            'mesh_stats': 0.02,
            'hull_features': 0.12,  # Convex hull indices (Corney et al.)
            'volume_surface': 0.08,  # Volume/surface based features
            # View-based descriptors (Section 3.3.1) - 15% weight
            'multiview_histogram': 0.05,
            'multiview_silhouette': 0.07,
            'multiview_fourier': 0.03
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


# =============================================================================
# Thumbnail Generator - OpenGL-based 2D preview rendering
# =============================================================================

class ThumbnailGenerator:
    """
    Generate 2D thumbnail images from 3D OBJ models using OpenGL rendering.
    Uses shared OpenGL context for efficient rendering.
    """
    
    def __init__(self, width: int = 256, height: int = 256):
        self.width = width
        self.height = height
        self.renderer = OpenGLRenderer(width, height)
        self.initialized = True  # Uses shared context
        
    def initialize(self) -> bool:
        """Initialize (always returns True with shared context)"""
        return self.renderer.initialize()
    
    def cleanup(self):
        """Cleanup (no-op with shared context)"""
        pass
    
    def generate(self, obj_path: str, output_path: str, 
                 azimuth: float = 45.0, elevation: float = 30.0) -> bool:
        """
        Generate a thumbnail image for a 3D model.
        
        Args:
            obj_path: Path to the .obj file
            output_path: Path where to save the thumbnail
            azimuth: Horizontal viewing angle in degrees
            elevation: Vertical viewing angle in degrees
            
        Returns:
            True if successful, False otherwise
        """
        # Load model using OpenGL-integrated Model3D
        model = Model3D.load_from_file(obj_path)
        if model is None:
            print(f"Failed to load OBJ file: {obj_path}")
            return False
        
        # Render the view using OpenGL
        image = self.renderer.render_view(model, azimuth, elevation)
        
        if image is None:
            print(f"Failed to render view for: {obj_path}")
            return False
        
        # Save the image
        try:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            pil_image = Image.fromarray(image)
            pil_image.save(output_path, quality=90)
            return True
            
        except Exception as e:
            print(f"Failed to save thumbnail: {e}")
            return False
    
    def generate_multi_angle(self, obj_path: str, output_dir: str, 
                             basename: str = None, num_views: int = 4) -> List[str]:
        """
        Generate multiple thumbnail images from different angles.
        
        Args:
            obj_path: Path to the .obj file
            output_dir: Directory where to save the thumbnails
            basename: Base name for output files
            num_views: Number of views to generate
            
        Returns:
            List of paths to generated thumbnails
        """
        # Load model using OpenGL-integrated Model3D
        model = Model3D.load_from_file(obj_path)
        if model is None:
            return []
        
        if basename is None:
            basename = os.path.splitext(os.path.basename(obj_path))[0]
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate views at different angles
        generated = []
        angles = [(45, 30), (135, 30), (225, 30), (315, 30), (0, 60), (90, 0)][:num_views]
        
        for i, (azimuth, elevation) in enumerate(angles):
            image = self.renderer.render_view(model, azimuth, elevation)
            
            if image is not None:
                output_path = os.path.join(output_dir, f"{basename}_view{i}.png")
                try:
                    pil_image = Image.fromarray(image)
                    pil_image.save(output_path, quality=90)
                    generated.append(output_path)
                except Exception as e:
                    print(f"Failed to save view {i}: {e}")
        
        return generated


# Global thumbnail generator instance (lazy initialization)
_thumbnail_generator: Optional[ThumbnailGenerator] = None


def get_thumbnail_generator() -> ThumbnailGenerator:
    """Get or create the global thumbnail generator instance"""
    global _thumbnail_generator
    if _thumbnail_generator is None:
        _thumbnail_generator = ThumbnailGenerator(width=256, height=256)
        _thumbnail_generator.initialize()
    return _thumbnail_generator


def generate_thumbnail(obj_path: str, output_path: str) -> bool:
    """
    Convenience function to generate a single thumbnail.
    
    Args:
        obj_path: Path to the .obj file
        output_path: Path where to save the thumbnail
        
    Returns:
        True if successful, False otherwise
    """
    generator = get_thumbnail_generator()
    return generator.generate(obj_path, output_path)


# Main entry point for testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python shape_descriptors_3d.py <obj_file>")
        sys.exit(1)
    
    obj_path = sys.argv[1]
    
    print(f"Extracting descriptors from: {obj_path}")
    print("=" * 60)
    
    extractor = Shape3DDescriptorExtractor()
    
    try:
        result = extractor.extract(obj_path)
        
        if result:
            print(f"\nModel: {result['filename']}")
            print(f"Vertices: {result['num_vertices']}")
            print(f"Faces: {result['num_faces']}")
            print(f"Surface Area: {result['surface_area']:.4f}")
            print(f"Volume: {result['volume']:.4f}")
            
            print(f"\n--- Descriptor Categories (from Survey Paper) ---")
            
            print(f"\n[Global Feature Distribution - Section 3.1.2]")
            for name in ['d1', 'd2', 'd3', 'd4', 'a3']:
                desc = result['descriptors'].get(name, [])
                print(f"  {name}: {len(desc)} values")
            
            print(f"\n[Global Features - Section 3.1.1]")
            for name in ['bbox', 'moments', 'mesh_stats', 'hull_features', 'volume_surface']:
                desc = result['descriptors'].get(name, [])
                print(f"  {name}: {len(desc)} values")
            
            print(f"\n[View-based - Section 3.3.1]")
            for name in ['multiview_histogram', 'multiview_silhouette', 'multiview_fourier']:
                desc = result['descriptors'].get(name, [])
                print(f"  {name}: {len(desc)} values")
            
            print(f"\nCombined descriptor: {len(result['combined_descriptor'])} values")
        else:
            print("Failed to extract descriptors")
    finally:
        extractor.cleanup()
