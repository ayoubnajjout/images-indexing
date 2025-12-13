"""
Visual Descriptor Extraction module.

Extracts:
- Color descriptors: HS histograms, dominant colors, LAB color moments
- Texture descriptors: Tamura features, Gabor filters
- Shape descriptors: Hu moments, orientation histograms, Fourier descriptors, shape metrics
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.stats import skew
from scipy.spatial.distance import euclidean
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans
from collections import Counter


class DescriptorExtractor:
    """Extract visual descriptors from images."""
    
    def __init__(self):
        """Initialize the descriptor extractor."""
        pass
    
    # ==================== PREPROCESSING ====================
    
    def preprocess_image(self, image: np.ndarray, size=(128, 128), gamma=2.2) -> np.ndarray:
        """Preprocess image for descriptor extraction."""
        image = cv2.resize(image, size)
        image = cv2.GaussianBlur(image, (3, 3), 0.5)
        image = np.power(image / 255.0, 1/gamma)
        image = np.uint8(image * 255)
        return image
    
    def extract_all(self, image: np.ndarray) -> dict:
        """
        Extract all visual descriptors from an image.
        
        Args:
            image: BGR image as numpy array.
            
        Returns:
            Dictionary containing color, texture, and shape descriptors.
        """
        if image is None or image.size == 0:
            return None
        
        # Ensure minimum size
        if image.shape[0] < 10 or image.shape[1] < 10:
            return None
        
        return {
            "color": self.extract_color_descriptors(image),
            "texture": self.extract_texture_descriptors(image),
            "shape": self.extract_shape_descriptors(image)
        }
    
    # ==================== COLOR DESCRIPTORS ====================
    
    def extract_color_descriptors(self, image: np.ndarray) -> dict:
        """Extract color-based descriptors."""
        # Preprocess image
        preprocessed = self.preprocess_image(image)
        
        # Convert to RGB for dominant color extraction
        rgb = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2HSV)
        
        return {
            "histogram": self._color_histogram(hsv),
            "hsHistogram": self._hs_histogram(preprocessed),
            "dominantColors": self._dominant_colors(rgb),
            "dominantColorsHS": self._dominant_colors_hs(preprocessed),
            "labColorMoments": self._lab_color_moments(preprocessed)
        }
    
    def _color_histogram(self, hsv_image: np.ndarray, bins: int = 16) -> list:
        """
        Compute color histogram in HSV space.
        
        Returns normalized histogram as a list.
        """
        # Compute histogram for H, S, V channels
        h_hist = cv2.calcHist([hsv_image], [0], None, [bins], [0, 180])
        s_hist = cv2.calcHist([hsv_image], [1], None, [bins], [0, 256])
        v_hist = cv2.calcHist([hsv_image], [2], None, [bins], [0, 256])
        
        # Normalize
        h_hist = cv2.normalize(h_hist, h_hist).flatten()
        s_hist = cv2.normalize(s_hist, s_hist).flatten()
        v_hist = cv2.normalize(v_hist, v_hist).flatten()
        
        # Concatenate into single feature vector
        histogram = np.concatenate([h_hist, s_hist, v_hist])
        return [round(float(v), 4) for v in histogram]
    
    def _hs_histogram(self, image: np.ndarray, bins=(30, 32)) -> list:
        """
        Compute 2D Hue-Saturation histogram.
        
        Returns flattened normalized histogram.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, _ = cv2.split(hsv)
        hist = cv2.calcHist([h, s], [0, 1], None, list(bins), [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return [round(float(v), 6) for v in hist]
    
    def _dominant_colors(self, rgb_image: np.ndarray, k: int = 4) -> list:
        """
        Extract dominant colors using K-means clustering on RGB.
        
        Returns list of dominant colors with percentages.
        """
        # Reshape image to be a list of pixels
        pixels = rgb_image.reshape(-1, 3)
        
        # Sample pixels if image is large
        if len(pixels) > 10000:
            indices = np.random.choice(len(pixels), 10000, replace=False)
            pixels = pixels[indices]
        
        # Apply K-means
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get cluster centers (colors) and labels
            colors = kmeans.cluster_centers_.astype(int)
            labels = kmeans.labels_
            
            # Count pixels per cluster
            counts = Counter(labels)
            total = sum(counts.values())
            
            # Build result
            dominant = []
            for i in range(k):
                r, g, b = colors[i]
                hex_color = f"#{r:02x}{g:02x}{b:02x}"
                percentage = round(counts[i] / total * 100, 1)
                color_name = self._get_color_name(r, g, b)
                
                dominant.append({
                    "color": hex_color.upper(),
                    "percentage": percentage,
                    "name": color_name
                })
            
            # Sort by percentage
            dominant.sort(key=lambda x: x["percentage"], reverse=True)
            return dominant
        except Exception:
            return []
    
    def _dominant_colors_hs(self, image: np.ndarray, k: int = 12) -> list:
        """
        Extract dominant colors using K-means clustering on HS (Hue-Saturation).
        
        Returns list of cluster centers in HS space.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3)[:, :2]  # Take only H and S
        
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(pixels)
            centers = kmeans.cluster_centers_
            return [[round(float(c[0]), 2), round(float(c[1]), 2)] for c in centers]
        except Exception:
            return []
    
    def _lab_color_moments(self, image: np.ndarray) -> list:
        """
        Compute color moments (mean, variance, skewness) in LAB color space.
        
        Returns 9 values (3 moments x 3 channels).
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        moments = []
        for i in range(3):
            channel = lab[:, :, i]
            mean_val = np.mean(channel)
            var_val = np.var(channel)
            skew_val = skew(channel.flatten())
            moments.extend([round(float(mean_val), 4), round(float(var_val), 4), round(float(skew_val), 4)])
        return moments
    
    def _get_color_name(self, r: int, g: int, b: int) -> str:
        """Get approximate color name from RGB values."""
        # Simple color naming based on HSV
        h, s, v = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
        
        if v < 40:
            return "Black"
        if s < 40:
            if v > 200:
                return "White"
            return "Gray"
        
        if h < 10 or h > 170:
            return "Red"
        elif h < 25:
            return "Orange"
        elif h < 35:
            return "Yellow"
        elif h < 85:
            return "Green"
        elif h < 130:
            return "Blue"
        elif h < 145:
            return "Purple"
        elif h < 170:
            return "Pink"
        
        return "Unknown"
    
    # ==================== TEXTURE DESCRIPTORS ====================
    
    def extract_texture_descriptors(self, image: np.ndarray) -> dict:
        """Extract texture-based descriptors."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        return {
            "tamura": self._tamura_features(gray),
            "gabor": self._gabor_features(gray)
        }
    
    def _tamura_features(self, gray: np.ndarray) -> dict:
        """
        Compute Tamura texture features.
        
        Returns coarseness, contrast, and directionality.
        """
        # Coarseness
        coarseness = self._tamura_coarseness(gray)
        
        # Contrast
        contrast = self._tamura_contrast(gray)
        
        # Directionality
        directionality = self._tamura_directionality(gray)
        
        return {
            "coarseness": round(float(coarseness), 4),
            "contrast": round(float(contrast), 4),
            "directionality": round(float(directionality), 4)
        }
    
    def _tamura_coarseness(self, gray: np.ndarray) -> float:
        """Compute Tamura coarseness."""
        h, w = gray.shape
        kmax = min(5, int(np.log2(min(h, w))) - 1)
        
        if kmax < 1:
            return 0.0
        
        # Compute average at different scales
        S = np.zeros((h, w, kmax))
        
        for k in range(kmax):
            size = 2 ** (k + 1)
            kernel = np.ones((size, size)) / (size * size)
            S[:, :, k] = cv2.filter2D(gray.astype(float), -1, kernel)
        
        # Compute differences
        E = np.zeros((h, w, kmax))
        for k in range(kmax):
            size = 2 ** k
            # Horizontal difference
            E_h = np.abs(np.roll(S[:, :, k], size, axis=1) - np.roll(S[:, :, k], -size, axis=1))
            # Vertical difference
            E_v = np.abs(np.roll(S[:, :, k], size, axis=0) - np.roll(S[:, :, k], -size, axis=0))
            E[:, :, k] = np.maximum(E_h, E_v)
        
        # Find best scale for each pixel
        k_best = np.argmax(E, axis=2)
        
        # Compute coarseness
        coarseness = np.mean(2 ** k_best)
        return coarseness / (2 ** kmax)  # Normalize
    
    def _tamura_contrast(self, gray: np.ndarray) -> float:
        """Compute Tamura contrast."""
        # Standard deviation
        std = np.std(gray)
        
        # Fourth moment (kurtosis)
        mean = np.mean(gray)
        n = gray.size
        mu4 = np.sum((gray - mean) ** 4) / n
        
        if std == 0:
            return 0.0
        
        kurtosis = mu4 / (std ** 4)
        
        if kurtosis == 0:
            return 0.0
        
        contrast = std / (kurtosis ** 0.25)
        return min(contrast / 128, 1.0)  # Normalize
    
    def _tamura_directionality(self, gray: np.ndarray) -> float:
        """Compute Tamura directionality."""
        # Compute gradients
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute gradient magnitude and direction
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        
        # Threshold
        threshold = np.mean(magnitude) * 0.5
        mask = magnitude > threshold
        
        if not np.any(mask):
            return 0.0
        
        # Compute direction histogram
        direction = np.arctan2(gy, gx)
        direction = direction[mask]
        
        # Histogram of directions
        hist, _ = np.histogram(direction, bins=16, range=(-np.pi, np.pi))
        hist = hist / hist.sum()
        
        # Compute entropy as measure of directionality
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log(hist))
        max_entropy = np.log(16)
        
        directionality = 1 - (entropy / max_entropy)
        return directionality
    
    def _gabor_features(self, gray: np.ndarray, num_orientations: int = 4, num_scales: int = 2) -> list:
        """
        Compute Gabor filter responses.
        
        Returns mean responses for different orientations and scales.
        """
        features = []
        
        for scale in range(num_scales):
            sigma = 3 + scale * 2
            wavelength = 10 + scale * 5
            
            for orientation in range(num_orientations):
                theta = orientation * np.pi / num_orientations
                
                # Create Gabor kernel
                kernel = cv2.getGaborKernel(
                    (21, 21), sigma, theta, wavelength, 0.5, 0, ktype=cv2.CV_64F
                )
                
                # Apply filter
                filtered = cv2.filter2D(gray, cv2.CV_64F, kernel)
                
                # Compute mean and std of response
                mean_response = np.mean(np.abs(filtered))
                features.append(round(float(mean_response) / 255, 4))
        
        return features
    
    # ==================== SHAPE DESCRIPTORS ====================
    
    def extract_shape_descriptors(self, image: np.ndarray) -> dict:
        """Extract shape-based descriptors."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Get the main contour for contour-based descriptors
        contour = self._get_main_contour(gray)
        
        result = {
            "huMoments": self._hu_moments(gray),
            "orientationHistogram": self._orientation_histogram(gray)
        }
        
        # Add contour-based descriptors if contour found
        if contour is not None:
            result["contourHuMoments"] = self._hu_moments_from_contour(contour)
            result["orientationHistogramContour"] = self._orientation_histogram_contour(contour)
            result["shapeMetrics"] = self._shape_metrics(contour)
            result["fourierDescriptors"] = self._fourier_descriptors(contour)
        
        return result
    
    def _get_main_contour(self, gray: np.ndarray):
        """Extract the main contour from grayscale image."""
        # Apply threshold (Otsu)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            # Try Canny edge detection as fallback
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return None
        
        # Return the largest contour
        return max(contours, key=cv2.contourArea)
    
    def _hu_moments(self, gray: np.ndarray) -> list:
        """
        Compute Hu moments from grayscale image.
        
        Returns 7 Hu moment invariants.
        """
        # Compute moments
        moments = cv2.moments(gray)
        
        # Compute Hu moments
        hu = cv2.HuMoments(moments).flatten()
        
        # Log transform for better numerical stability
        hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
        
        return [round(float(h), 4) for h in hu_log]
    
    def _hu_moments_from_contour(self, contour) -> list:
        """
        Compute Hu moments from a contour mask.
        
        Returns 7 Hu moment invariants.
        """
        # Create a mask from the contour
        pts = contour[:, 0, :]
        min_x, min_y = np.min(pts[:, 0]), np.min(pts[:, 1])
        max_x, max_y = np.max(pts[:, 0]), np.max(pts[:, 1])
        
        # Create mask
        mask = np.zeros((max_y - min_y + 1, max_x - min_x + 1), dtype=np.uint8)
        shifted = (pts - np.array([min_x, min_y])).astype(np.int32)
        cv2.drawContours(mask, [shifted], -1, 255, thickness=-1)
        
        # Compute moments
        moments = cv2.moments(mask)
        hu = cv2.HuMoments(moments).flatten()
        
        # Log transform
        eps = 1e-12
        hu_signed_log = []
        for h in hu:
            val = float(h)
            if abs(val) < eps:
                hu_signed_log.append(0.0)
            else:
                hu_signed_log.append(round(-np.sign(val) * np.log10(abs(val) + eps), 4))
        return hu_signed_log
    
    def _orientation_histogram(self, gray: np.ndarray, bins: int = 8) -> list:
        """
        Compute orientation histogram from edge detection.
        
        Returns normalized histogram of edge orientations.
        """
        # Find edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return [0.0] * bins
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        if len(largest_contour) < 3:
            return [0.0] * bins
        
        # Compute orientations along contour
        orientations = []
        for i in range(len(largest_contour)):
            p1 = largest_contour[i][0]
            p2 = largest_contour[(i + 1) % len(largest_contour)][0]
            
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            if dx != 0 or dy != 0:
                angle = np.arctan2(dy, dx)
                orientations.append(angle)
        
        if not orientations:
            return [0.0] * bins
        
        # Create histogram
        hist, _ = np.histogram(orientations, bins=bins, range=(-np.pi, np.pi))
        
        # Normalize
        total = hist.sum()
        if total > 0:
            hist = hist / total
        
        return [round(float(h), 4) for h in hist]
    
    def _orientation_histogram_contour(self, contour, bins: int = 16) -> list:
        """
        Compute orientation histogram from contour with resampling.
        
        Returns normalized histogram of edge orientations.
        """
        pts = contour[:, 0, :].astype(float)
        if len(pts) < 2:
            return [0.0] * bins
        
        # Resample for smoothing
        def resample_pts(pts, m):
            d = np.sqrt(((np.diff(pts, axis=0))**2).sum(axis=1))
            d = np.concatenate(([0.0], d))
            s = np.cumsum(d)
            if s[-1] == 0:
                return np.tile(pts[0], (m, 1))
            s_norm = s / s[-1]
            t = np.linspace(0, 1, m)
            x = np.interp(t, s_norm, pts[:, 0])
            y = np.interp(t, s_norm, pts[:, 1])
            return np.vstack((x, y)).T
        
        rpts = resample_pts(pts, max(128, len(pts)))
        diffs = np.diff(rpts, axis=0)
        angles = np.arctan2(diffs[:, 1], diffs[:, 0])
        angles = np.rad2deg(angles) % 360
        hist, _ = np.histogram(angles, bins=bins, range=(0, 360))
        s = np.sum(hist)
        if s == 0:
            return [0.0] * bins
        hist = hist.astype(float) / s
        return [round(float(h), 4) for h in hist]
    
    def _shape_metrics(self, contour) -> dict:
        """
        Compute shape metrics: solidity, aspect_ratio, compactness.
        """
        area = cv2.contourArea(contour)
        if area == 0:
            return {"solidity": 0.0, "aspectRatio": 1.0, "compactness": 1.0}
        
        # Solidity: area / convex hull area
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0
        
        # Aspect ratio: width / height of bounding rect
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / float(h) if h > 0 else 1.0
        
        # Compactness: 4π*area / perimeter²
        perimeter = cv2.arcLength(contour, True)
        compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 1.0
        
        return {
            "solidity": round(float(solidity), 4),
            "aspectRatio": round(float(aspect_ratio), 4),
            "compactness": round(float(compactness), 4)
        }
    
    def _fourier_descriptors(self, contour, num_descriptors: int = 30) -> list:
        """
        Extract Fourier descriptors from contour signature.
        
        Returns normalized FFT magnitudes.
        """
        pts = contour[:, 0, :].astype(float)
        if len(pts) < 3:
            return [0.0] * num_descriptors
        
        # Signature: distance from centroid
        center = np.mean(pts, axis=0)
        distances = np.sqrt(np.sum((pts - center)**2, axis=1))
        
        # Normalize distances
        if np.max(distances) > 0:
            distances = distances / np.max(distances)
        else:
            return [0.0] * num_descriptors
        
        # FFT of the signature
        fft = np.fft.fft(distances)
        # Take magnitudes of first descriptors (rotation invariant)
        magnitudes = np.abs(fft[:num_descriptors])
        
        # Normalize
        if np.sum(magnitudes) > 0:
            magnitudes = magnitudes / (np.sum(magnitudes) + 1e-10)
        
        return [round(float(m), 6) for m in magnitudes]
    
    # ==================== SIMILARITY COMPUTATION ====================
    
    def chi_square_distance(self, h1, h2) -> float:
        """Chi-square distance for histogram comparison."""
        h1, h2 = np.array(h1), np.array(h2)
        eps = 1e-10
        if h1.size == 0 or h2.size == 0:
            return 1.0
        num = (h1 - h2) ** 2
        den = h1 + h2 + eps
        chi = 0.5 * np.sum(num / den)
        return float(chi)
    
    def compute_similarity(self, desc1: dict, desc2: dict, weights_config: dict = None) -> float:
        """
        Compute similarity between two sets of descriptors.
        
        Args:
            desc1: First descriptor set
            desc2: Second descriptor set
            weights_config: Optional dict with 'color', 'texture', 'shape' boolean flags
        
        Returns a similarity score between 0 and 1.
        """
        result = self.compute_similarity_detailed(desc1, desc2, weights_config)
        return result['total']
    
    def compute_similarity_detailed(self, desc1: dict, desc2: dict, weights_config: dict = None) -> dict:
        """
        Compute similarity between two sets of descriptors with detailed breakdown.
        
        Args:
            desc1: First descriptor set
            desc2: Second descriptor set
            weights_config: Optional dict with 'color', 'texture', 'shape' boolean flags
        
        Returns a dict with total similarity and individual scores.
        """
        if not desc1 or not desc2:
            return {'total': 0.0, 'color': None, 'texture': None, 'shape': None}
        
        # Default weights config - all enabled
        if weights_config is None:
            weights_config = {'color': True, 'texture': True, 'shape': True}
        
        similarities = []
        weights = []
        details = {'color': None, 'texture': None, 'shape': None}
        
        # Color similarity (weight: 0.4)
        if weights_config.get('color', True) and "color" in desc1 and "color" in desc2:
            color_sim = self._color_similarity(desc1["color"], desc2["color"])
            details['color'] = round(color_sim, 4)
            similarities.append(color_sim)
            weights.append(0.4)
        
        # Texture similarity (weight: 0.3)
        if weights_config.get('texture', True) and "texture" in desc1 and "texture" in desc2:
            texture_sim = self._texture_similarity(desc1["texture"], desc2["texture"])
            details['texture'] = round(texture_sim, 4)
            similarities.append(texture_sim)
            weights.append(0.3)
        
        # Shape similarity (weight: 0.3)
        if weights_config.get('shape', True) and "shape" in desc1 and "shape" in desc2:
            shape_sim = self._shape_similarity(desc1["shape"], desc2["shape"])
            details['shape'] = round(shape_sim, 4)
            similarities.append(shape_sim)
            weights.append(0.3)
        
        if not similarities:
            return {'total': 0.0, **details}
        
        # Weighted average
        total_weight = sum(weights)
        weighted_sim = sum(s * w for s, w in zip(similarities, weights)) / total_weight
        
        return {
            'total': round(weighted_sim, 4),
            **details
        }
    
    def _color_similarity(self, c1: dict, c2: dict) -> float:
        """Compute color descriptor similarity using multiple methods."""
        scores = []
        
        # HS Histogram similarity (chi-square distance)
        if "hsHistogram" in c1 and "hsHistogram" in c2:
            hs_dist = self.chi_square_distance(c1["hsHistogram"], c2["hsHistogram"])
            hs_score = 1 / (1 + hs_dist)
            scores.append(hs_score)
        elif "histogram" in c1 and "histogram" in c2:
            # Fallback to basic histogram
            h1 = np.array(c1["histogram"])
            h2 = np.array(c2["histogram"])
            if len(h1) == len(h2) and len(h1) > 0:
                intersection = np.sum(np.minimum(h1, h2))
                norm_factor = min(np.sum(h1), np.sum(h2))
                if norm_factor > 0:
                    scores.append(intersection / norm_factor)
        
        # Dominant colors HS similarity
        if "dominantColorsHS" in c1 and "dominantColorsHS" in c2:
            dc1, dc2 = np.array(c1["dominantColorsHS"]), np.array(c2["dominantColorsHS"])
            if len(dc1) > 0 and len(dc2) > 0:
                max_hs_distance = np.sqrt(180**2 + 256**2)
                distances = [np.min([np.linalg.norm(np.array(c1_pt) - np.array(c2_pt)) for c2_pt in dc2]) for c1_pt in dc1]
                dc_score = 1 - (np.mean(distances) / max_hs_distance)
                dc_score = np.clip(dc_score, 0, 1)
                scores.append(float(dc_score))
        
        # LAB color moments similarity
        if "labColorMoments" in c1 and "labColorMoments" in c2:
            lab1, lab2 = np.array(c1["labColorMoments"]), np.array(c2["labColorMoments"])
            if len(lab1) == len(lab2):
                lab_dist = np.linalg.norm(lab1 - lab2)
                max_lab_distance = np.sqrt(3 * (255**2 + 255**2 + 255**2))
                lab_score = 1 - (lab_dist / max_lab_distance)
                lab_score = np.clip(lab_score, 0, 1)
                scores.append(float(lab_score))
        
        return np.mean(scores) if scores else 0.0
    
    def _texture_similarity(self, t1: dict, t2: dict) -> float:
        """Compute texture descriptor similarity."""
        similarities = []
        
        # Tamura similarity
        if "tamura" in t1 and "tamura" in t2:
            tam1 = t1["tamura"]
            tam2 = t2["tamura"]
            
            coarse_sim = 1 - abs(tam1["coarseness"] - tam2["coarseness"])
            contrast_sim = 1 - abs(tam1["contrast"] - tam2["contrast"])
            dir_sim = 1 - abs(tam1["directionality"] - tam2["directionality"])
            
            similarities.append((coarse_sim + contrast_sim + dir_sim) / 3)
        
        # Gabor similarity (cosine similarity)
        if "gabor" in t1 and "gabor" in t2:
            g1 = np.array(t1["gabor"])
            g2 = np.array(t2["gabor"])
            
            if len(g1) == len(g2) and np.linalg.norm(g1) > 0 and np.linalg.norm(g2) > 0:
                gabor_sim = np.dot(g1, g2) / (np.linalg.norm(g1) * np.linalg.norm(g2))
                similarities.append((gabor_sim + 1) / 2)  # Normalize to [0, 1]
        
        return np.mean(similarities) if similarities else 0.0
    
    def _shape_similarity(self, s1: dict, s2: dict) -> float:
        """Compute shape descriptor similarity using multiple methods."""
        similarities = []
        weights = []
        
        # Hu moments similarity (from contour if available, else from image)
        hu_key = "contourHuMoments" if "contourHuMoments" in s1 and "contourHuMoments" in s2 else "huMoments"
        if hu_key in s1 and hu_key in s2:
            hu1 = np.array(s1[hu_key])
            hu2 = np.array(s2[hu_key])
            
            if len(hu1) == len(hu2):
                dist = np.sqrt(np.sum((hu1 - hu2) ** 2))
                hu_sim = np.exp(-dist / 10)
                similarities.append(hu_sim)
                weights.append(0.20)
        
        # Orientation histogram similarity (contour-based if available)
        oh_key = "orientationHistogramContour" if "orientationHistogramContour" in s1 and "orientationHistogramContour" in s2 else "orientationHistogram"
        if oh_key in s1 and oh_key in s2:
            oh1 = np.array(s1[oh_key])
            oh2 = np.array(s2[oh_key])
            
            if len(oh1) == len(oh2):
                # Chi-square distance for histograms
                oh_dist = self.chi_square_distance(oh1, oh2)
                oh_sim = 1 / (1 + oh_dist)
                similarities.append(oh_sim)
                weights.append(0.30)
        
        # Shape metrics similarity
        if "shapeMetrics" in s1 and "shapeMetrics" in s2:
            sm1 = s1["shapeMetrics"]
            sm2 = s2["shapeMetrics"]
            
            s1_arr = np.array([sm1.get("solidity", 0), sm1.get("aspectRatio", 1), sm1.get("compactness", 1)])
            s2_arr = np.array([sm2.get("solidity", 0), sm2.get("aspectRatio", 1), sm2.get("compactness", 1)])
            
            sm_dist = np.linalg.norm(s1_arr - s2_arr)
            sm_sim = np.exp(-sm_dist)
            similarities.append(sm_sim)
            weights.append(0.25)
        
        # Fourier descriptors similarity (chi-square)
        if "fourierDescriptors" in s1 and "fourierDescriptors" in s2:
            f1 = np.array(s1["fourierDescriptors"])
            f2 = np.array(s2["fourierDescriptors"])
            
            if len(f1) == len(f2):
                f_dist = self.chi_square_distance(f1, f2)
                f_sim = 1 / (1 + f_dist)
                similarities.append(f_sim)
                weights.append(0.25)
        
        if not similarities:
            return 0.0
        
        # Weighted average
        total_weight = sum(weights)
        if total_weight > 0:
            return sum(s * w for s, w in zip(similarities, weights)) / total_weight
        return np.mean(similarities)
