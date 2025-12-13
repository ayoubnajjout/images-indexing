import cv2
import numpy as np
import json
import os

INPUT_DIR = "Formes"
OUTPUT_DIR = "Formes"


def orientation_histogram(contour, bins=16):
    pts = contour[:, 0, :].astype(float)
    if len(pts) < 2:
        return [0.0] * bins

    # rééchantillonner pour lissage
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
        return (hist * 0).tolist()
    hist = hist.astype(float) / s
    return hist.tolist()


def shape_descriptors(contour):
    """Calcule des descripteurs simples de forme: solidity, aspect_ratio, perimeter/area"""
    area = cv2.contourArea(contour)
    if area == 0:
        return {"solidity": 0.0, "aspect_ratio": 1.0, "compactness": 1.0}
    
    # Solidity: area / convex hull area
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0
    
    # Aspect ratio: largeur / hauteur du bounding rect
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / float(h) if h > 0 else 1.0
    
    # Compactness: 4π*area / perimeter²
    perimeter = cv2.arcLength(contour, True)
    compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 1.0
    
    return {"solidity": float(solidity), "aspect_ratio": float(aspect_ratio), "compactness": float(compactness)}


def hu_moments(contour):
    """Calculer les moments de Hu à partir d'une image masque du contour"""
    mask = np.zeros((np.max(contour[:, 0, 1]) + 1, np.max(contour[:, 0, 0]) + 1), dtype=np.uint8)
    # décaler le contour pour qu'il commence à (0,0)
    pts = contour[:, 0, :]
    min_x, min_y = np.min(pts[:, 0]), np.min(pts[:, 1])
    shifted = (pts - np.array([min_x, min_y])).astype(np.int32)
    cv2.drawContours(mask, [shifted], -1, 255, thickness=-1)
    moments = cv2.moments(mask)
    hu = cv2.HuMoments(moments).flatten()
    # Stabiliser avec log transform (garde signe)
    eps = 1e-12
    hu_signed_log = []
    for h in hu:
        val = float(h)
        if abs(val) < eps:
            hu_signed_log.append(0.0)
        else:
            hu_signed_log.append(-np.sign(val) * np.log10(abs(val) + eps))
    return hu_signed_log


def fourier_descriptors(contour, num_descriptors=32):
    """Extracteur de descripteurs de Fourier basés sur la signature du contour"""
    pts = contour[:, 0, :].astype(float)
    if len(pts) < 3:
        return [0.0] * num_descriptors
    
    # Signature: distance du centroïde
    center = np.mean(pts, axis=0)
    distances = np.sqrt(np.sum((pts - center)**2, axis=1))
    
    # Normaliser les distances
    if np.max(distances) > 0:
        distances = distances / np.max(distances)
    else:
        return [0.0] * num_descriptors
    
    # FFT de la signature
    fft = np.fft.fft(distances)
    # Prendre les magnitudes des premiers descripteurs (invariants de rotation)
    magnitudes = np.abs(fft[:num_descriptors])
    
    # Normaliser
    if np.sum(magnitudes) > 0:
        magnitudes = magnitudes / (np.sum(magnitudes) + 1e-10)
    
    return magnitudes.tolist()


def process_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    # Si image RGBA (transparent), convertir en gris à partir du canal alpha
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        gray = alpha
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # seuillage Otsu
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Trouver contours externes
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not cnts:
        return None

    # Contour le plus significatif: celui de plus grande aire
    contour = max(cnts, key=cv2.contourArea)

    oh = orientation_histogram(contour, bins=16)
    hu = hu_moments(contour)
    shape = shape_descriptors(contour)
    fourier = fourier_descriptors(contour, num_descriptors=30)

    return {"orientation_hist": oh, "hu_moments": hu, "shape": shape, "fourier": fourier}


def main():
    for file in os.listdir(INPUT_DIR):
        if not file.lower().endswith(".gif"):
            continue

        path = os.path.join(INPUT_DIR, file)
        features = process_image(path)

        if features is None:
            print(f"❌ {file} ignorée")
            continue

        json_path = os.path.join(OUTPUT_DIR, file.replace(".gif", ".json"))
        with open(json_path, "w") as f:
            json.dump(features, f, indent=4)

        print(f"✅ {file} → Hu + orientation_hist sauvegardés")


if __name__ == "__main__":
    main()
