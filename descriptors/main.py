from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from sklearn.cluster import KMeans
from scipy.stats import skew
from io import BytesIO
from PIL import Image
from scipy.spatial.distance import cosine, euclidean

app = FastAPI(title="Image Descriptors API")

# ------------------------ Prétraitement ------------------------ #
def preprocess_image(image: np.ndarray, size=(128, 128), gamma=2.2):
    image = cv2.resize(image, size)
    image = cv2.GaussianBlur(image, (3, 3), 0.5)
    image = np.power(image / 255.0, 1/gamma)
    image = np.uint8(image * 255)
    return image



# ------------------------ Histogramme HS ------------------------ #
def hs_histogram(image: np.ndarray, bins=(30, 32)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, _ = cv2.split(hsv)
    hist = cv2.calcHist([h, s], [0, 1], None, bins, [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.tolist()

# ------------------------ Couleurs dominantes ------------------------ #
def dominant_colors(image: np.ndarray, k=12):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 2)
    kmeans = KMeans(n_clusters=k, random_state=42).fit(pixels)
    centers = kmeans.cluster_centers_
    return centers.tolist()

# ------------------------ Color Moments LAB ------------------------ #
def color_moments_lab(image: np.ndarray):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    moments = []
    for i in range(3):
        channel = lab[:, :, i]
        mean_val = np.mean(channel)
        var_val = np.var(channel)
        skew_val = skew(channel.flatten())
        moments.extend([float(mean_val), float(var_val), float(skew_val)])
    return moments

# ------------------------ Calcul des descripteurs ------------------------ #
def extract_descriptors_from_array(image: np.ndarray):
    preprocessed = preprocess_image(image)
    hs_hist = hs_histogram(preprocessed)
    dom_colors = dominant_colors(preprocessed)
    lab_moments = color_moments_lab(preprocessed)
    return {
        "hs_histogram": np.array(hs_hist),
        "dominant_colors": np.array(dom_colors),
        "lab_color_moments": np.array(lab_moments)
    }

# ------------------------ Similarité entre deux images ------------------------ #
def compute_similarity(desc1, desc2):
    # Histogramme HS → chi2 distance
    def chi2_distance(h1, h2):
        h1, h2 = np.array(h1), np.array(h2)
        eps = 1e-10
        return 0.5 * np.sum(((h1 - h2)**2) / (h1 + h2 + eps))
    
    hs_dist = chi2_distance(desc1["hs_histogram"], desc2["hs_histogram"])
    hs_score = 1 / (1 + hs_dist)  # 0 à 1

    # Dominant colors
    dc1, dc2 = desc1["dominant_colors"], desc2["dominant_colors"]
    max_hs_distance = np.sqrt(180**2 + 256**2)
    distances = [np.min([euclidean(c1, c2) for c2 in dc2]) for c1 in dc1]
    dc_score = 1 - (np.mean(distances) / max_hs_distance)
    dc_score = np.clip(dc_score, 0, 1)

    # LAB moments
    lab_dist = euclidean(desc1["lab_color_moments"], desc2["lab_color_moments"])
    max_lab_distance = np.sqrt(3 * (255**2 + 255**2 + 255**2))
    lab_score = 1 - (lab_dist / max_lab_distance)
    lab_score = np.clip(lab_score, 0, 1)

    # Score global
    overall_score = (hs_score + dc_score + lab_score) / 3
    return float(np.clip(overall_score, 0, 1))




# ------------------------ API Endpoint : extraire descripteurs ------------------------ #
@app.post("/extract_descriptors/")
async def extract_descriptors(file: UploadFile = File(...)):
    image_bytes = await file.read()
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    desc = extract_descriptors_from_array(image)
    # Convertir numpy arrays en listes pour JSON
    return JSONResponse(content={
        "filename": file.filename,
        "hs_histogram": desc["hs_histogram"].tolist(),
        "dominant_colors": desc["dominant_colors"].tolist(),
        "lab_color_moments": desc["lab_color_moments"].tolist()
    })

# ------------------------ API Endpoint : comparer deux images ------------------------ #
@app.post("/compare_images/")
async def compare_images(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    # Charger images
    image1 = Image.open(BytesIO(await file1.read())).convert("RGB")
    image2 = Image.open(BytesIO(await file2.read())).convert("RGB")
    img1 = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2BGR)
    img2 = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2BGR)
    
    # Extraire descripteurs
    desc1 = extract_descriptors_from_array(img1)
    desc2 = extract_descriptors_from_array(img2)
    
    # Calcul similarité
    score = compute_similarity(desc1, desc2)
    
    return JSONResponse(content={
        "file1": file1.filename,
        "file2": file2.filename,
        "similarity_score": score
    })
