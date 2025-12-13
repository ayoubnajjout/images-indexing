import os
import json
import argparse
import numpy as np
import cv2

FOLDER = "Formes"


def load_features(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return {
        "orientation_hist": np.array(data.get("orientation_hist", []), dtype=float),
        "hu_moments": np.array(data.get("hu_moments", []), dtype=float),
        "shape": data.get("shape", {"solidity": 0.0, "aspect_ratio": 1.0, "compactness": 1.0}),
        "fourier": np.array(data.get("fourier", []), dtype=float),
    }


def chi_square_distance(h1, h2):
    """Chi-square distance for histogram comparison"""
    eps = 1e-10
    if h1.size == 0 or h2.size == 0:
        return 1.0
    num = (h1 - h2) ** 2
    den = h1 + h2 + eps
    chi = 0.5 * np.sum(num / den)
    return float(chi)


def euclidean_distance(a, b):
    """Euclidean distance"""
    if a.size == 0 or b.size == 0:
        return 1.0
    return float(np.linalg.norm(a - b))


def simple_distance(hu1, h1, shape1, fourier1, hu2, h2, shape2, fourier2):
    """Combined distance: chi-square(hist) + euclidean(hu) + euclidean(shape) + euclidean(fourier)"""
    d_h = chi_square_distance(h1, h2)
    d_hu = euclidean_distance(hu1, hu2)
    
    # Shape distance
    s1 = np.array([shape1["solidity"], shape1["aspect_ratio"], shape1["compactness"]], dtype=float)
    s2 = np.array([shape2["solidity"], shape2["aspect_ratio"], shape2["compactness"]], dtype=float)
    d_shape = euclidean_distance(s1, s2)
    
    # Fourier distance
    d_fourier = chi_square_distance(fourier1, fourier2)
    
    # Poids: 35% hist, 25% hu, 20% shape, 20% fourier
    return 0.30 * d_h + 0.20 * d_hu + 0.25 * d_shape + 0.25 * d_fourier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query", required=True, help="Nom de l'image GIF")
    args = parser.parse_args()

    query_img = args.query
    if query_img.lower().endswith('.json'):
        query_json = os.path.join(FOLDER, os.path.basename(query_img))
        query_img_name = os.path.basename(query_img).replace('.json', '')
    else:
        query_json = os.path.join(FOLDER, query_img.replace('.gif', '.json'))
        query_img_name = query_img.replace('.gif', '')

    if not os.path.exists(query_json):
        print("❌ Fichier JSON introuvable :", query_json)
        return

    # Load all features
    files = [f for f in os.listdir(FOLDER) if f.endswith('.json')]
    features = {}
    for f in files:
        try:
            features[f.replace('.json','')] = load_features(os.path.join(FOLDER, f))
        except Exception:
            pass

    query_name = os.path.basename(query_json).replace('.json','')
    if query_name not in features:
        print("❌ Fichier JSON introuvable :", query_json)
        return

    qfeat = features[query_name]

    query_img_path = os.path.join(FOLDER, query_img_name + ".gif")
    query_img_cv = cv2.imread(query_img_path)
    if query_img_cv is None:
        for ext in ('.png', '.jpg', '.jpeg'):
            p = os.path.join(FOLDER, query_img_name + ext)
            if os.path.exists(p):
                query_img_path = p
                query_img_cv = cv2.imread(p)
                break
    if query_img_cv is None:
        print("Image requête introuvable :", query_img_path)
        return

    results = []
    for name, feat in features.items():
        if name == query_name:
            continue
        h2 = np.array(feat['orientation_hist'], dtype=float)
        hu2 = np.array(feat['hu_moments'], dtype=float)
        shape2 = feat['shape']
        fourier2 = np.array(feat['fourier'], dtype=float)
        d = simple_distance(qfeat['hu_moments'], qfeat['orientation_hist'], qfeat['shape'], qfeat['fourier'], hu2, h2, shape2, fourier2)
        results.append((name, d))

    results.sort(key=lambda x: x[1])

    print("\n🔍 Top 6 images les plus similaires (Formes):\n")
    top_images = [(query_img_name, 0.0, query_img_cv)]
    for name, d in results[:6]:
        print(f"  ➤ {name}   (distance = {d:.4f})")
        for ext in ('.gif', '.png', '.jpg', '.jpeg'):
            img_path = os.path.join(FOLDER, name + ext)
            if os.path.exists(img_path):
                img_cv = cv2.imread(img_path)
                break
        else:
            print("Image introuvable :", name)
            continue
        img_cv = cv2.resize(img_cv, (query_img_cv.shape[1], query_img_cv.shape[0]))
        top_images.append((name, d, img_cv))

    grid_rows, grid_cols = 2, 3
    img_h, img_w = query_img_cv.shape[0], query_img_cv.shape[1]
    grid_img = np.ones((grid_rows*img_h, grid_cols*img_w, 3), dtype=np.uint8) * 255

    for idx, (name, d, img_cv) in enumerate(top_images[:grid_rows*grid_cols]):
        row = idx // grid_cols
        col = idx % grid_cols
        y, x = row*img_h, col*img_w
        grid_img[y:y+img_h, x:x+img_w] = img_cv
        label = f"{name} ({d:.3f})" if d > 0 else "requete"
        cv2.putText(grid_img, label, (x+5, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    cv2.imshow("Grille des images similaires (Formes)", grid_img)
    out_name = f"test_{query_img_name}.png"
    cv2.imwrite(out_name, grid_img)
    print(f"Image enregistrée sous : {out_name}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
