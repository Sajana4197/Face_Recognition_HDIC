import argparse
import os
import pickle
import cv2
import numpy as np
import dlib
from skimage.feature import local_binary_pattern, hog
from scipy.spatial.distance import hamming

# Configuration
IMG_SIZE = 64
D = 30000
PROTOTYPE_PATH = "prototypes2.pkl"
LBP_POINTS = 8
LBP_RADIUS = 1
HOG_BINS = 9

# Weighting (can be tuned)
WEIGHTS = {
    "intensity": 1.0,
    "gradient": 1.0,
    "lbp": 1.0,
    "hog": 1.0
}

# Models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Encoding setup
def generate_random_hv(D):
    return np.random.choice([0, 1], size=D).astype(np.uint8)

def xor_hv(a, b):
    return np.bitwise_xor(a, b)

def majority_vote(hv_list):
    return (np.sum(hv_list, axis=0) >= (len(hv_list) / 2)).astype(np.uint8)

def create_encoding_dicts():
    loc_dict = {}
    feature_dicts = {
        "intensity": {},
        "gradient": {},
        "lbp": {},
        "hog": {}
    }

    for x in range(IMG_SIZE):
        for y in range(IMG_SIZE):
            loc_dict[(x, y)] = generate_random_hv(D)

    for i in range(256):
        feature_dicts["intensity"][i] = generate_random_hv(D)
        feature_dicts["gradient"][i] = generate_random_hv(D)
        feature_dicts["lbp"][i] = generate_random_hv(D)

    for b in range(HOG_BINS):
        feature_dicts["hog"][b] = generate_random_hv(D)

    return loc_dict, feature_dicts

# Face alignment (same as datasetPreprocessing2)
def detect_face_multi_method(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for up in [1, 2, 0]:
        faces = detector(gray, up)
        if len(faces) > 0:
            return faces[0], gray

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return dlib.rectangle(x, y, x + w, y + h), gray

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    faces = detector(enhanced, 1)
    if len(faces) > 0:
        return faces[0], enhanced

    return None, gray

def align_face_robust(img):
    face, gray = detect_face_multi_method(img)
    if face is None:
        raise ValueError("No face detected")

    try:
        landmarks = predictor(gray, face)
        left = np.mean([[landmarks.part(i).x, landmarks.part(i).y] for i in range(36, 42)], axis=0)
        right = np.mean([[landmarks.part(i).x, landmarks.part(i).y] for i in range(42, 48)], axis=0)
        dx, dy = right[0] - left[0], right[1] - left[1]
        angle = np.degrees(np.arctan2(dy, dx))
        center = tuple(np.mean([left, right], axis=0).astype(int))
        M = cv2.getRotationMatrix2D(center, angle, 1)
        aligned = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]))
    except:
        aligned = gray

    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    x, y = max(0, x - 10), max(0, y - 10)
    face_crop = aligned[y:y + h + 20, x:x + w + 20]

    if face_crop.shape[0] < 20 or face_crop.shape[1] < 20:
        raise ValueError("Face too small")

    face_eq = cv2.equalizeHist(face_crop)
    return cv2.resize(face_eq, (IMG_SIZE, IMG_SIZE))

def encode_image(img, loc_dict, feature_dicts):
    lbp = local_binary_pattern(img, LBP_POINTS, LBP_RADIUS, method='uniform').astype(np.uint8)
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad = cv2.convertScaleAbs(np.hypot(gx, gy))
    hog_feat = hog(img, orientations=HOG_BINS, pixels_per_cell=(8, 8), cells_per_block=(1, 1), feature_vector=False)
    hog_feat = hog_feat.squeeze() if hog_feat.ndim == 3 else hog_feat

    feature_hvs = {"intensity": [], "gradient": [], "lbp": [], "hog": []}

    for x in range(IMG_SIZE):
        for y in range(IMG_SIZE):
            loc = loc_dict[(x, y)]
            i = img[x, y]
            g = grad[x, y]
            l = lbp[x, y] if lbp[x, y] in feature_dicts["lbp"] else 0
            h_bin = int(np.clip(hog_feat[x // 8, y // 8].argmax(), 0, 8)) if x // 8 < hog_feat.shape[0] and y // 8 < hog_feat.shape[1] else 0

            feature_hvs["intensity"].append(xor_hv(loc, feature_dicts["intensity"][i]))
            feature_hvs["gradient"].append(xor_hv(loc, feature_dicts["gradient"][g]))
            feature_hvs["lbp"].append(xor_hv(loc, feature_dicts["lbp"][l]))
            feature_hvs["hog"].append(xor_hv(loc, feature_dicts["hog"][h_bin]))

    fused = []
    for key in feature_hvs:
        if feature_hvs[key]:
            fused.append(WEIGHTS[key] * majority_vote(feature_hvs[key]))

    return majority_vote(fused)

def tanimoto(a, b):
    intersection = np.sum(np.bitwise_and(a, b))
    union = np.sum(np.bitwise_or(a, b))
    return intersection / union if union != 0 else 0

def recognize(image_path, method):
    with open(PROTOTYPE_PATH, 'rb') as f:
        prototypes = pickle.load(f)

    loc_dict, feature_dicts = create_encoding_dicts()

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")

    try:
        query = encode_image(align_face_robust(img), loc_dict, feature_dicts)
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        return

    best_match = None
    best_score = -1 if method == "tanimoto" else float("inf")

    for person, protos in prototypes.items():
        for proto in protos:
            if method == "tanimoto":
                score = tanimoto(query, proto)
                if score > best_score:
                    best_score = score
                    best_match = person
            else:
                score = 1 - hamming(query, proto)
                if score > best_score:
                    best_score = score
                    best_match = person

    print("\n================ FACE RECOGNITION ===================")
    print(f"Input Image: {image_path}")
    print(f"Method: {method.upper()}")
    print(f"Best Match: {best_match}")
    print(f"Similarity Score: {best_score:.4f}")
    print("=====================================================")

# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument("--method", type=str, choices=["tanimoto", "hamming"], default="hamming", help="Similarity metric")
    args = parser.parse_args()
    print("Loading models...")
    recognize(args.input, args.method)
