# datasetPreprocessing2.py

import os
import cv2
import numpy as np
import pickle
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern, hog
from tqdm import tqdm
import dlib

# Configuration
IMG_SIZE = 64
D = 30000
N_CLUSTERS = 3
DATASET_PATH = "dataset"
OUTPUT_PATH = "prototypes2.pkl"

# Feature weights (adjust as needed)
FEATURE_WEIGHTS = {
    'intensity': 1.0,
    'gradient': 1.0,
    'lbp': 1.0,
    'hog': 1.0,
}

# Initialize models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def generate_random_hv():
    return np.random.choice([0, 1], size=D).astype(np.uint8)

def xor_hv(a, b):
    return np.bitwise_xor(a, b)

def majority_vote(hvs):
    sum_vec = np.sum(hvs, axis=0)
    return (sum_vec >= (len(hvs) / 2)).astype(np.uint8)

def create_encoding_dicts():
    loc_dict = {}
    feature_dicts = {key: {} for key in FEATURE_WEIGHTS}
    
    for x in range(IMG_SIZE):
        for y in range(IMG_SIZE):
            loc_dict[(x, y)] = generate_random_hv()
    
    for i in range(256):
        feature_dicts['intensity'][i] = generate_random_hv()
        feature_dicts['gradient'][i] = generate_random_hv()
        feature_dicts['lbp'][i] = generate_random_hv()
    
    for i in range(9):  # HOG bins
        feature_dicts['hog'][i] = generate_random_hv()

    return loc_dict, feature_dicts

def detect_face_multi_method(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for up in [1, 2, 0]:
        faces = detector(gray, up)
        if len(faces):
            return faces[0], gray
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    if len(faces):
        x, y, w, h = faces[0]
        return dlib.rectangle(x, y, x + w, y + h), gray
    return None, gray

def align_face(img):
    face, gray = detect_face_multi_method(img)
    if face is None:
        raise ValueError("No face found")

    try:
        lm = predictor(gray, face)
        left = np.mean([[lm.part(i).x, lm.part(i).y] for i in range(36, 42)], axis=0)
        right = np.mean([[lm.part(i).x, lm.part(i).y] for i in range(42, 48)], axis=0)
        angle = np.degrees(np.arctan2(right[1] - left[1], right[0] - left[0]))
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
    return cv2.resize(cv2.equalizeHist(face_crop), (IMG_SIZE, IMG_SIZE))

def encode_image(img, loc_dict, feature_dicts):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.clip(cv2.convertScaleAbs(np.hypot(gx, gy)), 0, 255).astype(np.uint8)
    lbp = local_binary_pattern(img, 8, 1, 'uniform').astype(np.uint8)
    hog_feat = hog(img, orientations=9, pixels_per_cell=(8,8), cells_per_block=(1,1), feature_vector=False)
    hog_feat = hog_feat.squeeze()

    channel_hvs = []

    for key in FEATURE_WEIGHTS:
        weighted_hvs = []
        for x in range(IMG_SIZE):
            for y in range(IMG_SIZE):
                pos_hv = loc_dict[(x, y)]
                if key == 'intensity':
                    code = int(img[x, y])
                elif key == 'gradient':
                    code = int(grad[x, y])
                elif key == 'lbp':
                    code = int(lbp[x, y]) if int(lbp[x, y]) in feature_dicts['lbp'] else 0
                elif key == 'hog':
                    bin_val = hog_feat[x//8, y//8].argmax() if x//8 < hog_feat.shape[0] and y//8 < hog_feat.shape[1] else 0
                    code = int(bin_val)
                feat_hv = feature_dicts[key][code]
                weighted_hvs.append(xor_hv(pos_hv, feat_hv))
        fused = majority_vote(weighted_hvs)
        channel_hvs.append(fused)

    final_hv = majority_vote(channel_hvs)
    return final_hv

def build_prototypes():
    loc_dict, feature_dicts = create_encoding_dicts()
    prototypes = {}

    for person in tqdm(os.listdir(DATASET_PATH)):
        person_path = os.path.join(DATASET_PATH, person)
        if not os.path.isdir(person_path): continue
        hvs = []
        for img_file in os.listdir(person_path):
            try:
                img = cv2.imread(os.path.join(person_path, img_file))
                if img is None: continue
                pre = align_face(img)
                hv = encode_image(pre, loc_dict, feature_dicts)
                hvs.append(hv)
            except Exception as e:
                print(f"Skipping {img_file}: {e}")
        if len(hvs) >= N_CLUSTERS:
            km = KMeans(n_clusters=N_CLUSTERS).fit(hvs)
            prototypes[person] = km.cluster_centers_.astype(np.uint8)
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(prototypes, f)
    print("✅ Prototypes saved.")

if __name__ == '__main__':
    build_prototypes()
