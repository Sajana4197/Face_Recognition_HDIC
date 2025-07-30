import os
import cv2
import dlib
import numpy as np
import pickle
from sklearn.cluster import KMeans

# Configuration
D = 20000
IMG_SIZE = 64
DATASET_PATH = 'dataset'
PROTOTYPE_PATH = 'prototypes_clustered.pkl'
N_CLUSTERS = 3  # Number of clusters/prototypes per person

# Load detector, predictor, and face cascade
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Hypervector utilities
def generate_random_hv(D=20000):
    return np.random.choice([0, 1], size=D).astype(np.uint8)

def majority_vote(hv_list):
    sum_vec = np.sum(hv_list, axis=0)
    return (sum_vec >= (len(hv_list) / 2)).astype(np.uint8)

def create_encoding_dicts(img_size, levels=256, D=20000):
    loc_dict = {(x, y): generate_random_hv(D) for x in range(img_size) for y in range(img_size)}
    int_dict = {i: generate_random_hv(D) for i in range(levels)}
    grad_dict = {i: generate_random_hv(D) for i in range(levels)}
    return loc_dict, int_dict, grad_dict

# Preprocessing functions
def preprocess_image(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def detect_face_multi_method(img):
    gray = preprocess_image(img)
    for upsample in [1, 2, 0]:
        faces = detector(gray, upsample)
        if len(faces) > 0:
            return faces[0], gray, "dlib"

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_rect = (x, y, w, h)
        return face_rect, gray, "opencv"

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    faces = detector(enhanced, 1)
    if len(faces) > 0:
        return faces[0], enhanced, "dlib_enhanced"

    return None, gray, "none"

def align_face_robust(img):
    face, gray, method = detect_face_multi_method(img)
    if face is None:
        raise ValueError("No face detected with any method.")

    try:
        if method == "opencv":
            x, y, w, h = face
            face_rect = dlib.rectangle(x, y, x + w, y + h)
            landmarks = predictor(gray, face_rect)
        else:
            landmarks = predictor(gray, face)

        left_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

        left_eye = (int(np.mean([p[0] for p in left_eye_points])), int(np.mean([p[1] for p in left_eye_points])))
        right_eye = (int(np.mean([p[0] for p in right_eye_points])), int(np.mean([p[1] for p in right_eye_points])))

        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))

        center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1)
        aligned = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]))

    except Exception as e:
        print(f"Landmark alignment failed: {e}, using simple crop")
        aligned = gray

    if method == "opencv":
        x, y, w, h = face
    else:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

    padding = int(min(w, h) * 0.1)
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(aligned.shape[1] - x, w + 2 * padding)
    h = min(aligned.shape[0] - y, h + 2 * padding)

    face_crop = aligned[y:y + h, x:x + w]

    if face_crop.shape[0] < 20 or face_crop.shape[1] < 20:
        raise ValueError("Face too small after cropping")

    face_eq = cv2.equalizeHist(face_crop)
    face_resized = cv2.resize(face_eq, (IMG_SIZE, IMG_SIZE))

    return face_resized

def compute_gradient(img):
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)

def encode_image(img, loc_dict, int_dict, grad_dict):
    gradient = compute_gradient(img)
    hv_list = []
    for x in range(IMG_SIZE):
        for y in range(IMG_SIZE):
            pos_hv = loc_dict[(x, y)]
            int_hv = int_dict[img[x, y]]
            grad_hv = grad_dict[gradient[x, y]]
            combined = np.bitwise_xor(np.bitwise_xor(pos_hv, int_hv), grad_hv)
            hv_list.append(combined)
    return majority_vote(hv_list)

def build_clustered_prototypes(dataset_path, loc_dict, int_dict, grad_dict):
    clustered_prototypes = {}

    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)
        if not os.path.isdir(person_path):
            continue

        encodings = []
        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            img = cv2.imread(img_path)
            try:
                preprocessed = align_face_robust(img)
                encoding = encode_image(preprocessed, loc_dict, int_dict, grad_dict)
                encodings.append(encoding)
            except Exception as e:
                print(f"Skipping {img_path}: {e}")

        if len(encodings) == 0:
            print(f"No valid encodings for {person}")
            continue

        encodings = np.array(encodings)

        if len(encodings) < N_CLUSTERS:
            print(f"Person {person} has fewer images than clusters; using all encodings as prototypes.")
            prototypes = list(encodings)
        else:
            print(f"Clustering encodings for {person}...")
            kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
            labels = kmeans.fit_predict(encodings)

            prototypes = []
            for cluster_label in range(N_CLUSTERS):
                cluster_encodings = encodings[labels == cluster_label]
                prototype = majority_vote(cluster_encodings)
                prototypes.append(prototype)

        clustered_prototypes[person] = prototypes
        print(f"Stored {len(prototypes)} prototypes for {person}")

    return clustered_prototypes

if __name__ == "__main__":
    loc_dict, int_dict, grad_dict = create_encoding_dicts(IMG_SIZE, 256, D)
    prototypes = build_clustered_prototypes(DATASET_PATH, loc_dict, int_dict, grad_dict)

    with open(PROTOTYPE_PATH, 'wb') as f:
        pickle.dump({
            'prototypes': prototypes,
            'loc_dict': loc_dict,
            'int_dict': int_dict,
            'grad_dict': grad_dict
        }, f)

    print("Clustered prototypes saved successfully.")
