import os
import cv2
import dlib
import numpy as np
import pickle

# Configuration
D = 20000
IMG_SIZE = 64
DATASET_PATH = 'dataset'
PROTOTYPE_PATH = 'prototypes.pkl'

detector = dlib.get_frontal_face_detector()
cnn_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

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

def align_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cnn_detector(gray, 1)  # 1 upsampling is usually enough

    if len(faces) == 0:
        raise ValueError("No face detected.")

    # cnn_detector returns rectangles in 'rect'
    face_rect = faces[0].rect
    landmarks = predictor(gray, face_rect)

    left_eye = (int(landmarks.part(36).x), int(landmarks.part(36).y))
    right_eye = (int(landmarks.part(45).x), int(landmarks.part(45).y))

    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    aligned = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]))

    x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
    face_crop = aligned[y:y+h, x:x+w]
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

def build_prototypes(dataset_path, loc_dict, int_dict, grad_dict):
    prototypes = {}
    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)
        if not os.path.isdir(person_path):
            continue
        encodings = []
        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            img = cv2.imread(img_path)
            try:
                preprocessed = align_face(img)
                encoding = encode_image(preprocessed, loc_dict, int_dict, grad_dict)
                encodings.append(encoding)
            except Exception as e:
                print(f"Skipping {img_path}: {e}")
        if encodings:
            prototype = majority_vote(encodings)
            prototypes[person] = prototype
            print(f"Built prototype for {person}")
    return prototypes

if __name__ == "__main__":
    loc_dict, int_dict, grad_dict = create_encoding_dicts(IMG_SIZE, 256, D)
    prototypes = build_prototypes(DATASET_PATH, loc_dict, int_dict, grad_dict)
    with open(PROTOTYPE_PATH, 'wb') as f:
        pickle.dump({'prototypes': prototypes, 'loc_dict': loc_dict, 'int_dict': int_dict, 'grad_dict': grad_dict}, f)
    print("Prototypes saved successfully.")