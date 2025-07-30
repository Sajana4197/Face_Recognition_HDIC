import cv2
import dlib
import numpy as np
import pickle
import argparse
from datetime import datetime

# Configuration
IMG_SIZE = 64
PROTOTYPE_PATH = 'prototypes_clustered.pkl'

# Load detector, predictor, and face cascade
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Import shared preprocessing and encoding functions from datasetPreprocessing
from datasetPreprocessing2 import align_face_robust, encode_image

def hamming_distance(hv1, hv2):
    return np.sum(hv1 != hv2)

def tanimoto_similarity(hv1, hv2):
    intersection = np.sum(hv1 & hv2)
    union = np.sum(hv1) + np.sum(hv2) - intersection
    return intersection / union if union != 0 else 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face Recognition using HDIC')
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument('--method', type=str, default='hamming', choices=['hamming', 'tanimoto'], help='Similarity method')
    parser.add_argument('--threshold', type=float, default=0.65, help='Recognition threshold')
    args = parser.parse_args()

    with open(PROTOTYPE_PATH, 'rb') as f:
        data = pickle.load(f)

    prototypes = data['prototypes']
    loc_dict = data['loc_dict']
    int_dict = data['int_dict']
    grad_dict = data['grad_dict']

    img = cv2.imread(args.input)
    preprocessed = align_face_robust(img)
    input_hv = encode_image(preprocessed, loc_dict, int_dict, grad_dict)

    best_person = None
    if args.method == 'tanimoto':
        best_score = -1
    else:
        best_score = 0


    for person, proto_list in prototypes.items():
        for proto in proto_list:
            if args.method == 'tanimoto':
                score = tanimoto_similarity(input_hv, proto)
                if score > best_score:
                    best_score = score
                    best_person = person
            else:
                distance = hamming_distance(input_hv, proto)
                similarity = 1 - (distance / len(input_hv))
                if similarity > best_score:
                    best_score = similarity
                    best_person = person

    print("="*60)
    print("FACE RECOGNITION RESULTS")
    print("="*60)
    print(f"Input Image: {args.input}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Similarity Method: {args.method.upper()}")
    print(f"Threshold: {args.threshold:.3f}")
    print("-"*60)
    if best_score >= args.threshold:
        print(f"Best Match: {best_person}")
        print(f"Similarity Score: {best_score:.4f}")
    else:
        print(f"Best Match: {best_person}")
        print(f"Similarity Score: {best_score:.4f}")
        print("\u274C NO MATCH")
        print("Person not found in database")
    print("="*60)
