import cv2
import dlib
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
D = 10000
IMG_SIZE = 64
PROTOTYPE_PATH = 'prototypes.pkl'

detector = dlib.get_frontal_face_detector()

# Preprocessing: face detection, cropping, equalization, resizing
def preprocess_image(img, show=False, title=''):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        raise ValueError("No face detected in the image.")

    face = faces[0]
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    face_crop = gray[y:y+h, x:x+w]
    face_eq = cv2.equalizeHist(face_crop)
    face_resized = cv2.resize(face_eq, (IMG_SIZE, IMG_SIZE))

    if show:
        cv2.imshow(f'Preprocessed {title}', face_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return face_resized

# Hypervector utilities
def generate_random_hv(D=10000):
    return np.random.choice([0, 1], size=D).astype(np.uint8)

def xor_hv(hv1, hv2):
    return np.bitwise_xor(hv1, hv2)

def majority_vote(hv_list):
    sum_vec = np.sum(hv_list, axis=0)
    return (sum_vec >= (len(hv_list) / 2)).astype(np.uint8)

def create_encoding_dicts(img_size, intensity_levels=256, D=10000):
    loc_dict = {}
    int_dict = {}
    for x in range(img_size):
        for y in range(img_size):
            loc_dict[(x, y)] = generate_random_hv(D)
    for i in range(intensity_levels):
        int_dict[i] = generate_random_hv(D)
    return loc_dict, int_dict

def encode_image(img, loc_dict, int_dict):
    hv_list = []
    for x in range(IMG_SIZE):
        for y in range(IMG_SIZE):
            intensity = int(img[x, y])
            pixel_hv = xor_hv(loc_dict[(x, y)], int_dict[intensity])
            hv_list.append(pixel_hv)
    return majority_vote(hv_list)

def load_prototypes(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# Compare input image to prototypes
def identify_person(input_image_path, prototypes, loc_dict, int_dict):
    img = cv2.imread(input_image_path)
    preprocessed = preprocess_image(img, show=True, title='Input Image')
    input_hv = encode_image(preprocessed, loc_dict, int_dict)

    best_match = None
    highest_similarity = -1

    for person, proto_hv in prototypes.items():
        sim = cosine_similarity([input_hv], [proto_hv])[0][0]
        print(f"Similarity with {person}: {sim:.4f}")

        if sim > highest_similarity:
            highest_similarity = sim
            best_match = person

    return best_match, highest_similarity

# Main execution
if __name__ == "__main__":
    loc_dict, int_dict = create_encoding_dicts(IMG_SIZE, 256, D)
    prototypes = load_prototypes(PROTOTYPE_PATH)

    input_image_path = "E:\\Accedemic\\FYP\\GitHub\\AppleNeuralHashAlgorithm\\images\\person_im1.png"
    person, similarity = identify_person(input_image_path, prototypes, loc_dict, int_dict)

    if person:
        print(f"\nInput image matches with: {person} (Similarity: {similarity:.4f})")
    else:
        print("No matching person found.")
