import numpy as np
import cv2
import dlib
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
D = 10000
IMG_SIZE = 256

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download required

# Utility functions remain the same
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

# Preprocessing: face detection, cropping, equalization, and resizing
def preprocess_image(img, show=False, title=''):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        raise ValueError("No face detected in the image.")

    # For simplicity, use the first detected face
    face = faces[0]
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    face_crop = gray[y:y+h, x:x+w]

    # Contrast normalization
    face_eq = cv2.equalizeHist(face_crop)

    # Resize to standard size
    face_resized = cv2.resize(face_eq, (IMG_SIZE, IMG_SIZE))

    if show:
        cv2.imshow(f'Preprocessed {title}', face_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return face_resized

# Encode the preprocessed image
def encode_image(img, loc_dict, int_dict):
    hv_list = []
    for x in range(IMG_SIZE):
        for y in range(IMG_SIZE):
            intensity = int(img[x, y])
            pixel_hv = xor_hv(loc_dict[(x, y)], int_dict[intensity])
            hv_list.append(pixel_hv)
    return majority_vote(hv_list)

# Image comparison function
def compare_images(image_path1, image_path2, loc_dict, int_dict):
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)

    pre1 = preprocess_image(img1, show=True, title='Image 1')
    pre2 = preprocess_image(img2, show=True, title='Image 2')

    hv1 = encode_image(pre1, loc_dict, int_dict)
    hv2 = encode_image(pre2, loc_dict, int_dict)

    similarity = cosine_similarity([hv1], [hv2])[0][0]
    return similarity

# Initialize
loc_dict, int_dict = create_encoding_dicts(IMG_SIZE, 256, D)

# Example usage
image_path1 = "E:\\Accedemic\\FYP\\GitHub\\AppleNeuralHashAlgorithm\\images\\person_im1.png"
image_path2 = "E:\\Accedemic\\FYP\\GitHub\\AppleNeuralHashAlgorithm\\images\\person_im2.png"

similarity_score = compare_images(image_path1, image_path2, loc_dict, int_dict)
print(f'Similarity between images: {similarity_score:.4f}')
