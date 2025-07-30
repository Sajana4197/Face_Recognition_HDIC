import os
import cv2
import dlib
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Configuration
D = 20000
IMG_SIZE = 64
DATASET_PATH = 'dataset'
PROTOTYPE_PATH = 'prototypes.pkl'

# Initialize detectors with fallback options
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Alternative detector (OpenCV's Haar cascade as fallback)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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

def preprocess_image(img):
    """Preprocess image to improve face detection"""
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Enhance contrast and reduce noise
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    return gray

def detect_face_multi_method(img):
    """Try multiple face detection methods"""
    gray = preprocess_image(img)
    
    # Method 1: Dlib with different upsample levels
    for upsample in [1, 2, 0]:  # Try different upsampling levels
        faces = detector(gray, upsample)
        if len(faces) > 0:
            return faces[0], gray, "dlib"
    
    # Method 2: OpenCV Haar cascade (fallback)
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) > 0:
        # Convert to dlib rectangle format
        x, y, w, h = faces[0]
        face_rect = dlib.rectangle(x, y, x+w, y+h)
        return face_rect, gray, "opencv"
    
    # Method 3: Try with different image enhancements
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    faces = detector(enhanced, 1)
    if len(faces) > 0:
        return faces[0], enhanced, "dlib_enhanced"
    
    return None, gray, "none"

def align_face_robust(img):
    """Robust face alignment with multiple fallback strategies"""
    face, gray, method = detect_face_multi_method(img)
    
    if face is None:
        raise ValueError("No face detected with any method.")
    
    # Try to get landmarks for alignment
    try:
        landmarks = predictor(gray, face)
        
        # Check if we have valid eye landmarks
        left_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
        
        # Calculate eye centers
        left_eye = (int(np.mean([p[0] for p in left_eye_points])), 
                   int(np.mean([p[1] for p in left_eye_points])))
        right_eye = (int(np.mean([p[0] for p in right_eye_points])), 
                    int(np.mean([p[1] for p in right_eye_points])))
        
        # Calculate rotation angle
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Calculate center point
        center_x = (left_eye[0] + right_eye[0]) // 2
        center_y = (left_eye[1] + right_eye[1]) // 2
        center = (center_x, center_y)
        
        # Apply rotation
        M = cv2.getRotationMatrix2D(center, angle, 1)
        aligned = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]))
        
    except Exception as e:
        print(f"Landmark alignment failed: {e}, using simple crop")
        aligned = gray
    
    # Extract face region
    if method == "opencv":
        # For OpenCV detection, face is already (x, y, w, h)
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
    else:
        # For dlib detection
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
    
    # Add some padding and ensure we don't go out of bounds
    padding = int(min(w, h) * 0.1)
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(aligned.shape[1] - x, w + 2*padding)
    h = min(aligned.shape[0] - y, h + 2*padding)
    
    # Crop face
    face_crop = aligned[y:y+h, x:x+w]
    
    # Ensure minimum size
    if face_crop.shape[0] < 20 or face_crop.shape[1] < 20:
        raise ValueError("Face too small after cropping")
    
    # Apply histogram equalization
    face_eq = cv2.equalizeHist(face_crop)
    
    # Resize to target size
    face_resized = cv2.resize(face_eq, (IMG_SIZE, IMG_SIZE))
    
    return face_resized

def compute_gradient(img):
    """Compute gradient magnitude"""
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)

def encode_image(img, loc_dict, int_dict, grad_dict):
    """Encode image using hyperdimensional vectors"""
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
    """Build prototypes with improved error handling"""
    prototypes = {}
    total_images = 0
    processed_images = 0
    
    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)
        if not os.path.isdir(person_path):
            continue
            
        print(f"\nProcessing {person}...")
        encodings = []
        person_processed = 0
        person_total = 0
        
        for img_file in os.listdir(person_path):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                continue
                
            img_path = os.path.join(person_path, img_file)
            person_total += 1
            total_images += 1
            
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"  Could not load {img_file}")
                    continue
                    
                preprocessed = align_face_robust(img)
                encoding = encode_image(preprocessed, loc_dict, int_dict, grad_dict)
                encodings.append(encoding)
                person_processed += 1
                processed_images += 1
                
                if person_processed % 50 == 0:
                    print(f"  Processed {person_processed}/{person_total} images")
                    
            except Exception as e:
                print(f"  Skipping {img_file}: {e}")
                
        if encodings:
            prototype = majority_vote(encodings)
            prototypes[person] = prototype
            print(f"  Built prototype for {person} from {len(encodings)} images")
        else:
            print(f"  WARNING: No valid images found for {person}")
    
    print(f"\nSummary: Processed {processed_images}/{total_images} images")
    return prototypes

def validate_setup():
    """Validate that required files exist"""
    if not os.path.exists('shape_predictor_68_face_landmarks.dat'):
        print("WARNING: shape_predictor_68_face_landmarks.dat not found!")
        print("Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        return False
    
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset path {DATASET_PATH} not found!")
        return False
        
    return True

if __name__ == "__main__":
    if not validate_setup():
        exit(1)
        
    print("Creating encoding dictionaries...")
    loc_dict, int_dict, grad_dict = create_encoding_dicts(IMG_SIZE, 256, D)
    
    print("Building prototypes...")
    prototypes = build_prototypes(DATASET_PATH, loc_dict, int_dict, grad_dict)
    
    if prototypes:
        with open(PROTOTYPE_PATH, 'wb') as f:
            pickle.dump({
                'prototypes': prototypes, 
                'loc_dict': loc_dict, 
                'int_dict': int_dict, 
                'grad_dict': grad_dict
            }, f)
        print(f"\nPrototypes saved successfully for {len(prototypes)} people.")
    else:
        print("ERROR: No prototypes were built!")