import os
import cv2
import dlib
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import ndimage
from skimage.feature import local_binary_pattern
import warnings
warnings.filterwarnings('ignore')

# Configuration
D = 20000
IMG_SIZE = 64
DATASET_PATH = 'dataset'
PROTOTYPE_PATH = 'prototypes_enhanced.pkl'
QUALITY_THRESHOLD = 0.6
MAX_CLUSTERS = 5

# Load detector, predictor, and face cascade
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Enhanced Hypervector utilities
def generate_bipolar_hv(D=20000):
    """Generate bipolar hypervector (-1, +1) for better separation"""
    return np.random.choice([-1, 1], size=D).astype(np.int8)

def weighted_bundle(hvs, weights=None):
    """Bundle hypervectors with weights"""
    if weights is None:
        weights = [1.0] * len(hvs)
    
    weighted_sum = np.zeros_like(hvs[0], dtype=np.float32)
    for hv, weight in zip(hvs, weights):
        weighted_sum += hv.astype(np.float32) * weight
    
    return np.sign(weighted_sum).astype(np.int8)

def create_enhanced_encoding_dicts(img_size, levels=256, D=20000):
    """Create comprehensive encoding dictionaries with better orthogonality"""
    print("Creating enhanced encoding dictionaries...")
    
    # Spatial location encoding
    loc_dict = {}
    for x in range(img_size):
        for y in range(img_size):
            loc_dict[(x, y)] = generate_bipolar_hv(D)
    
    # Intensity encoding
    int_dict = {}
    for i in range(levels):
        int_dict[i] = generate_bipolar_hv(D)
    
    # Gradient magnitude encoding
    grad_dict = {}
    for i in range(levels):
        grad_dict[i] = generate_bipolar_hv(D)
    
    # Gradient orientation encoding (8 bins for 0-360 degrees)
    orient_dict = {}
    for i in range(8):
        orient_dict[i] = generate_bipolar_hv(D)
    
    # Texture feature encoding (LBP patterns)
    texture_dict = {}
    for i in range(256):  # LBP can have values 0-255
        texture_dict[i] = generate_bipolar_hv(D)
    
    # Multi-scale encoding
    scale_dict = {}
    for i in range(4):  # 4 different scales
        scale_dict[i] = generate_bipolar_hv(D)
    
    # Gabor response encoding
    gabor_dict = {}
    for i in range(levels):
        gabor_dict[i] = generate_bipolar_hv(D)
    
    return {
        'location': loc_dict,
        'intensity': int_dict,
        'gradient': grad_dict,
        'orientation': orient_dict,
        'texture': texture_dict,
        'scale': scale_dict,
        'gabor': gabor_dict
    }

# Enhanced Feature Extraction
def extract_gabor_features(img, frequencies=[0.1, 0.15, 0.25], orientations=[0, 45, 90, 135]):
    """Extract Gabor filter responses for texture analysis"""
    gabor_responses = []
    for freq in frequencies:
        for angle in orientations:
            kernel_real, _ = cv2.getGaborKernel((15, 15), 3, np.radians(angle), 
                                               2 * np.pi * freq, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(img, cv2.CV_8UC3, kernel_real)
            gabor_responses.append(filtered)
    
    # Combine responses (take mean for simplicity)
    if gabor_responses:
        combined = np.mean(gabor_responses, axis=0).astype(np.uint8)
    else:
        combined = img
    return combined

def extract_lbp_features(img, radius=2, n_points=16):
    """Extract Local Binary Pattern features"""
    lbp = local_binary_pattern(img, n_points, radius, method='uniform')
    # Normalize to 0-255 range
    lbp_normalized = ((lbp - lbp.min()) / (lbp.max() - lbp.min()) * 255).astype(np.uint8)
    return lbp_normalized

def compute_enhanced_gradients(img):
    """Compute comprehensive gradient information"""
    # Sobel gradients
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # Magnitude and orientation
    magnitude = cv2.magnitude(grad_x, grad_y)
    orientation = cv2.phase(grad_x, grad_y, angleInDegrees=True)
    
    # Laplacian for edge detection
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    
    # Normalize all to 0-255
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    orientation = cv2.normalize(orientation, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    laplacian = cv2.normalize(np.abs(laplacian), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return magnitude, orientation, laplacian

def assess_image_quality(img):
    """Comprehensive image quality assessment"""
    # Sharpness (Laplacian variance)
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    sharpness = laplacian.var()
    
    # Contrast (standard deviation)
    contrast = img.std()
    
    # Brightness distribution (prefer balanced lighting)
    mean_brightness = img.mean()
    brightness_score = 1.0 - abs(mean_brightness - 128) / 128
    
    # Gradient strength (edge content)
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_strength = np.sqrt(grad_x**2 + grad_y**2).mean()
    
    # Normalize components
    sharpness_norm = min(1.0, sharpness / 500)
    contrast_norm = min(1.0, contrast / 60)
    gradient_norm = min(1.0, gradient_strength / 50)
    
    # Weighted combination
    quality_score = (sharpness_norm * 0.35 + 
                    contrast_norm * 0.35 + 
                    brightness_score * 0.15 + 
                    gradient_norm * 0.15)
    
    return quality_score

# Preprocessing functions (enhanced versions)
def preprocess_image(img):
    """Enhanced preprocessing with noise reduction"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply slight denoising
    denoised = cv2.bilateralFilter(gray, 5, 50, 50)
    return denoised

def detect_face_multi_method(img):
    """Multi-method face detection with enhanced preprocessing"""
    gray = preprocess_image(img)
    
    # Method 1: dlib with different upsample values
    for upsample in [1, 2, 0]:
        faces = detector(gray, upsample)
        if len(faces) > 0:
            return faces[0], gray, "dlib"

    # Method 2: OpenCV cascade
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return (x, y, w, h), gray, "opencv"

    # Method 3: Enhanced preprocessing + dlib
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    faces = detector(enhanced, 1)
    if len(faces) > 0:
        return faces[0], enhanced, "dlib_enhanced"

    return None, gray, "none"

def align_face_robust(img):
    """Enhanced face alignment with better error handling"""
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

        # Eye landmarks
        left_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

        left_eye = (int(np.mean([p[0] for p in left_eye_points])), 
                   int(np.mean([p[1] for p in left_eye_points])))
        right_eye = (int(np.mean([p[0] for p in right_eye_points])), 
                    int(np.mean([p[1] for p in right_eye_points])))

        # Calculate rotation angle
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))

        # Rotation center
        center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

        # Apply rotation
        M = cv2.getRotationMatrix2D(center, angle, 1)
        aligned = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]))

    except Exception as e:
        print(f"Landmark alignment failed: {e}, using original")
        aligned = gray

    # Extract face region
    if method == "opencv":
        x, y, w, h = face
    else:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

    # Add padding
    padding = int(min(w, h) * 0.15)
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(aligned.shape[1] - x, w + 2 * padding)
    h = min(aligned.shape[0] - y, h + 2 * padding)

    face_crop = aligned[y:y + h, x:x + w]

    if face_crop.shape[0] < 20 or face_crop.shape[1] < 20:
        raise ValueError("Face too small after cropping")

    # Enhanced preprocessing
    face_eq = cv2.equalizeHist(face_crop)
    face_resized = cv2.resize(face_eq, (IMG_SIZE, IMG_SIZE))

    return face_resized

def encode_image_enhanced(img, encoding_dicts):
    """Enhanced image encoding with multiple features and spatial weighting"""
    # Extract all features
    grad_magnitude, grad_orientation, laplacian = compute_enhanced_gradients(img)
    lbp_features = extract_lbp_features(img)
    gabor_features = extract_gabor_features(img)
    
    # Convert orientation to 8-bin representation (0-7)
    grad_orient_binned = (grad_orientation * 7 // 255).astype(np.uint8)
    
    # Create spatial importance weights (center more important)
    center_x, center_y = img.shape[0] // 2, img.shape[1] // 2
    spatial_weights = np.zeros(img.shape)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            spatial_weights[x, y] = 1.0 - (dist / max_dist) * 0.4  # Center: 1.0, edges: 0.6
    
    hv_list = []
    weights = []
    
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            # Get base vectors
            pos_hv = encoding_dicts['location'][(x, y)]
            
            # Feature vectors
            features_to_bundle = []
            feature_weights = []
            
            # Original intensity (normalized importance)
            features_to_bundle.append(encoding_dicts['intensity'][img[x, y]])
            feature_weights.append(0.25)
            
            # Gradient magnitude
            features_to_bundle.append(encoding_dicts['gradient'][grad_magnitude[x, y]])
            feature_weights.append(0.20)
            
            # Gradient orientation
            features_to_bundle.append(encoding_dicts['orientation'][grad_orient_binned[x, y]])
            feature_weights.append(0.20)
            
            # Texture (LBP)
            lbp_val = min(255, max(0, lbp_features[x, y]))  # Ensure valid range
            features_to_bundle.append(encoding_dicts['texture'][lbp_val])
            feature_weights.append(0.20)
            
            # Gabor response
            features_to_bundle.append(encoding_dicts['gabor'][gabor_features[x, y]])
            feature_weights.append(0.15)
            
            # Bundle features
            feature_bundle = weighted_bundle(features_to_bundle, feature_weights)
            
            # Bind with spatial location
            combined = pos_hv * feature_bundle  # Element-wise multiplication
            
            hv_list.append(combined)
            weights.append(spatial_weights[x, y])
    
    # Final weighted bundling
    final_hv = weighted_bundle(hv_list, weights)
    return final_hv

def adaptive_clustering(encodings, max_clusters=MAX_CLUSTERS, min_clusters=2):
    """Adaptively determine optimal number of clusters using silhouette analysis"""
    if len(encodings) < min_clusters:
        return len(encodings), list(range(len(encodings)))
    
    max_clusters = min(max_clusters, len(encodings))
    best_score = -1
    best_n_clusters = min_clusters
    best_labels = None
    
    for n_clusters in range(min_clusters, max_clusters + 1):
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(encodings)
            score = silhouette_score(encodings, labels)
            
            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters
                best_labels = labels
                
        except Exception as e:
            print(f"Clustering with {n_clusters} clusters failed: {e}")
            continue
    
    return best_n_clusters, best_labels

def build_enhanced_prototypes(dataset_path, encoding_dicts):
    """Build enhanced prototypes with quality assessment and adaptive clustering"""
    enhanced_prototypes = {}
    
    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)
        if not os.path.isdir(person_path):
            continue
        
        print(f"Processing {person}...")
        encodings = []
        quality_scores = []
        
        for img_file in os.listdir(person_path):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
                
            img_path = os.path.join(person_path, img_file)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Could not read {img_path}")
                continue
            
            try:
                # Preprocess and assess quality
                preprocessed = align_face_robust(img)
                quality_score = assess_image_quality(preprocessed)
                
                if quality_score >= QUALITY_THRESHOLD:
                    # Encode image
                    encoding = encode_image_enhanced(preprocessed, encoding_dicts)
                    encodings.append(encoding)
                    quality_scores.append(quality_score)
                    print(f"  ✓ {img_file} (quality: {quality_score:.3f})")
                else:
                    print(f"  ✗ {img_file} (low quality: {quality_score:.3f})")
                    
            except Exception as e:
                print(f"  ✗ {img_file}: {e}")
        
        if len(encodings) == 0:
            print(f"No valid encodings for {person}")
            continue
        
        encodings = np.array(encodings)
        quality_scores = np.array(quality_scores)
        
        # Generate prototypes
        if len(encodings) <= 3:
            # Use all encodings for small datasets
            prototypes = list(encodings)
            print(f"  Using all {len(prototypes)} encodings as prototypes")
        else:
            # Adaptive clustering
            n_clusters, labels = adaptive_clustering(encodings)
            
            prototypes = []
            for cluster_id in range(n_clusters):
                cluster_mask = (labels == cluster_id)
                cluster_encodings = encodings[cluster_mask]
                cluster_qualities = quality_scores[cluster_mask]
                
                # Create quality-weighted prototype
                prototype = weighted_bundle(list(cluster_encodings), list(cluster_qualities))
                prototypes.append(prototype)
            
            print(f"  Generated {len(prototypes)} prototypes from {n_clusters} clusters")
        
        enhanced_prototypes[person] = prototypes
    
    return enhanced_prototypes

if __name__ == "__main__":
    print("Starting enhanced dataset preprocessing...")
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Hypervector dimension: {D}")
    print(f"Quality threshold: {QUALITY_THRESHOLD}")
    
    # Create encoding dictionaries
    encoding_dicts = create_enhanced_encoding_dicts(IMG_SIZE, 256, D)
    print("Encoding dictionaries created.")
    
    # Build enhanced prototypes
    prototypes = build_enhanced_prototypes(DATASET_PATH, encoding_dicts)
    
    # Save results
    print(f"Saving to {PROTOTYPE_PATH}...")
    with open(PROTOTYPE_PATH, 'wb') as f:
        pickle.dump({
            'prototypes': prototypes,
            'encoding_dicts': encoding_dicts,
            'config': {
                'IMG_SIZE': IMG_SIZE,
                'D': D,
                'QUALITY_THRESHOLD': QUALITY_THRESHOLD
            }
        }, f)
    
    print("Enhanced prototypes saved successfully!")
    print(f"Total persons processed: {len(prototypes)}")
    for person, protos in prototypes.items():
        print(f"  {person}: {len(protos)} prototypes")