import os
import cv2
import dlib
import numpy as np
import pickle
from datetime import datetime
import argparse

# Configuration
D = 20000
IMG_SIZE = 64
PROTOTYPE_PATH = 'prototypes.pkl'
SIMILARITY_THRESHOLD = 0.65  # Adjust this based on your requirements

# Initialize detectors
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_image(img):
    """Preprocess image to improve face detection"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    return gray

def detect_face_multi_method(img):
    """Try multiple face detection methods"""
    gray = preprocess_image(img)
    
    # Method 1: Dlib with different upsample levels
    for upsample in [1, 2, 0]:
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

def majority_vote(hv_list):
    """Compute majority vote for hypervector list"""
    sum_vec = np.sum(hv_list, axis=0)
    return (sum_vec >= (len(hv_list) / 2)).astype(np.uint8)

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

def cosine_similarity(hv1, hv2):
    """Calculate cosine similarity between two hypervectors"""
    # Convert to float for computation
    hv1_f = hv1.astype(np.float32)
    hv2_f = hv2.astype(np.float32)
    
    # Compute cosine similarity
    dot_product = np.dot(hv1_f, hv2_f)
    norm1 = np.linalg.norm(hv1_f)
    norm2 = np.linalg.norm(hv2_f)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

def hamming_similarity(hv1, hv2):
    """Calculate Hamming similarity between two hypervectors"""
    # XOR to find differences, then count similarities
    differences = np.bitwise_xor(hv1, hv2)
    similarities = len(hv1) - np.sum(differences)
    return similarities / len(hv1)

def load_prototypes(prototype_path):
    """Load stored prototypes and encoding dictionaries"""
    if not os.path.exists(prototype_path):
        raise FileNotFoundError(f"Prototype file {prototype_path} not found. Please run preprocessing first.")
    
    with open(prototype_path, 'rb') as f:
        data = pickle.load(f)
    
    return data['prototypes'], data['loc_dict'], data['int_dict'], data['grad_dict']

def recognize_face(input_image_path, prototype_path=PROTOTYPE_PATH, threshold=SIMILARITY_THRESHOLD, 
                  similarity_method='hamming', show_all_scores=False):
    """
    Recognize face in input image
    
    Args:
        input_image_path: Path to input image
        prototype_path: Path to stored prototypes
        threshold: Similarity threshold for recognition
        similarity_method: 'hamming' or 'cosine'
        show_all_scores: Whether to show all similarity scores
    
    Returns:
        Dictionary with recognition results
    """
    
    # Load prototypes
    try:
        prototypes, loc_dict, int_dict, grad_dict = load_prototypes(prototype_path)
        print(f"Loaded {len(prototypes)} prototypes")
    except Exception as e:
        return {'error': f"Failed to load prototypes: {e}"}
    
    # Load and preprocess input image
    try:
        input_img = cv2.imread(input_image_path)
        if input_img is None:
            return {'error': f"Could not load image: {input_image_path}"}
        
        print("Preprocessing input image...")
        preprocessed_img = align_face_robust(input_img)
        
        print("Encoding input image...")
        input_encoding = encode_image(preprocessed_img, loc_dict, int_dict, grad_dict)
        
    except Exception as e:
        return {'error': f"Failed to process input image: {e}"}
    
    # Compare with all prototypes
    similarities = {}
    
    print(f"Comparing with {len(prototypes)} prototypes...")
    for person_name, prototype in prototypes.items():
        if similarity_method == 'cosine':
            similarity = cosine_similarity(input_encoding, prototype)
        else:  # hamming
            similarity = hamming_similarity(input_encoding, prototype)
        
        similarities[person_name] = similarity
    
    # Find best match
    best_match = max(similarities.items(), key=lambda x: x[1])
    best_person, best_score = best_match
    
    # Prepare results
    results = {
        'input_image': input_image_path,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'best_match': best_person,
        'best_score': best_score,
        'threshold': threshold,
        'similarity_method': similarity_method,
        'is_match': best_score >= threshold,
        'all_scores': similarities if show_all_scores else None
    }
    
    return results

def display_results(results):
    """Display recognition results in a formatted way"""
    print("\n" + "="*60)
    print("FACE RECOGNITION RESULTS")
    print("="*60)
    print(f"Input Image: {results['input_image']}")
    print(f"Timestamp: {results['timestamp']}")
    print(f"Similarity Method: {results['similarity_method'].upper()}")
    print(f"Threshold: {results['threshold']:.3f}")
    print("-"*60)
    
    if 'error' in results:
        print(f"ERROR: {results['error']}")
        return
    
    print(f"Best Match: {results['best_match']}")
    print(f"Similarity Score: {results['best_score']:.4f}")
    
    if results['is_match']:
        print("✅ MATCH FOUND!")
        print(f"Person identified: {results['best_match']}")
    else:
        print("❌ NO MATCH")
        print("Person not found in database")
    
    if results['all_scores']:
        print("\nAll Similarity Scores:")
        print("-"*40)
        sorted_scores = sorted(results['all_scores'].items(), key=lambda x: x[1], reverse=True)
        for person, score in sorted_scores[:10]:  # Show top 10
            status = "✅" if score >= results['threshold'] else "❌"
            print(f"{status} {person}: {score:.4f}")
    
    print("="*60)

def batch_recognize(input_folder, prototype_path=PROTOTYPE_PATH, threshold=SIMILARITY_THRESHOLD):
    """Recognize faces in multiple images"""
    results = []
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(input_folder, filename)
            print(f"\nProcessing: {filename}")
            
            result = recognize_face(image_path, prototype_path, threshold)
            result['filename'] = filename
            results.append(result)
            
            # Quick summary
            if 'error' in result:
                print(f"ERROR: {result['error']}")
            elif result['is_match']:
                print(f"✅ MATCH: {result['best_match']} (Score: {result['best_score']:.4f})")
            else:
                print(f"❌ NO MATCH (Best: {result['best_match']}, Score: {result['best_score']:.4f})")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Face Recognition using HDIC')
    parser.add_argument('--input', '-i', required=True, help='Input image path or folder')
    parser.add_argument('--prototypes', '-p', default=PROTOTYPE_PATH, help='Prototypes file path')
    parser.add_argument('--threshold', '-t', type=float, default=SIMILARITY_THRESHOLD, 
                       help='Similarity threshold for recognition')
    parser.add_argument('--method', '-m', choices=['hamming', 'cosine'], default='hamming',
                       help='Similarity calculation method')
    parser.add_argument('--show-all', '-a', action='store_true', 
                       help='Show all similarity scores')
    parser.add_argument('--batch', '-b', action='store_true',
                       help='Process folder of images')
    
    args = parser.parse_args()
    
    if args.batch:
        if not os.path.isdir(args.input):
            print(f"Error: {args.input} is not a directory")
            return
        
        results = batch_recognize(args.input, args.prototypes, args.threshold)
        
        # Summary statistics
        total = len(results)
        matches = sum(1 for r in results if r.get('is_match', False))
        errors = sum(1 for r in results if 'error' in r)
        
        print(f"\n{'='*60}")
        print("BATCH PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total images processed: {total}")
        print(f"Matches found: {matches}")
        print(f"No matches: {total - matches - errors}")
        print(f"Errors: {errors}")
        print(f"Success rate: {((total - errors) / total * 100):.1f}%")
        
    else:
        if not os.path.isfile(args.input):
            print(f"Error: {args.input} is not a file")
            return
        
        results = recognize_face(
            args.input, 
            args.prototypes, 
            args.threshold, 
            args.method, 
            args.show_all
        )
        
        display_results(results)

if __name__ == "__main__":
    # Example usage if run directly
    if len(os.sys.argv) == 1:
        # Demo mode - replace with your test image path
        test_image = "test_image.jpg"  # Replace with your test image path
        
        if os.path.exists(test_image):
            print("Running in demo mode...")
            results = recognize_face(test_image, show_all_scores=True)
            display_results(results)
        else:
            print("Usage examples:")
            print("python face_recognition.py --input test_image.jpg")
            print("python face_recognition.py --input test_folder --batch")
            print("python face_recognition.py --input test_image.jpg --threshold 0.7 --show-all")
    else:
        main()