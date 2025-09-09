# evaluate_performance.py
import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import timeit

from datasetPreprocessing2 import align_face_robust, encode_image
from face_recognition2 import hamming_distance, tanimoto_similarity

# Config
IMG_SIZE = 64
PROTOTYPE_PATH = 'prototypes_clustered.pkl'
DATASET_PATH = 'test'   # <-- your test set folder with 50 persons
METHOD = 'hamming'   # or 'tanimoto'
PLOT_DIR = 'plots'

os.makedirs(PLOT_DIR, exist_ok=True)

# Load prototypes + encoding dicts
with open(PROTOTYPE_PATH, 'rb') as f:
    data = pickle.load(f)

prototypes = data['prototypes']
loc_dict, int_dict, grad_dict = data['loc_dict'], data['int_dict'], data['grad_dict']

# ---- Helper functions ----
def compute_similarity(input_hv, proto, method="hamming"):
    if method == "tanimoto":
        return tanimoto_similarity(input_hv, proto)
    else:
        distance = hamming_distance(input_hv, proto)
        return 1 - (distance / len(input_hv))

def load_and_encode(img_path):
    img = cv2.imread(img_path)
    preprocessed = align_face_robust(img)
    return encode_image(preprocessed, loc_dict, int_dict, grad_dict)

# ---- Collect scores across all persons ----
persons = [p for p in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, p))]

genuine_scores = []
imposter_scores = []

# Loop through each person
for target_person in tqdm(persons, desc="Evaluating persons"):
    target_path = os.path.join(DATASET_PATH, target_person)
    target_images = [os.path.join(target_path, f) for f in os.listdir(target_path)]

    # ---- Genuine comparisons ----
    for img_path in tqdm(target_images, desc=f"Genuine ({target_person})", leave=False):
        try:
            hv = load_and_encode(img_path)
            for proto in prototypes[target_person]:
                sim = compute_similarity(hv, proto, METHOD)
                genuine_scores.append(sim)
        except Exception as e:
            print(f"Skipping {img_path}: {e}")

    # ---- Imposter comparisons ----
    for other_person in tqdm(persons, desc=f"Imposters vs {target_person}", leave=False):
        if other_person == target_person:
            continue

        other_path = os.path.join(DATASET_PATH, other_person)
        img_files = os.listdir(other_path)
        found_valid = False

        for img_file in img_files:  # try each image until success
            img_path = os.path.join(other_path, img_file)
            try:
                hv = load_and_encode(img_path)
                for proto in prototypes[target_person]:
                    sim = compute_similarity(hv, proto, METHOD)
                    imposter_scores.append(sim)
                found_valid = True
                break  # stop after first valid image
            except Exception as e:
                print(f"Skipping {img_path}: {e}")

        if not found_valid:
            print(f"No valid images found for imposter {other_person}, skipping...")

# ---- Threshold Sweep ----
thresholds = np.linspace(0, 1, 100)
FAR, FRR, TAR, TRR = [], [], [], []

for t in tqdm(thresholds, desc="Threshold Sweep"):
    # Genuine
    genuine_accepts = sum(s >= t for s in genuine_scores)
    genuine_rejects = len(genuine_scores) - genuine_accepts
    
    # Imposter
    imposter_accepts = sum(s >= t for s in imposter_scores)
    imposter_rejects = len(imposter_scores) - imposter_accepts
    
    TAR.append(genuine_accepts / len(genuine_scores))
    FRR.append(genuine_rejects / len(genuine_scores))
    FAR.append(imposter_accepts / len(imposter_scores))
    TRR.append(imposter_rejects / len(imposter_scores))

# ---- Equal Error Rate (EER) ----
diffs = np.abs(np.array(FAR) - np.array(FRR))
eer_idx = np.argmin(diffs)
eer_threshold = thresholds[eer_idx]
eer_value = (FAR[eer_idx] + FRR[eer_idx]) / 2
print(f"EER: {eer_value:.3f} at threshold {eer_threshold:.3f}")

# ---- ROC Curve ----
plt.figure()
plt.plot(FAR, TAR, label="ROC Curve")
plt.xlabel("False Accept Rate (FAR)")
plt.ylabel("True Accept Rate (TAR)")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(PLOT_DIR, "roc_curve.png"))

# ---- FAR & FRR vs Threshold (with EER) ----
plt.figure()
plt.plot(thresholds, FAR, label="FAR", color="red")
plt.plot(thresholds, FRR, label="FRR", color="blue")
plt.axvline(eer_threshold, color="green", linestyle="--", label=f"EER Threshold = {eer_threshold:.3f}")
plt.axhline(eer_value, color="purple", linestyle="--", label=f"EER = {eer_value:.3f}")
plt.scatter([eer_threshold], [eer_value], color="black", zorder=5)  # Mark the EER point
plt.xlabel("Threshold")
plt.ylabel("Rate")
plt.title("FAR & FRR vs Threshold (with EER)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(PLOT_DIR, "far_frr_threshold.png"))

# ---- TAR & TRR vs Threshold ----
plt.figure()
plt.plot(thresholds, TAR, label="TAR", color="green")
plt.plot(thresholds, TRR, label="TRR", color="orange")
plt.xlabel("Threshold")
plt.ylabel("Rate")
plt.title("TAR & TRR vs Threshold")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(PLOT_DIR, "tar_trr_threshold.png"))

# ---- DET Curve (FRR vs FAR, log scale) ----
plt.figure()
plt.plot(FAR, FRR, label="DET Curve")
plt.scatter([FAR[eer_idx]], [FRR[eer_idx]], color="black", zorder=5, label="EER Point")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("False Accept Rate (FAR)")
plt.ylabel("False Reject Rate (FRR)")
plt.title("DET Curve")
plt.legend()
plt.grid(True, which="both")
plt.savefig(os.path.join(PLOT_DIR, "det_curve.png"))

# ---- Timing Test ----
sample_img = os.path.join(DATASET_PATH, persons[0], os.listdir(os.path.join(DATASET_PATH, persons[0]))[0])
hv = load_and_encode(sample_img)
proto = prototypes[persons[0]][0]

def test_timing():
    _ = compute_similarity(hv, proto, METHOD)

avg_time = timeit.timeit(test_timing, number=50) / 50
print(f"Average comparison time: {avg_time*1000:.3f} ms")
