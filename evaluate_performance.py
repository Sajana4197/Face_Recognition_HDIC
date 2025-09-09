import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    roc_curve,
    auc,
    confusion_matrix,
    DetCurveDisplay,
)

from datasetPreprocessing2 import align_face_robust, encode_image, IMG_SIZE

# Load prototypes and encoding dicts
with open("prototypes_clustered.pkl", "rb") as f:
    data = pickle.load(f)

prototypes = data["prototypes"]
loc_dict = data["loc_dict"]
int_dict = data["int_dict"]
grad_dict = data["grad_dict"]

dataset_path = "dataset"
threshold = 0.65
method = "hamming"

def hamming_distance(hv1, hv2):
    return np.sum(hv1 != hv2)

def tanimoto_similarity(hv1, hv2):
    intersection = np.sum(hv1 & hv2)
    union = np.sum(hv1) + np.sum(hv2) - intersection
    return intersection / union if union != 0 else 0

# Store results
y_true = []
y_pred = []
scores = []
binary_labels = []  # 1 = match, 0 = impostor

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)
    if not os.path.isdir(person_path):
        continue

    for img_file in os.listdir(person_path):
        img_path = os.path.join(person_path, img_file)
        img = cv2.imread(img_path)
        try:
            aligned = align_face_robust(img)
            encoded = encode_image(aligned, loc_dict, int_dict, grad_dict)
        except Exception as e:
            print(f"Skipping {img_path}: {e}")
            continue

        best_match = None
        best_score = -1 if method == "tanimoto" else 0

        for candidate, proto_list in prototypes.items():
            for proto in proto_list:
                if method == "tanimoto":
                    score = tanimoto_similarity(encoded, proto)
                else:
                    dist = hamming_distance(encoded, proto)
                    score = 1 - (dist / len(encoded))

                if score > best_score:
                    best_score = score
                    best_match = candidate

        y_true.append(person)
        pred_label = best_match if best_score >= threshold else "unknown"
        y_pred.append(pred_label)
        scores.append(best_score)
        binary_labels.append(1 if pred_label == person else 0)

# === METRICS ===
print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred))

# === ROC Curve ===
fpr, tpr, thresholds = roc_curve(binary_labels, scores)
roc_auc = auc(fpr, tpr)
fnr = 1 - tpr

# EER
eer_idx = np.nanargmin(np.abs(fpr - fnr))
eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
eer_threshold = thresholds[eer_idx]

# FAR / FRR at threshold
far = fpr[eer_idx]
frr = fnr[eer_idx]

print("\n=== Verification Metrics ===")
print(f"AUC: {roc_auc:.4f}")
print(f"EER: {eer:.4f} at threshold {eer_threshold:.4f}")
print(f"FAR (False Accept Rate): {far:.4f}")
print(f"FRR (False Reject Rate): {frr:.4f}")

# === ROC Curve Plot ===
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color='blue')
plt.plot([0, 1], [0, 1], linestyle="--", color='gray')
plt.scatter(fpr[eer_idx], tpr[eer_idx], color="red", label=f"EER = {eer:.2%}")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.savefig("roc_curve.png")
plt.close()

# === DET Curve Plot ===
DetCurveDisplay(fpr=fpr, fnr=fnr).plot()
plt.title("DET Curve")
plt.grid(True)
plt.savefig("det_curve.png")
plt.close()

print("\nPlots saved as 'roc_curve.png' and 'det_curve.png'")
