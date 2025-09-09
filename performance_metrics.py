import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
import time
from datasetPreprocessing2 import align_face_robust, encode_image
from face_recognition2 import hamming_distance, tanimoto_similarity

class FaceRecognitionEvaluator:
    def __init__(self, prototype_path='prototypes_clustered.pkl', test_dataset_path='dataset'):
        """
        Initialize the evaluator
        
        Args:
            prototype_path: Path to the prototypes file
            test_dataset_path: Path to test dataset (organized as dataset/person_name/images)
        """
        self.prototype_path = prototype_path
        self.test_dataset_path = test_dataset_path
        self.load_prototypes()
        
    def load_prototypes(self):
        """Load prototypes and encoding dictionaries"""
        with open(self.prototype_path, 'rb') as f:
            data = pickle.load(f)
        
        self.prototypes = data['prototypes']
        self.loc_dict = data['loc_dict']
        self.int_dict = data['int_dict']
        self.grad_dict = data['grad_dict']
        self.person_names = list(self.prototypes.keys())
        print(f"Loaded prototypes for {len(self.person_names)} people")
    
    def predict_person(self, image_path, method='hamming', threshold=0.52):
        """
        Predict person from image
        
        Returns:
            predicted_person, confidence_score, processing_time
        """
        start_time = time.time()
        
        try:
            img = cv2.imread(image_path)
            preprocessed = align_face_robust(img)
            input_hv = encode_image(preprocessed, self.loc_dict, self.int_dict, self.grad_dict)
            
            best_person = None
            best_score = -1 if method == 'tanimoto' else 0
            
            for person, proto_list in self.prototypes.items():
                for proto in proto_list:
                    if method == 'tanimoto':
                        score = tanimoto_similarity(input_hv, proto)
                        if score > best_score:
                            best_score = score
                            best_person = person
                    else:  # hamming
                        distance = hamming_distance(input_hv, proto)
                        similarity = 1 - (distance / len(input_hv))
                        if similarity > best_score:
                            best_score = similarity
                            best_person = person
            
            processing_time = time.time() - start_time
            
            # Apply threshold
            if best_score >= threshold:
                return best_person, best_score, processing_time
            else:
                return "unknown", best_score, processing_time
                
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return "error", 0.0, time.time() - start_time
    
    def evaluate_performance(self, method='hamming', threshold=0.52):
        """
        Evaluate performance on test dataset
        
        Returns:
            Dictionary containing all performance metrics
        """
        true_labels = []
        predicted_labels = []
        confidence_scores = []
        processing_times = []
        
        print(f"Evaluating with method: {method}, threshold: {threshold}")
        
        # Process test images
        for person_name in os.listdir(self.test_dataset_path):
            person_path = os.path.join(self.test_dataset_path, person_name)
            if not os.path.isdir(person_path):
                continue
                
            print(f"Processing {person_name}...")
            for img_file in os.listdir(person_path):
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                img_path = os.path.join(person_path, img_file)
                predicted, confidence, proc_time = self.predict_person(
                    img_path, method=method, threshold=threshold
                )
                
                true_labels.append(person_name)
                predicted_labels.append(predicted)
                confidence_scores.append(confidence)
                processing_times.append(proc_time)
        
        # Calculate metrics
        metrics = self.calculate_metrics(true_labels, predicted_labels, confidence_scores, processing_times)
        return metrics
    
    def calculate_metrics(self, true_labels, predicted_labels, confidence_scores, processing_times):
        """Calculate comprehensive performance metrics"""
        
        # Basic classification metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        
        # Handle unknown predictions for precision, recall, f1
        known_mask = np.array(predicted_labels) != "unknown"
        true_known = np.array(true_labels)[known_mask]
        pred_known = np.array(predicted_labels)[known_mask]
        
        if len(pred_known) > 0:
            precision = precision_score(true_known, pred_known, average='weighted', zero_division=0)
            recall = recall_score(true_known, pred_known, average='weighted', zero_division=0)
            f1 = f1_score(true_known, pred_known, average='weighted', zero_division=0)
        else:
            precision = recall = f1 = 0.0
        
        # Recognition rate (excluding unknowns)
        correct_predictions = sum(1 for t, p in zip(true_labels, predicted_labels) if t == p and p != "unknown")
        total_predictions = len(true_labels)
        recognition_rate = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # False acceptance and rejection rates
        false_accepts = sum(1 for t, p in zip(true_labels, predicted_labels) 
                           if t != p and p != "unknown" and p != "error")
        false_rejects = sum(1 for t, p in zip(true_labels, predicted_labels) 
                           if p == "unknown" and t in self.person_names)
        
        total_genuine = sum(1 for t in true_labels if t in self.person_names)
        total_impostor_attempts = total_predictions - total_genuine
        
        far = false_accepts / total_impostor_attempts if total_impostor_attempts > 0 else 0
        frr = false_rejects / total_genuine if total_genuine > 0 else 0
        
        # Performance metrics
        avg_processing_time = np.mean(processing_times)
        avg_confidence = np.mean(confidence_scores)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'recognition_rate': recognition_rate,
            'false_acceptance_rate': far,
            'false_rejection_rate': frr,
            'avg_processing_time': avg_processing_time,
            'avg_confidence': avg_confidence,
            'true_labels': true_labels,
            'predicted_labels': predicted_labels,
            'confidence_scores': confidence_scores,
            'processing_times': processing_times
        }
        
        return metrics
    
    def plot_confusion_matrix(self, metrics, method='hamming', save_path='confusion_matrix.png'):
        """Generate and save confusion matrix plot"""
        true_labels = metrics['true_labels']
        predicted_labels = metrics['predicted_labels']
        
        # Get unique labels
        all_labels = sorted(list(set(true_labels + predicted_labels)))
        
        # Create confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels, labels=all_labels)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=all_labels, yticklabels=all_labels)
        plt.title(f'Confusion Matrix - {method.upper()} Method')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to {save_path}")
    
    def plot_performance_metrics(self, metrics, method='hamming', save_path='performance_metrics.png'):
        """Plot key performance metrics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy, Precision, Recall, F1
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metric_values = [metrics['accuracy'], metrics['precision'], 
                        metrics['recall'], metrics['f1_score']]
        
        ax1.bar(metric_names, metric_values, color=['skyblue', 'lightgreen', 'orange', 'pink'])
        ax1.set_title(f'Classification Metrics - {method.upper()}')
        ax1.set_ylabel('Score')
        ax1.set_ylim(0, 1)
        for i, v in enumerate(metric_values):
            ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Error rates
        error_names = ['False Acceptance Rate', 'False Rejection Rate']
        error_values = [metrics['false_acceptance_rate'], metrics['false_rejection_rate']]
        
        ax2.bar(error_names, error_values, color=['red', 'darkred'])
        ax2.set_title('Error Rates')
        ax2.set_ylabel('Rate')
        for i, v in enumerate(error_values):
            ax2.text(i, v + max(error_values) * 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Confidence score distribution
        ax3.hist(metrics['confidence_scores'], bins=20, alpha=0.7, color='purple')
        ax3.set_title('Confidence Score Distribution')
        ax3.set_xlabel('Confidence Score')
        ax3.set_ylabel('Frequency')
        ax3.axvline(np.mean(metrics['confidence_scores']), color='red', 
                   linestyle='--', label=f'Mean: {np.mean(metrics["confidence_scores"]):.3f}')
        ax3.legend()
        
        # Processing time distribution
        ax4.hist(metrics['processing_times'], bins=20, alpha=0.7, color='green')
        ax4.set_title('Processing Time Distribution')
        ax4.set_xlabel('Processing Time (seconds)')
        ax4.set_ylabel('Frequency')
        ax4.axvline(np.mean(metrics['processing_times']), color='red', 
                   linestyle='--', label=f'Mean: {np.mean(metrics["processing_times"]):.3f}s')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Performance metrics plot saved to {save_path}")
    
    def plot_threshold_analysis(self, method='hamming', save_path='threshold_analysis.png'):
        """Analyze performance across different thresholds"""
        thresholds = np.arange(0.1, 1.0, 0.05)
        accuracies = []
        f1_scores = []
        recognition_rates = []
        fars = []
        frrs = []
        
        print("Performing threshold analysis...")
        for threshold in thresholds:
            print(f"Testing threshold: {threshold:.2f}")
            metrics = self.evaluate_performance(method=method, threshold=threshold)
            accuracies.append(metrics['accuracy'])
            f1_scores.append(metrics['f1_score'])
            recognition_rates.append(metrics['recognition_rate'])
            fars.append(metrics['false_acceptance_rate'])
            frrs.append(metrics['false_rejection_rate'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Performance metrics vs threshold
        ax1.plot(thresholds, accuracies, 'b-', label='Accuracy', marker='o')
        ax1.plot(thresholds, f1_scores, 'g-', label='F1-Score', marker='s')
        ax1.plot(thresholds, recognition_rates, 'r-', label='Recognition Rate', marker='^')
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Score')
        ax1.set_title(f'Performance vs Threshold - {method.upper()}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Error rates vs threshold
        ax2.plot(thresholds, fars, 'r-', label='False Acceptance Rate', marker='o')
        ax2.plot(thresholds, frrs, 'b-', label='False Rejection Rate', marker='s')
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Error Rate')
        ax2.set_title('Error Rates vs Threshold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Threshold analysis plot saved to {save_path}")
        
        return thresholds, accuracies, f1_scores, recognition_rates, fars, frrs
    
    def compare_methods(self, thresholds=[0.5, 0.65, 0.8], save_path='method_comparison.png'):
        """Compare different similarity methods"""
        methods = ['hamming', 'tanimoto']
        results = {}
        
        for method in methods:
            results[method] = {}
            for threshold in thresholds:
                print(f"Evaluating {method} with threshold {threshold}")
                metrics = self.evaluate_performance(method=method, threshold=threshold)
                results[method][threshold] = metrics
        
        # Plot comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        metric_names = ['accuracy', 'f1_score', 'recognition_rate']
        metric_labels = ['Accuracy', 'F1-Score', 'Recognition Rate']
        
        width = 0.35
        x = np.arange(len(thresholds))
        
        for i, (metric, label) in enumerate(zip(metric_names, metric_labels)):
            ax = [ax1, ax2, ax3][i]
            
            hamming_values = [results['hamming'][t][metric] for t in thresholds]
            tanimoto_values = [results['tanimoto'][t][metric] for t in thresholds]
            
            ax.bar(x - width/2, hamming_values, width, label='Hamming', alpha=0.8)
            ax.bar(x + width/2, tanimoto_values, width, label='Tanimoto', alpha=0.8)
            
            ax.set_xlabel('Threshold')
            ax.set_ylabel(label)
            ax.set_title(f'{label} Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(thresholds)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Processing time comparison
        hamming_times = [results['hamming'][t]['avg_processing_time'] for t in thresholds]
        tanimoto_times = [results['tanimoto'][t]['avg_processing_time'] for t in thresholds]
        
        ax4.bar(x - width/2, hamming_times, width, label='Hamming', alpha=0.8)
        ax4.bar(x + width/2, tanimoto_times, width, label='Tanimoto', alpha=0.8)
        ax4.set_xlabel('Threshold')
        ax4.set_ylabel('Processing Time (seconds)')
        ax4.set_title('Processing Time Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(thresholds)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Method comparison plot saved to {save_path}")
        
        return results
    
    def generate_report(self, metrics, method='hamming', threshold=0.52, save_path='performance_report.txt'):
        """Generate a comprehensive text report"""
        report = []
        report.append("="*60)
        report.append("FACE RECOGNITION SYSTEM PERFORMANCE REPORT")
        report.append("="*60)
        report.append(f"Method: {method.upper()}")
        report.append(f"Threshold: {threshold}")
        report.append(f"Test Dataset: {self.test_dataset_path}")
        report.append(f"Number of People in Database: {len(self.person_names)}")
        report.append(f"Total Test Images: {len(metrics['true_labels'])}")
        report.append("")
        
        report.append("CLASSIFICATION METRICS:")
        report.append("-"*30)
        report.append(f"Accuracy: {metrics['accuracy']:.4f}")
        report.append(f"Precision: {metrics['precision']:.4f}")
        report.append(f"Recall: {metrics['recall']:.4f}")
        report.append(f"F1-Score: {metrics['f1_score']:.4f}")
        report.append(f"Recognition Rate: {metrics['recognition_rate']:.4f}")
        report.append("")
        
        report.append("ERROR RATES:")
        report.append("-"*30)
        report.append(f"False Acceptance Rate (FAR): {metrics['false_acceptance_rate']:.4f}")
        report.append(f"False Rejection Rate (FRR): {metrics['false_rejection_rate']:.4f}")
        report.append("")
        
        report.append("PERFORMANCE METRICS:")
        report.append("-"*30)
        report.append(f"Average Processing Time: {metrics['avg_processing_time']:.4f} seconds")
        report.append(f"Average Confidence Score: {metrics['avg_confidence']:.4f}")
        report.append("")
        
        report.append("DETAILED STATISTICS:")
        report.append("-"*30)
        confidence_scores = np.array(metrics['confidence_scores'])
        processing_times = np.array(metrics['processing_times'])
        
        report.append(f"Confidence Score - Min: {np.min(confidence_scores):.4f}, Max: {np.max(confidence_scores):.4f}, Std: {np.std(confidence_scores):.4f}")
        report.append(f"Processing Time - Min: {np.min(processing_times):.4f}s, Max: {np.max(processing_times):.4f}s, Std: {np.std(processing_times):.4f}s")
        
        report.append("="*60)
        
        with open(save_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Performance report saved to {save_path}")
        return '\n'.join(report)

def main():
    """Main function to run comprehensive evaluation"""
    # Initialize evaluator
    evaluator = FaceRecognitionEvaluator(
        prototype_path='prototypes_clustered.pkl',
        test_dataset_path='dataset'  # Change this to your test dataset path
    )
    
    # Evaluate with default settings
    print("Starting comprehensive evaluation...")
    metrics = evaluator.evaluate_performance(method='hamming', threshold=0.52)
    
    # Generate plots
    evaluator.plot_confusion_matrix(metrics, method='hamming')
    evaluator.plot_performance_metrics(metrics, method='hamming')
    
    # Threshold analysis
    evaluator.plot_threshold_analysis(method='hamming')
    
    # Compare methods
    evaluator.compare_methods(thresholds=[0.5, 0.65, 0.8])
    
    # Generate report
    report = evaluator.generate_report(metrics, method='hamming', threshold=0.65)
    print("\n" + report)

if __name__ == "__main__":
    main()