import torch
import numpy as np
import pandas as pd
import sys
sys.path.append('src')

from models.distilbert_classifier import DistilBertClassifire
from models.dataset import commitDataset
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def evaluate_with_threshold(model, device, data_loader, threshold=0.5):

    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']
            
            # Get predictions
            outputs = model(input_ids, attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Apply custom threshold
            predictions = (probabilities[:, 1] >= threshold).long()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary', zero_division=0
    )
    
    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def find_best_threshold():
    """Test different thresholds and find the best one"""
    
    # Load model
    print("Loading model...")
    model = DistilBertClassifire(n_classes=2, dropout=0.3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load('outputs/models/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load test data
    print("Loading test data...")
    test_dataset = commitDataset(csv_files='data/processed/test.csv', max_length=128)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Test different thresholds
    print("\n" + "=" * 80)
    print("TESTING DIFFERENT THRESHOLDS")
    print("=" * 80)
    print(f"{'Threshold':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 80)
    
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    results = []
    
    for threshold in thresholds:
        metrics = evaluate_with_threshold(model, device, test_loader, threshold)
        results.append(metrics)
        
        print(f"{threshold:<12.2f} {metrics['accuracy']:<12.4f} "
              f"{metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
              f"{metrics['f1']:<12.4f}")
    
    # Find best threshold for recall
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    # Best for recall
    best_recall = max(results, key=lambda x: x['recall'])
    print(f"\nBest for Recall:")
    print(f"  Threshold: {best_recall['threshold']:.2f}")
    print(f"  Recall: {best_recall['recall']:.4f}")
    print(f"  Precision: {best_recall['precision']:.4f}")
    print(f"  F1: {best_recall['f1']:.4f}")
    
    # Best for F1
    best_f1 = max(results, key=lambda x: x['f1'])
    print(f"\nBest for F1 (balanced):")
    print(f"  Threshold: {best_f1['threshold']:.2f}")
    print(f"  Recall: {best_f1['recall']:.4f}")
    print(f"  Precision: {best_f1['precision']:.4f}")
    print(f"  F1: {best_f1['f1']:.4f}")
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv('outputs/results/threshold_analysis.csv', index=False)
    print(f"\n✓ Results saved to: outputs/results/threshold_analysis.csv")


if __name__ == "__main__":
    find_best_threshold()