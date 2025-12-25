import torch
import os
import sys
import numpy as np
from datetime import datetime

sys.path.append('src')

from models.distilbert_classifier import DistilBertClassifire
from models.dataset import create_date_loaders
from models.trainer import JITDefectTrainer

def main():
    print("=" * 80)
    print("JIT DEFECT PREDICTION - MODEL TRAINING")
    print("=" * 80)
    #configuration
    CONFIG={
        #path
        'tarin_csv':'data/processed/train.csv',
        'val_csv':'data/processed/val.csv',
        'test_csv':'data/processed/test.csv',
        #model parameters
        'max_length':128,
        'dropout':0.3,
        #training parameters
        'batch_size':16,
        'num_epochs':3,
        'learning_rate':2e-5,
        'weight_decay':0.01,
        #path
        'model_save_path':'outputs/models/best_model.pth',
        'results_path': 'outputs/results/'
    }
    print("\n Configuration:")
    for key,value in CONFIG.items():
        print(f"{key}:{value}")
    
    # Create output directories
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/results',exist_ok=True)

    #setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"the device using : {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    #load data
    print("\n" + "=" * 80)
    train_loader, val_loader, test_loader = create_date_loaders(CONFIG['tarin_csv'],CONFIG['val_csv'],CONFIG['test_csv'],
                                                               batch_size = CONFIG['batch_size'], max_length = CONFIG['max_length'])
    train_dataset = train_loader.dataset
    train_labels = train_dataset.get_labels()
     #creat model
    print("\n" + "=" * 80)
    print("Initializing Model")
    print("=" * 80)
    model = DistilBertClassifire(n_classes=2, dropout=CONFIG['dropout'])
    print(f"\n✓ Model created")
    print(f"  Total parameters: {model.get_num_parameters():,}")
    print(f"  Trainable parameters: {model.get_num_trainable_parameters():,}")
    #create trainer
    trainer = JITDefectTrainer(model=model, device=device,learning_rate=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
    print("\n✓ Computing class weights for imbalanced data...")
    class_weights = trainer.compute_class_weights(train_labels)
    print(f"  Class weights: {class_weights}")
    #training
    print("\n" + "=" * 80)
    history = trainer.train(train_loader=train_loader,val_loader=val_loader,num_epochs=CONFIG['num_epochs'],class_weights= class_weights,save_path=CONFIG['model_save_path'])
    #final evalation on test set
    print("\n" + "=" * 80)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 80)
    trainer.load_best_model(CONFIG['model_save_path'])
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)
    print("\n" + "=" * 80)
    print("FINAL TEST RESULTS (For Thesis)")
    print("=" * 80)
    print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall:    {test_metrics['recall']:.4f}")
    print(f"Test F1-Score:  {test_metrics['f1']:.4f}")
    print("=" * 80)
    #save result
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = os.path.join(CONFIG['results_path'],f'result_{timestamp}.txt')
    with open(result_file,'w') as f:
        f.write("=" * 80 + "\n")
        f.write("JIT DEFECT PREDICTION - TRAINING RESULTS\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        f.write("CONFIGURATION:\n")
        for key, value in CONFIG.items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("FINAL TEST RESULTS:\n")
        f.write("=" * 80 + "\n")
        f.write(f"Accuracy:  {test_metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"Recall:    {test_metrics['recall']:.4f}\n")
        f.write(f"F1-Score:  {test_metrics['f1']:.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("TRAINING HISTORY:\n")
        f.write("=" * 80 + "\n")
        for epoch in range(len(history['train_losses'])):
            f.write(f"\nEpoch {epoch + 1}:\n")
            f.write(f"  Train Loss: {history['train_losses'][epoch]:.4f}\n")
            f.write(f"  Val Loss:   {history['val_losses'][epoch]:.4f}\n")
            f.write(f"  Val Acc:    {history['val_accuracies'][epoch]:.4f}\n")
            f.write(f"  Val F1:     {history['val_f1_scores'][epoch]:.4f}\n")
    print(f"\n✓ Results saved to: {result_file}")

    #save prediction
    predictions_file = os.path.join(CONFIG['results_path'], f'test_predictions_{timestamp}.txt')
    
    with open(predictions_file, 'w') as f:
        f.write("Test Set Predictions\n")
        f.write("=" * 80 + "\n")
        f.write("Format: True_Label | Predicted_Label | Probability\n")
        f.write("=" * 80 + "\n\n")
        
        for i, (true_label, pred_label, prob) in enumerate(zip(
            test_metrics['labels'],
            test_metrics['predictions'],
            test_metrics['probabilities']
        )):
            f.write(f"{i+1:4d}: {true_label} | {pred_label} | {prob:.4f}\n")
    
    print(f"✓ Predictions saved to: {predictions_file}")
    
    print("\n" + "=" * 80)
    print("✓✓✓ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nBest model saved at: {CONFIG['model_save_path']}")
    print(f"Results saved at: {result_file}")
    print(f"\nYou can now use these results in your thesis!")

if __name__ == "__main__":
    main()
