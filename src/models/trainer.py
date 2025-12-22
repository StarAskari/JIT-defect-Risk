import torch
import torch.nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm
import time

class JITDefectionTrainer:
    def __init__(self, model, device, learning_rate = 2e-5, weight_decay = 0.01):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr = learning_rate,
            weight_decay=weight_decay
        )
        #for tracking best model
        self.train_losses =[]
        self.val_losses=[]
        self.val_accuracies=[]
        self.val_f1_scores = []
    
    def compute_class_weights(self, train_labels):
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
        return torch.tensor(class_weights, dtype=torch.float).to(self.device)
    
    def train_epoc(self, train_loader, class_weights=None):
        self.model.train()
        total_loss = 0
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()

        progress_bar = tqdm(train_loader, desc = "Training")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['aatention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            self.optimizer.zero_gard()
            output = self.model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_gard_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix({'loss' :f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def evaluate(self, data_loader):
        self.model.eval()
        total_loss =0
        all_prediction = []
        all_labels =[]
        all_probabilities =[]

        criterion = nn.CrossEntropyLoss()
        progress_bar = tqdm(data_loader, desc="Evaluating")

        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = batch['label'].to(self.device)
                probabilities = torch.softmax(outputs, dim =1 )
                predictions = torch.argmax(probabilities, dim=1)

                total_loss +=loss.item()
                all_predictions.extend(predictions.cpu().numpu())
                all_labels.extend(labels.cpu().numpy)
                all_probabilities.extend(probabilities[:,1].cpu().numpy())
        # Calculate metrics
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='binary', zero_division=0
        )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': np.array(all_predictions),
            'labels': np.array(all_labels),
            'probabilities': np.array(all_probabilities)
        }

    def train(self, train_loader, val_loader, num_epochs, class_weights=None, 
              save_path='outputs/models/best_model.pth'):
        print("=" * 80)
        print("Starting Training")
        print("=" * 80)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 80)
            
            # Training
            start_time = time.time()
            train_loss = self.train_epoch(train_loader, class_weights)
            train_time = time.time() - start_time
            
            # Validation
            start_time = time.time()
            val_metrics = self.evaluate(val_loader)
            val_time = time.time() - start_time
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_metrics['loss'])
            self.val_accuracies.append(val_metrics['accuracy'])
            self.val_f1_scores.append(val_metrics['f1'])
            
            # Print metrics
            print(f"\nTrain Loss: {train_loss:.4f} (Time: {train_time:.2f}s)")
            print(f"Val Loss:   {val_metrics['loss']:.4f} (Time: {val_time:.2f}s)")
            print(f"Val Acc:    {val_metrics['accuracy']:.4f}")
            print(f"Val Prec:   {val_metrics['precision']:.4f}")
            print(f"Val Recall: {val_metrics['recall']:.4f}")
            print(f"Val F1:     {val_metrics['f1']:.4f}")
            
            # Save best model based on F1 score
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.best_val_loss = val_metrics['loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_f1': val_metrics['f1'],
                    'val_loss': val_metrics['loss'],
                }, save_path)
                print(f"✓ Saved best model (F1: {val_metrics['f1']:.4f})")
        
        print("\n" + "=" * 80)
        print("Training Complete!")
        print(f"Best Val F1: {self.best_val_f1:.4f}")
        print("=" * 80)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'val_f1_scores': self.val_f1_scores
        }
    
    def load_best_model(self, save_path='outputs/models/best_model.pth'):
        """Load the best saved model"""
        checkpoint = torch.load(save_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded best model from {save_path}")
        print(f"  Best Val F1: {checkpoint['val_f1']:.4f}")
        return checkpoint

if __name__ == "__main__":
    print("Trainer module loaded successfully!")
    print("Use this module by importing: from models.trainer import JITDefectTrainer")
