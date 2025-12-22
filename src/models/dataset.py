import torch
from torch.utils.data import dataset, dataloader
import pandas as pd
from transformers import DistilBertTokenizer

class commitDataset(dataset):
    def __init__(self, csv_files=None, dataframe=None, tokenizer_name='distilbert-base-uncased',
                 max_length= 128, message_column='clean_message', label_column='label'):
        if dataframe is not None:
            self.data = dataframe
        elif csv_files is not None:
            self.data = pd.read_csv(csv_files)
        else:
            raise ValueError("Either csv_file or dataframe must be provided")
        
        #initialize tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.message_column = message_column
        self.label_column = label_column
        
        #extract message and labeles
        self.messages = self.data[message_column].values
        self.labels = self.data[label_column].values
        print(f"Loaded dataset with {len(self.messages)} examples")

    def __len__(self):
        #return the length of dataset
        return len(self.messages)
    
    def __getitem__(self, idx):
        message = str(self.messages[idx])
        label = self.labels[idx]
        #tokenize
        encoding=self.tokenizer(
            message,
            add_special_token = True,
            max_length = self.max_length,
            padding = 'max_length',
            truncation = True,
            return_attention_mask = True,
            return_turnsor = 'pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
    
    def get_labels(self):
        return self.labels
    
    def get_class_count(self):
        """Return counts of each class"""
        unique, counts =torch.unique(torch.tensor(self.labels), return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
    
def create_date_loader(train_csv, val_csv, test_csv, batch_size = 16, max_length = 128):
    print("=" * 80)
    print("Creating Data Loaders")
    print("=" * 80)

    #create dataset
    print("\n1 loading training data...")
    train_dataset = commitDataset(csv_files=train_csv, max_length=max_length)
    print (f"class distribution :{train_dataset.get_class_count()}")
    
    print("\n2. loading validation data..")
    val_dataset = commitDataset(csv_files=val_csv, max_length=max_length)
    print(f"class distribution : {val_dataset.get_class_count()}")

    print("\n3. loading test data...")
    test_dataset = commitDataset(csv_files=test_csv, max_length=max_length)
    print(f"class distribution :{test_dataset.get_class_count()}")

    train_loader = dataloader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=0)
    val_loader = dataloader(val_dataset, batch_size = batch_size , shuffle = False, num_workers=0)
    test_loader = dataloader(test_dataset, batch_size = batch_size , shuffle = False, num_workers=0)

    print(f"\n✓ Created data loaders:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val:   {len(val_loader)} batches")
    print(f"  Test:  {len(test_loader)} batches")
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    print("=" * 80)
    print("Testing CommitDataset")
    print("=" * 80)
    import pandas as pd

    test_data=pd.DataFrame({
            'clean_message': [
            'fix authentication bug in login module',
            'add new feature for user dashboard',
            'update documentation for api endpoints'
        ],
        'label': [1, 0, 0]
    })
    dataset = commitDataset(dataframe=test_data, max_length=128)
    print("n1\. first item in dataset:")
    item = dataset[0]
    print(f"input IDs shape: {item['input_ids'].shape}")
    print(f"attention mask shape: {item['attention-mask'].shape}")
    print(f"Label: {item['label']}")

    # Decode to see tokens
    print("n2\. decode tokens:")
    tokens =dataset.tokenizer.convert_ids_to_tokens(item['input_ids'])
    print(f"Tokens: {tokens[:20]}...") #first 20 tokens

    # Create data loader
    print("\n3. Testing DataLoader:")
    loader = dataloader(dataset, batch_size=2, shuffel=True)

    batch = next(iter(loader))
    print(f"   Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"   Batch attention_mask shape: {batch['attention_mask'].shape}")
    print(f"   Batch labels shape: {batch['label'].shape}")
    print("\n✓ Dataset test successful!")
