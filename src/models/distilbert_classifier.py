import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig

class DistilBertClassifire(nn.Module):
    #distilbertModel for binary classification
    def __init__(self, n_classes= 2, dropout= 0.3, pretrained_model= 'distilbert-base-uncased'):
        super(DistilBertClassifire, self).__init__()
        #load pretrained distilbert
        self.distilbert = DistilBertModel.from_pretrained(pretrained_model)
        # Uncomment to freeze:
        # for param in self.distilbert.parameters():
        #     param.requires_grad = False
        self.hidden_size = self.distilbert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifire = nn.Linear(self.hidden_size, n_classes)
        self._init_weights(self.classifire)
        
    def _init_weights(self, module):
        """Initialize weights for the classifier layer"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.0)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, input_ids, attention_mask):
        output = self.distilbert(input_ids = input_ids, attention_mask = attention_mask)
        pooled_output = output.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifire(pooled_output)
        return logits

    def predict_proba(self, input_ids, attention_mask):
        logits = self.forward(input_ids,attention_mask)
        probabbilities= torch.softmax(logits, dim=1)
        return probabbilities

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

#test the model
if __name__ == "__main__" :
    print("=" * 80)
    print("Testing DistilBERT Classifier")
    print("=" * 80)

    model = DistilBertClassifire(n_classes=2, dropout=3)
    print(f"\n total parameters: {model.get_num_parameters():,}")
    print(f"\n trainable parameters :{model.get_num_trainable_parameters():,}")
    
    batch_size = 4
    seq_length = 128

    dummy_input_ids = torch.randint(0,30522, (batch_size, seq_length))
    dummy_attention_mask = torch.ones(batch_size, seq_length)

    print(f"\n input shape: {dummy_input_ids.shape}")

    model.eval()
    with torch.no_grad():
        logits = model(dummy_input_ids, dummy_attention_mask)
        probs = model.predict_proba(dummy_input_ids, dummy_attention_mask)

    print(f"Output logits shape: {logits.shape}")
    print(f"Output probabilities shape: {probs.shape}")
    print(f"\nSample probabilities (first 2 examples):")
    print(probs[:2])
    print("\n✓ Model test successful!")