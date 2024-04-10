import torch
import torch.nn as nn
from transformers import RobertaModel

class RoBERTaClassifier(torch.nn.Module):
    def __init__(self, args):
        super(RoBERTaClassifier, self).__init__()
        self.encoder = RobertaModel.from_pretrained("roberta-base")
        if args.freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 2)
        )

    def forward(self, input_ids, attention_mask):
        x = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        x = x[0][:, 0]
        x = self.classifier(x)
        return x
    
def get_model(args):
    model = RoBERTaClassifier(args)
    return model