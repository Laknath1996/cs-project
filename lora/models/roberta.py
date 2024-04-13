import torch
import torch.nn as nn
from transformers import RobertaModel
from lora.layers.vanilla import Linear

class RoBERTaClassifier(torch.nn.Module):
    def __init__(self, args):
        super(RoBERTaClassifier, self).__init__()

        # get the pretrained roberta model
        self.encoder = RobertaModel.from_pretrained("roberta-base")
            
        # prepare the model
        if args.run == "full-ft":
            for p in self.encoder.parameters():
                p.requires_grad = True
        elif args.run == "partial-ft":
            for p in self.encoder.parameters():
                p.requires_grad = False
        elif args.run == "lora-ft":
            self.encoder = inject_lora_matrices(self.encoder, args)
            
        self.classifier = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(768, 768),
            nn.Tanh(),
            nn.Dropout(args.dropout),
            nn.Linear(768, 2),
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        x = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        x = x[0][:, 0]
        x = self.classifier(x)
        return x

def inject_lora_matrices(model, args):
    # freeze the entire model
    for p in model.parameters():
        p.requires_grad = False
    
    # inject lora matrices to Wq and Wv
    for i in range(model.config.num_hidden_layers):
        old = model.encoder.layer[i].attention.self.query
        model.encoder.layer[i].attention.self.query = Linear(768, 768, r=args.lora_r, lora_alpha=args.lora_alpha)
        model.encoder.layer[i].attention.self.query.weight = old.weight
        model.encoder.layer[i].attention.self.query.bias = old.bias

        old = model.encoder.layer[i].attention.self.value
        model.encoder.layer[i].attention.self.value = Linear(768, 768, r=args.lora_r, lora_alpha=args.lora_alpha)
        model.encoder.layer[i].attention.self.value.weight = old.weight
        model.encoder.layer[i].attention.self.value.bias = old.bias
    
    return model

def get_model(args):
    model = RoBERTaClassifier(args)
    return model
