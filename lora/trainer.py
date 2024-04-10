import torch
from torch.utils.data import DataLoader

from transformers import RobertaTokenizer
from datasets import load_dataset

import numpy as np
from tqdm import tqdm

from lora.models import get_model
from lora.utils import init_torch_seeds

class Trainer:
    def __init__(self, args) -> None:
        self.args = args

        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        dataset = load_dataset("nyu-mll/glue", "sst2")
        self.dataset = dataset.map(lambda e : self.tokenizer(e['sentence'], padding='max_length'), batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        self.trainloader = DataLoader(
            self.dataset['train'], 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.workers
            )
        self.testloader = DataLoader(
            self.dataset['validation'], 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.workers
            )

        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        self.model = get_model(args).to(self.device)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
    def finetune(self):
        args = self.args
        device = self.device

        # Set random initialization seed, easy to reproduce.
        init_torch_seeds(args.seed)

        self.model.train()

        for epoch in range(args.epochs):
            train_loss = 0.0
            num_correct_preds = 0
            num_train_samples = 0
            progress = tqdm(enumerate(self.trainloader), total=len(self.trainloader))
            for i, data in progress:
                ids = data['input_ids'].long().to(device)
                masks = data['attention_mask'].long().to(device)
                targets = data['label'].long().to(device)
                batchsize = len(targets)

                outputs = self.model(ids, masks)
                loss = self.loss_function(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                num_train_samples += batchsize
                train_loss += loss.item() * batchsize
                num_correct_preds += (torch.argmax(outputs, axis=1) == targets).detach().cpu().numpy().sum()

                progress.set_description(
                    f"[{epoch + 1}/{args.epochs}][{i + 1}/{len(self.trainloader)}] "
                    f"loss: {loss.item():.4f}"
                )
            
            info = {
                "epoch" : epoch + 1,
                "train_loss" : np.round(train_loss/num_train_samples, 4),
                "train_acc" : np.round(num_correct_preds/num_train_samples, 4)
            }
            print(info)

    def evaluate(self):
        device = self.device

        self.model.eval()

        num_correct_preds = 0
        num_test_samples = 0
        progress = tqdm(enumerate(self.testloader), total=len(self.testloader))
        with torch.no_grad():
            for i, data in progress:
                ids = data['input_ids'].long().to(device)
                masks = data['attention_mask'].long().to(device)
                targets = data['label'].long().to(device)
                batchsize = len(targets)

                outputs = self.model(ids, masks)

                num_test_samples += batchsize
                num_correct_preds += (torch.argmax(outputs, axis=1) == targets).detach().cpu().numpy().sum()

                progress.set_description(
                    f"[{i + 1}/{len(self.testloader)}] "
                )

        info = {
            "test_acc" : np.round(num_correct_preds/num_test_samples, 4)
        }
        print(info)




