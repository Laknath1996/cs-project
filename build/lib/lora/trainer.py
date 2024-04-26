import math
import torch
from torch.utils.data import DataLoader

from transformers import RobertaTokenizer
from datasets import load_dataset

import numpy as np
from tqdm import tqdm

# from lora.models.roberta_pretrained import get_model
from lora.models.roberta_scratch import get_model
from lora.utils import init_torch_seeds
from lora.utils import linear_schedule_with_warmup

class Trainer:
    def __init__(self, args, log) -> None:
        self.args = args
        self.log = log

        # get tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        # tokenize the entire dataset
        dataset = load_dataset("nyu-mll/glue", "sst2")
        self.dataset = dataset.map(
            lambda e: self.tokenizer(
                e["sentence"],
                None,
                max_length=256,
                pad_to_max_length=True,
                return_token_type_ids=True,
            ),
            batched=True,
        )
        self.dataset.set_format(
            type="torch",
            columns=["input_ids", "token_type_ids", "attention_mask", "label"],
        )

        # define dataloaders
        self.trainloader = DataLoader(
            self.dataset["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
        )
        self.testloader = DataLoader(
            self.dataset["validation"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
        )

        # device
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        # define training objects
        self.model = get_model(args).to(self.device)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        num_training_steps = len(self.trainloader) * args.epochs
        self.scheduler = linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=math.ceil(num_training_steps * args.warmup_ratio),
            num_training_steps=num_training_steps,
        )

    def finetune(self):
        args = self.args
        device = self.device

        # Set random initialization seed, easy to reproduce.
        init_torch_seeds(args.seed)

        iter = 0
        max_val_acc = 0.0
        for epoch in range(args.epochs):
            # run an epoch
            self.model.train()
            train_loss = 0.0
            num_correct_preds = 0
            num_train_samples = 0
            progress = tqdm(enumerate(self.trainloader), total=len(self.trainloader))
            for i, data in progress:
                ids = data["input_ids"].long().to(device)
                token_type_ids = data["token_type_ids"].long().to(device)
                masks = data["attention_mask"].long().to(device)
                targets = data["label"].long().to(device)
                batchsize = len(targets)

                outputs = self.model(ids, masks, token_type_ids)
                loss = self.loss_function(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                num_train_samples += batchsize
                train_loss += loss.item() * batchsize
                num_correct_preds += (
                    (torch.argmax(outputs, axis=1) == targets)
                    .detach()
                    .cpu()
                    .numpy()
                    .sum()
                )

                progress.set_description(
                    f"[{epoch + 1}/{args.epochs}][{i + 1}/{len(self.trainloader)}] "
                    f"loss: {loss.item():.4f}"
                )

                iter += 1
                self.scheduler.step(iter)

            # validate
            val_acc = self.evaluate()

            info = {
                "epoch": epoch + 1,
                "train_loss": np.round(train_loss / num_train_samples, 4),
                "train_acc": np.round(num_correct_preds / num_train_samples, 4),
                "val_acc": np.round(val_acc, 4),
            }
            self.log.info(f"{info}")

            # save the best model
            if val_acc > max_val_acc:
                self.log.info(f"saving model @ val_acc = {val_acc:.4f}")
                max_val_acc = val_acc
                if args.run == "partial-ft":
                    torch.save(self.model.classifier.state_dict(), f"{args.run}.pth")
                else:
                    torch.save(self.model.state_dict(), f"{args.run}.pth")

    def evaluate(self):
        device = self.device

        self.model.eval()

        num_correct_preds = 0
        num_test_samples = 0
        progress = tqdm(enumerate(self.testloader), total=len(self.testloader))
        with torch.no_grad():
            for i, data in progress:
                ids = data["input_ids"].long().to(device)
                token_type_ids = data["token_type_ids"].long().to(device)
                masks = data["attention_mask"].long().to(device)
                targets = data["label"].long().to(device)
                batchsize = len(targets)

                outputs = self.model(ids, masks, token_type_ids)

                num_test_samples += batchsize
                num_correct_preds += (
                    (torch.argmax(outputs, axis=1) == targets)
                    .detach()
                    .cpu()
                    .numpy()
                    .sum()
                )

                progress.set_description(f"[{i + 1}/{len(self.testloader)}] ")

        return num_correct_preds / num_test_samples


# def main():
#     import logging
#     import hydra

#     log = logging.getLogger(__name__)

#     @hydra.main(config_name='config.yaml')
#     def main(args):
#         log.info(f"{args}")
#         trainer = Trainer(args, log)
#         trainer.finetune()
#         trainer.evaluate()


# if __name__ == "__main__":
#     main()