import torch
import hydra
from lora.trainer import Trainer

@hydra.main(config_name='config.yaml')
def main(args):
    print(args)
    trainer = Trainer(args)
    trainer.finetune()
    trainer.evaluate()

    # save weights
    model = trainer.model
    if args.freeze:
        torch.save(model.classifier.state_dict(), f"weights/{args.run}.pth")
    else:
        torch.save(model.state_dict(), f"weights/{args.run}.pth")

if __name__ == "__main__":
    main()