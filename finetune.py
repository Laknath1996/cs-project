import torch
import hydra
from lora.trainer import Trainer
import logging

log = logging.getLogger(__name__)

@hydra.main(config_name='config.yaml')
def main(args):
    log.info(f"{args}")
    trainer = Trainer(args, log)
    trainer.finetune()
    trainer.evaluate()
    
if __name__ == "__main__":
    main()