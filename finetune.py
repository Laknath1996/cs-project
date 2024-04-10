from omegaconf import OmegaConf
from lora.trainer import Trainer

args = OmegaConf.load('config.yaml')
trainer = Trainer(args)
trainer.finetune()
trainer.evaluate()