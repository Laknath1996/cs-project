python finetune.py -m run="full-finetune" freeze=False device="cuda:0" batch_size=64
python finetune.py -m run="partial-finetune" freeze=True device="cuda:1"