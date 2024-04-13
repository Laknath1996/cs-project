python3 finetune.py -m run="full-ft" freeze=False device="cuda:0" lr=2e-5 epochs=10
python3 finetune.py -m run="partial-ft" freeze=True device="cuda:1" lr=5e-4 epochs=60
python3 finetune.py -m run="lora-ft" device="cuda:1" lr=5e-4 epochs=60