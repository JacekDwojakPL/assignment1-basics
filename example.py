import os
from pathlib import Path
from cs336_basics.BPE import BPETrainer

MODE = 'valid'
datapath = Path(os.getcwd() + "/data")
outputpath = Path(os.getcwd() + "/outputs")
filename = f"example.txt"
vocab_size = 10000
input_path = datapath / filename
trainer = BPETrainer(vocab_size, ["<|endoftext|>"])
trainer.train(datapath / filename)
trainer.serialize(outputpath / "tiny_stories_vocab.json", outputpath / "tiny_stories_merges.txt")