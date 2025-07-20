import os
from pathlib import Path
from cs336_basics.train_bpe import train_bpe
from tests.common import FIXTURES_PATH

def run_training():
    MODE = 'valid'
    datapath = Path(os.getcwd() + "/data")
    # filename = f"TinyStoriesV2-GPT4-{MODE}.txt"
    filename = f"example.txt"
    input_path = datapath / filename
    vocab, merges = train_bpe(FIXTURES_PATH / 'corpus.en', 150, ["<|endoftext|>"])
    # print(vocab, merges)

if __name__ == "__main__":
    run_training()