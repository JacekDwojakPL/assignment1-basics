import multiprocessing as mp
from collections import Counter
import os
import regex as re
from pathlib import Path
from .utils import get_word_counts


def train_bpe(input_path, vocab_size, special_tokens):
    vocab = {i:bytes(chr(i), "utf-8") for i in range(256)}
    merges = []
    word_counts, pair_counts, pair_to_word, words_to_pair = get_word_counts(input_path, vocab_size, 24, "<|endoftext|>".encode())
    for i in range(vocab_size):
        new_pair = None
        # get most frequent pair
        most_frequent_pair = sorted(pair_counts.most_common(), key=lambda x: (x[1], x[0]))[-1]
        if not most_frequent_pair:
            continue
    
        pair, pair_count = most_frequent_pair
        # get words with that pair
        pair_words = pair_to_word[pair]
        # make new merge and vocab index
        merges.append((bytes(f"{pair[0]}", "utf-8"),  bytes(f"{pair[1]}", "utf-8")))
        vocab[256 + i] = bytes(f"{pair[0]}{pair[1]}", "utf-8")
        for word in pair_words:
            split = list(filter(lambda s: len(s), re.split(re.escape(f"{pair[0]}{pair[1]}"), word)))
            if not len(split):
                continue
            split = split[0]
            new_pair = (split[-1], f"{pair[0]}{pair[1]}")
            word_count = word_counts[word]
            pair_counts[new_pair] += word_count
            pairs = words_to_pair[word]
            
            for p in pairs:
                if f"{p[0]}{p[1]}" in f"{new_pair[0]}{new_pair[1]}":
                    del pair_counts[p]
            words_to_pair[word].add(new_pair)
            if pair in words_to_pair[word]:
                words_to_pair[word].remove(pair)
        del pair_counts[pair]
        del pair_to_word[pair]
        
        pair_to_word[new_pair] = pair_words

    return vocab, merges
