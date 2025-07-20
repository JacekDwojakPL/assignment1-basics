import multiprocessing as mp
from collections import Counter, defaultdict
import regex as re
from .pretokenization_example import find_chunk_boundaries

SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# SPLIT_PATTERN = r"\w+"
words_counter = Counter()
pairs_counter = Counter()
pair_to_words = defaultdict(set)
words_to_pair = defaultdict(set)

def split_pretoken_to_pairs(pretoken):
    for first_letter, second_letter in zip(pretoken[:-1], pretoken[1:]):
        yield (first_letter, second_letter)

def update_counters(pretokens):
    words_counter.update(pretokens)
    for pretoken in pretokens:
        for pair in split_pretoken_to_pairs(pretoken):
            pairs_counter.update([pair])
            pair_to_words[pair].add(pretoken)
            words_to_pair[pretoken].add(pair)

def read_and_count_chunk(input_path, start_chunk, end_chunk, split_special_token):
    result = []
    with open(input_path) as f:
        f.seek(start_chunk)
        text = f.read(end_chunk - start_chunk)
        for doc in re.splititer(re.escape(split_special_token.decode("utf-8", errors="ignore")), text):
            matches = re.finditer(SPLIT_PATTERN, doc)
            result.extend([m.group() for m in matches])
    
    return result

def get_word_counts(input_path, num_chunks, special_token=b"<|endoftext|>"):
    f = open(input_path, "rb")
    chunks = find_chunk_boundaries(f, num_chunks, split_special_token=special_token)
    f.close()
    pool = mp.Pool(mp.cpu_count())
    for start_chunk, end_chunk in zip(chunks[:-1], chunks[1:]):
        pool.apply_async(read_and_count_chunk, (input_path, start_chunk, end_chunk, special_token), callback=update_counters)
    pool.close()
    pool.join()
    
    return (words_counter, pairs_counter, pair_to_words, words_to_pair)