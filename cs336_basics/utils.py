import multiprocessing as mp
from collections import Counter, defaultdict
import regex as re
from .pretokenization_example import find_chunk_boundaries

SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
BYTE_MAP = {i: bytes([i]) for i in range(256)}

def init_counters():
    words_counter = Counter()
    pairs_counter = Counter()
    pairs_to_word = defaultdict(set)
    word_to_pairs = defaultdict(list)
    return (words_counter, pairs_counter, pairs_to_word, word_to_pairs)

def perform_merge(word, pair):
    out = []
    counter = 0
    while counter < len(word):
        if counter + 1 < len(word) and (word[counter], word[counter + 1]) == pair:
            out.append(word[counter]+word[counter+1])
            counter += 2
        else:
            out.append(word[counter])
            counter += 1
    return tuple(out)

def split_pretoken_to_pairs(pretoken):
    for first_letter, second_letter in zip(pretoken, pretoken[1:]):
        yield (first_letter, second_letter)

def update_counters(pretokens, counters):
    words_counter, pairs_counter, pairs_to_word, word_to_pairs = counters
    for pretoken in pretokens:
        words_counter.update([pretoken])
        pairs = []
        for pair in split_pretoken_to_pairs(pretoken):
            pairs_counter.update([pair])
            pairs_to_word[pair].add(pretoken)
            pairs.append(pair)
        word_to_pairs[pretoken] = pairs

def read_and_count_chunk(input_path, start_chunk, end_chunk, split_special_token):
    result = []
    with open(input_path) as f:
        f.seek(start_chunk)
        text = f.read(end_chunk - start_chunk)
        for doc in re.splititer(split_special_token, text):
            matches = re.finditer(SPLIT_PATTERN, doc)
            for m in matches:
                result.extend([tuple(BYTE_MAP[p] for p in m.group().encode())])
    return result

def get_word_counts(input_path, num_chunks, special_tokens=[]):
    counters = init_counters()
    split_special_token = "|".join([token for token in special_tokens])
    cb = lambda pretokens: update_counters(pretokens, counters)
    f = open(input_path, "rb")
    chunks = find_chunk_boundaries(f, num_chunks, split_special_token=split_special_token.encode("utf-8"))
    f.close()
    pool = mp.Pool(mp.cpu_count())
    for start_chunk, end_chunk in zip(chunks[:-1], chunks[1:]):
        pool.apply_async(read_and_count_chunk, (input_path, start_chunk, end_chunk, split_special_token), callback=cb)
    pool.close()
    pool.join()

    return counters