import multiprocessing as mp
from collections import Counter, defaultdict
import regex as re
from .pretokenization import find_chunk_boundaries
import logging

logging.basicConfig(
    level=logging.DEBUG,  # Set minimum level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s"
)

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
    counters = init_counters()
    with open(input_path, "rb") as f:
        f.seek(start_chunk)
        text = f.read(end_chunk - start_chunk).decode("utf-8", errors="ignore")
        for doc in re.splititer(split_special_token, text):
            matches = re.finditer(SPLIT_PATTERN, doc)
            for m in matches:
                result.extend([tuple(BYTE_MAP[p] for p in m.group().encode())])
    update_counters(result, counters)
    logging.info(f"end processing chunk {start_chunk} {end_chunk}")
    return counters

def get_word_counts(input_path, num_chunks, special_tokens=[]):
    words_counter, pairs_counter, pairs_to_word, word_to_pairs = init_counters()
    split_special_token = "|".join([token for token in special_tokens])
    f = open(input_path, "rb")
    chunks = find_chunk_boundaries(f, num_chunks, split_special_token=split_special_token.encode("utf-8"))
    chunk_pairs = [(chunks[i], chunks[i+1]) for i in range(len(chunks)-1)]
    f.close()
    
    for i in range(0, len(chunk_pairs), mp.cpu_count()):
        batch = chunk_pairs[i:i+mp.cpu_count()]
        
        with mp.Pool(mp.cpu_count()) as pool:
            result = []

            for start_chunk, end_chunk in batch:
                logging.info(f"start processing chunk {start_chunk} {end_chunk}")
                result.append(pool.apply_async(read_and_count_chunk, 
                                               (input_path, start_chunk, end_chunk, split_special_token)))
        
            for r in result:
                l_words_counter, l_pairs_counter, l_pairs_to_word, l_word_to_pairs = r.get()
                words_counter.update(l_words_counter)
                pairs_counter.update(l_pairs_counter)
                for pair, words in l_pairs_to_word.items():
                    pairs_to_word[pair].update(words)
                for word, pairs in l_word_to_pairs.items():
                    word_to_pairs[word] = pairs

    return (words_counter, pairs_counter, pairs_to_word, word_to_pairs)