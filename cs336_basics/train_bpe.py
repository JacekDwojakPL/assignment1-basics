from .utils import get_word_counts, split_pretoken_to_pairs, perform_merge
from tqdm import tqdm
def train_bpe(input_path, vocab_size, special_tokens):
    vocab = {i:token.encode() for (i, token) in enumerate(special_tokens)}
    vocab.update({i + len(special_tokens): bytes([i]) for i in range(256)})

    merges = []
    words_counter, pairs_counter, pairs_to_word, word_to_pairs = get_word_counts(input_path, 1, special_tokens)
    if vocab_size > len(vocab):
        vocab_size = vocab_size - len(vocab)

    for i in tqdm(range(vocab_size)):
        # get most frequent pair
        if not pairs_counter:
            break
        pair = max(pairs_counter.items(), key=lambda p: (p[1], p[0]))[0]
        del pairs_counter[pair]
        # get words with that pair
        affected_words = pairs_to_word[pair].copy()
        # make new merge and vocab index
        new_idx = 256 + len(special_tokens) + i
        vocab[new_idx] = pair[0] + pair[1]
        merges.append((pair[0], pair[1]))

        for word in affected_words:
            word_count = words_counter[word]
            old_pairs = word_to_pairs[word]
            new_word = perform_merge(word, pair)
            new_pairs = [p for p in split_pretoken_to_pairs(new_word)]

            for p_remove in old_pairs:
                pairs_counter[p_remove] -= word_count
                if pairs_counter[p_remove] <= 0:
                    del pairs_counter[p_remove]
                pairs_to_word[p_remove].discard(word)

            for p_add in new_pairs:
                pairs_counter[p_add] += word_count
                pairs_to_word[p_add].add(new_word)
            
            del word_to_pairs[word]
            word_to_pairs[new_word] = new_pairs
            words_counter[new_word] = word_count
            del words_counter[word]

    return vocab, merges
