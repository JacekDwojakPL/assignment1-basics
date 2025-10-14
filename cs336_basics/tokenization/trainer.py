from .utils import get_word_counts, split_pretoken_to_pairs, perform_merge
from tqdm import tqdm
import json

class BPETrainer:
    def __init__(self, vocab_size:int=0, special_tokens:list[str]=[]):
        self.vocab_size =  vocab_size
        self.special_tokens = special_tokens
        self.vocab = {}
        self.merges = []
    
    def train(self, input_path:str) -> tuple[dict, list[bytes]]:
        self.vocab = {i:token.encode() for (i, token) in enumerate(self.special_tokens)}
        self.vocab.update({i + len(self.special_tokens): bytes([i]) for i in range(256)})

        words_counter, pairs_counter, pairs_to_word, word_to_pairs = get_word_counts(input_path, 64, self.special_tokens)
        if self.vocab_size > len(self.vocab):
            self.vocab_size = self.vocab_size - len(self.vocab)

        for i in tqdm(range(self.vocab_size)):
            # get most frequent pair
            if not pairs_counter:
                break
            pair = max(pairs_counter.items(), key=lambda p: (p[1], p[0]))[0]
            del pairs_counter[pair]
            # get words with that pair
            affected_words = pairs_to_word[pair].copy()
            # make new merge and vocab index
            new_idx = 256 + len(self.special_tokens) + i
            self.vocab[new_idx] = pair[0] + pair[1]
            self.merges.append((pair[0], pair[1]))

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

        return self.vocab, self.merges
    
    def serialize(self, vocab_output_path:str="", merges_output_path:str=""):
        assert len(self.vocab), "no vocabulary to serialize. need to train first"
        assert len(self.merges), "no merges to serialize. need to train first"
        assert vocab_output_path, "add vocab output path with filename"
        assert merges_output_path, "add merges output path with filename"
        
        with open(vocab_output_path, "w") as f:
            f.write(json.dumps(self.vocab, default=repr))
            f.close()
        
        with open(merges_output_path, "w") as f:
            for m in self.merges:
                f.write(f"{m[0]} {m[1]}\n")
            f.close()
        
            
        
