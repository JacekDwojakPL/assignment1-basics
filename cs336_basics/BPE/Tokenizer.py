import regex as re
import json
from .utils import SPLIT_PATTERN, perform_merge

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=[]):
        self.vocab = vocab
        self.inverse_vocab = {v:k for k,v in self.vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens
        self.special_tokens_split_pattern = "|".join(["(" + re.escape(token) +")" for token in special_tokens]) if special_tokens else None
    
    def encode(self, text):
        ids = [] 
        
        if not len(text):
            return ids
            
        docs = re.split(self.special_tokens_split_pattern, text) if self.special_tokens_split_pattern else []
        pretokens = []
        if len(docs):
            for d in docs:
                if not d or not len(d):
                    continue
                if d in self.special_tokens:
                    pretokens.append(d)
                else:
                    pretokens.extend([m.encode() for m in re.findall(SPLIT_PATTERN, d)])
        else:
            pretokens.extend([m.encode() for m in re.findall(SPLIT_PATTERN, text)])
        
        for pretoken in pretokens:
            
            if self.special_tokens and pretoken in self.special_tokens:
                ids.append(pretoken.encode())
                continue

            pretoken_copy = [bytes([b]) for b in pretoken]
            merged = False
            
            while not merged:
                merged = True
                new_pretoken = []
                
                merge_candidates = list(filter(lambda p: p[0] in pretoken_copy and p[1] in pretoken_copy and p[0]+p[1] in pretoken, self.merges))
                
                for i in range(len(pretoken_copy)):
                    if i < len(pretoken_copy)-1:
                        pair_to_merge = merge_candidates.pop(0) if len(merge_candidates) else None
                        if(pair_to_merge):
                            new_pretoken = perform_merge(pretoken_copy, pair_to_merge)
                            if new_pretoken == pretoken_copy:
                                continue
                            merged = False
                            break
                        else:
                            new_pretoken.append(pretoken_copy[i])
                    else:
                        new_pretoken.append(pretoken_copy[i])
                pretoken_copy = new_pretoken
            ids.extend(pretoken_copy)

        ret = [self.inverse_vocab[id] for id in ids]

        return ret
    
    def encode_iterable(self, pointer):
        return []
    
    def decode(self, ids):
        if not len(ids):
            return ''
        out = []
        for id in ids:
            try:
                out.append(self.vocab[id])
            except KeyError:
                out.append(id)

        return b"".join(out).decode("utf-8", "replace")

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=[]):
        vocab = {}
        merges = []
        pattern = re.compile(r"b(['\"])(.*?)\1\s+b(['\"])(.*?)\3")
        
        with open(vocab_filepath, "r") as vf:
            raw_vocab = json.load(vf)
            for k, v in raw_vocab.items():
                vocab[int(eval(k))] = eval(v)
            vf.close()
        
        with open(merges_filepath, 'r') as mf:
            for line in mf.readlines():
                match = pattern.search(line.rstrip())
                merges.append((match.group(2).encode(), match.group(4).encode()))
            mf.close()

        return cls(vocab, merges, special_tokens)