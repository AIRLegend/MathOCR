import re
from nltk import regexp_tokenize


class Vocabulary:
    def __init__(self, vocab):
        self.token_idx = {}
        self.idx_token = {}
        self.vocab = vocab
        self._init_vocab_dicts(vocab)
    
    def _init_vocab_dicts(self, vocab):
        for i, tok in enumerate(vocab):
            self.token_idx[i] = tok
            self.idx_token[tok] = i
            
    def __getitem__(self, item):
        if item is " ": item = 'SPACE'
        retval = None
        try:
            if type(item) == str:
                retval = self.idx_token[item]
            elif type(item) == int:
                retval = self.token_idx[item]
        except KeyError:
            if type(item) != str:
                retval = 'UNK'
            else:
                retval = self.idx_token['UNK']
        return retval
    
    def __len__(self):
        return len(self.token_idx)


class LatexTokenizer:
    
    def __init__(self):
        self._REGEX_CLEAN = "(\\\\t(?![a-z])|\\\\rule|\\\\null|\\\\hfill|\\\\ref\{.{2,8}\}"+\
        "|\\\\label{\w+}|\\\\hspace\{.{2,7}\}|\\\\vspace\{.{2,7}\}|"+\
        "\\\\q+uad|\\\\label\{.{0,20}\})"
    
        self._REGEX_TOKEN = "(\\\\[a-zA-Z]+|[\s\^\{\}\(\)\_=0-9a-zA-Z\-\;\,\.\&\+\*\/\\\[\]<>])"
        
        # There are some malformed letters like "\F". This will remove the "\"
        self._REGEX_LETTERS = "(\\\\\\\\(?=[A-Z])|\\\\(?=[A-Z]))(?![A-Za-z]{3,})"  
    
    def tokenize(self, formula):
        clean_formula = re.sub(self._REGEX_CLEAN, '', formula)
        clean_formula = re.sub(self._REGEX_LETTERS, '', clean_formula)
        tokens = regexp_tokenize(clean_formula, self._REGEX_TOKEN)
        return tokens
    
    def encode(self, sequence, vocabulary):
        return  [vocabulary['START']] + [vocabulary[t] for t in sequence]  + [vocabulary['END']]
    
    def decode(self, sequence, vocabulary):
        return [vocabulary[t]  for t in sequence]
