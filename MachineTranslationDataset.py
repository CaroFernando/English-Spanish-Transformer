from transformers import AutoTokenizer

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd


class TextTranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, max_len=512):
        self.src_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.tgt_tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")

        # set eos token and bos token
        self.src_tokenizer.bos_token = self.src_tokenizer.cls_token
        self.src_tokenizer.eos_token = self.src_tokenizer.sep_token
        self.tgt_tokenizer.bos_token = self.tgt_tokenizer.cls_token
        self.tgt_tokenizer.eos_token = self.tgt_tokenizer.sep_token

        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.max_len = max_len

        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.max_len = max_len
    
    def src_vocab_size(self):
        return len(self.src_tokenizer)
    
    def tgt_vocab_size(self):
        return len(self.tgt_tokenizer)
    
    def tgt_tokens_to_string(self, tokens):
        return self.tgt_tokenizer.decode(tokens)

    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, item):
        src_sentence = str(self.src_sentences[item])
        tgt_sentence = str(self.tgt_sentences[item])

        src_tokens = self.src_tokenizer.encode(src_sentence, add_special_tokens=True)
        tgt_tokens = self.tgt_tokenizer.encode(tgt_sentence, add_special_tokens=True)

        # handle too long sentences
        if len(src_tokens) > self.max_len:
            src_tokens = src_tokens[:self.max_len]
        if len(tgt_tokens) > self.max_len:
            tgt_tokens = tgt_tokens[:self.max_len]

        return torch.tensor(src_tokens), torch.tensor(tgt_tokens)
    
    def collate_fn(self, batch):
        src_tokens, tgt_tokens = zip(*batch)
        src_tokens = pad_sequence(src_tokens, batch_first=True, padding_value=self.src_tokenizer.pad_token_id)
        tgt_tokens = pad_sequence(tgt_tokens, batch_first=True, padding_value=self.tgt_tokenizer.pad_token_id)
        return src_tokens, tgt_tokens
