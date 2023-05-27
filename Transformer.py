import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

import pytorch_lightning as pl
import re
import pandas as pd

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
    
class Transformer(pl.LightningModule):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(self.device)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3).to(self.device)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        nopeak_mask = nopeak_mask.to(self.device)
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_mask = src_mask.to(self.device)
        tgt_mask = tgt_mask.to(self.device)

        emb_src = self.encoder_embedding(src)
        emb_tgt = self.decoder_embedding(tgt)

        emb_src = self.positional_encoding(emb_src)
        emb_tgt = self.positional_encoding(emb_tgt)

        src_embedded = self.dropout(emb_src)
        tgt_embedded = self.dropout(emb_tgt)

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output
    

class TransformerMachineTranslation(pl.LightningModule):
    def __init__(self, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, src_tokenizer, tgt_tokenizer):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_seq_length = max_seq_length
        
        self.model = Transformer(len(self.src_tokenizer), len(self.tgt_tokenizer), d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
        self.loss = nn.CrossEntropyLoss(ignore_index=self.tgt_tokenizer.pad_token_id, label_smoothing=0.1)

    def forward(self, src, tgt):
        return self.model(src, tgt)
        
    def valiate_translations(self):
        sentence1 = "I have been living in this city for a long time"
        translated_sentence1 = self.translate(sentence1)
        
        sentence2 = "i can't speak english very well"
        translated_sentence2 = self.translate(sentence2)

        sentence3 = "this is not working very well"
        translated_sentence3 = self.translate(sentence3)

        columns = ["English", "Translated"]
        data = [[sentence1, translated_sentence1], [sentence2, translated_sentence2], [sentence3, translated_sentence3]]
        df = pd.DataFrame(data, columns=columns)
        return df
    
    def training_step(self, batch, batch_idx):
        src, tgt = batch
        output = self(src, tgt[:,:-1])
        output = output.contiguous().view(-1, output.size(-1))
        tgt = tgt[:,1:].contiguous().view(-1)
        loss = self.loss(output, tgt)
        self.log("train_loss", loss, prog_bar=True)

        if self.global_step % 2000 == 0:
            df = self.valiate_translations()
            self.logger.experiment.add_text("Translations", df.to_markdown(), self.global_step)

        return loss
    
    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        output = self(src, tgt[:,:-1])
        output = output.contiguous().view(-1, output.size(-1))
        tgt = tgt[:,1:].contiguous().view(-1)
        loss = self.loss(output, tgt)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def on_validation_epoch_end(self):
        df = self.valiate_translations()
        self.logger.experiment.add_text("Translations", df.to_markdown(), self.current_epoch)

    def calc_lr(self, step, dim_embed, warmup_steps):
        return dim_embed**(-0.5) * min((step+1)**(-0.5), (step+1) * warmup_steps**(-1.5))

    def configure_optimizers(self):
        op = torch.optim.Adam(self.parameters(), lr = 0.1, betas=(0.9, 0.98))
        warmup_steps = 1000
        sch = torch.optim.lr_scheduler.LambdaLR(op, lambda step: self.calc_lr(step, self.d_model, warmup_steps))

        return {
            "optimizer": op, 
            "lr_scheduler": {
                "scheduler": sch,
                "interval": "step",
                "frequency": 1
            }         
        }
    
    def translate(self, sentence):
        self.eval()
        src = self.src_tokenizer.encode(sentence)
        src = torch.tensor(src).unsqueeze(0)
        src = src.to(self.device)
        tgt = torch.ones(1, self.max_seq_length).long().to(self.device)
        tgt[:,0] = self.tgt_tokenizer.bos_token_id
        for i in range(1, self.max_seq_length):
            output = self(src, tgt)
            output = output.argmax(dim=-1)
            tgt[:,i] = output[:,i]
            if output[:,i] == self.tgt_tokenizer.eos_token_id:
                break

        tgt = tgt[0].tolist()
        for i in range(len(tgt)):
            if tgt[i] == self.tgt_tokenizer.pad_token_id:
                tgt = tgt[:i]
                break
        tgt = self.tgt_tokenizer.decode(tgt)
        return tgt
    
