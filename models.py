# models.py
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=4096):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, text_vocab_size, gloss_vocab_size, 
                embedding_dim=1024, nhead=8, 
                num_encoder_layers=2, num_decoder_layers=2, 
                dropout=0.1, max_len=100):
        super().__init__()
        
        # Embedding layers
        self.text_embedding = nn.Embedding(text_vocab_size, embedding_dim)
        self.gloss_embedding = nn.Embedding(gloss_vocab_size, embedding_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout, max_len)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layer
        self.fc_out = nn.Linear(embedding_dim, gloss_vocab_size)
        
    def generate_square_subsequent_mask(self, sz):
        """Generate a mask for the decoder to prevent looking at future tokens"""
        mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
    def forward(self, src, tgt):
        # Create padding masks
        src_padding_mask = (src == 0)  # Assuming 0 is the <pad> index
        tgt_padding_mask = (tgt == 0)
        
        # Embed and add positional encoding
        src_embedded = self.positional_encoding(self.text_embedding(src))
        tgt_embedded = self.positional_encoding(self.gloss_embedding(tgt))
        
        # Generate target mask
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(src.device)
        
        # Transformer forward pass
        output = self.transformer(
            src_embedded,
            tgt_embedded,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )
        
        return self.fc_out(output)