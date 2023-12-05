import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=150):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TokenEmbedding(nn.Module):
    def __init__(self, num_vocab, maxlen=150, embedding_dim=16, dropout_rate=0.1):
        super(TokenEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.emb = nn.Embedding(num_vocab, embedding_dim)
        self.pos_emb = PositionalEncoding(embedding_dim, dropout=dropout_rate, max_len=maxlen)
    def forward(self, inputs):
        x = self.emb(inputs)
        x = x * math.sqrt(self.embedding_dim) #!!!! 
        x = self.pos_emb(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, fully_connected_dim,
                 dropout_rate=0.1, layernorm_eps=1e-6):
        super(EncoderLayer, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout_rate, batch_first=True)
        """self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, fully_connected_dim),
            nn.ReLU(),
            nn.Linear(fully_connected_dim, embedding_dim)
            )"""

        self.ffn = nn.Sequential(
            nn.Conv1d(embedding_dim, fully_connected_dim, 3, padding='same'),
            nn.BatchNorm1d(fully_connected_dim),
            nn.ReLU(),
            nn.Conv1d(fully_connected_dim, embedding_dim, 3, padding='same'),
            nn.BatchNorm1d(embedding_dim)
            )
        self.norm1 = nn.LayerNorm(normalized_shape=embedding_dim, eps=layernorm_eps)
        self.norm2 = nn.LayerNorm(normalized_shape=embedding_dim, eps=layernorm_eps)
        self.dropout_ffn = nn.Dropout(dropout_rate)

    def forward(self, inputs, mask):

        self_mha_output, _ = self.mha(inputs, inputs, inputs, key_padding_mask = mask) # !!! mask

        skip_attention = self.norm1(inputs + self_mha_output)

        ffn_output = self.ffn(skip_attention.permute(0, 2, 1)).permute(0, 2, 1)

        ffn_output = self.dropout_ffn(ffn_output)

        encoder_layer_out = self.norm2(skip_attention + ffn_output)

        return encoder_layer_out

class Classifier(nn.Module):
    def __init__(self, num_class, num_layers, num_heads, fully_connected_dim, embedding_dim, max_len,
                 num_vocab, dropout_rate=0.1, layernorm_eps=1e-6):
        super(Classifier, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.fully_connected_dim = fully_connected_dim

        self.pos_encoding = TokenEmbedding(num_vocab, max_len, embedding_dim, dropout_rate)

        self.enc_layers = nn.ModuleList([EncoderLayer(embedding_dim=embedding_dim,
                                        num_heads=num_heads,
                                        fully_connected_dim=fully_connected_dim,
                                        dropout_rate=dropout_rate,
                                        layernorm_eps=layernorm_eps) for _ in range(self.num_layers)]
                                        )

        self.conv = nn.Conv1d(self.embedding_dim, 1, 1)

        self.linear = nn.Sequential(
            nn.Linear(max_len, max_len),
            nn.ReLU(),
            nn.Linear(max_len, num_class)
            )

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, inputs, mask):
        x = self.pos_encoding(inputs)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)

        x = self.conv(x.permute(0, 2, 1))
        out = self.linear(torch.squeeze(x))

        return out