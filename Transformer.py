import torch
from torch import nn

class Attention(nn.Module):
    def __init__(self, hidden_size, heads=8):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        self.head_dim = hidden_size // heads
        self.fc = nn.Linear(hidden_size, hidden_size)
        assert (
            self.head_dim * heads == hidden_size
        ), "Hidden size must be divisible by heads"

    def forward(self, key, value, query, mask=None):
        N = query.shape[0]
        query_len, key_len, value_len = query.shape[1], key.shape[1], value.shape[1]
        query = torch.reshape(query, (N, query_len, self.heads, self.head_dim))
        key = torch.reshape(key, (N, key_len, self.heads, self.head_dim))
        value = torch.reshape(value, (N, value_len, self.heads, self.head_dim))
        matmul = torch.einsum("nqhd,nkhd->nhqk", [query, key])
        scale = matmul / (self.hidden_size ** (1 / 2))
        if mask is not None:
            scale = scale.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(scale, dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, value]).reshape(
            N, query_len, self.hidden_size
        )
        out = self.fc(out)
        return out


class Transformer_Block(nn.Module):
    def __init__(self, hidden_size, heads, dropout, forward_expansion):
        super(Transformer_Block, self).__init__()
        self.attention = Attention(hidden_size, heads)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, forward_expansion * hidden_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * hidden_size, hidden_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, key, value, query, mask=None):
        attention = self.attention(key, value, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            Transformer_Block(embed_size, heads, dropout, forward_expansion)
            for _ in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = Attention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer = Transformer_Block(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, tgt_mask):
        attention = self.attention(x, x, x, tgt_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer(key, value, query, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        tgt_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(tgt_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
            for _ in range(num_layers)
        )
        self.fc_out = nn.Linear(embed_size, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, tgt_mask)
        out = self.fc_out(x)
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        src_pad_idx,
        tgt_pad_idx,
        device="cpu",
        embed_size=512,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        max_length=100,
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )
        self.decoder = Decoder(
            tgt_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)
    def make_tgt_mask(self, tgt):
        N, tgt_len = tgt.shape
        tgt_mask = torch.tril(torch.ones((tgt_len, tgt_len))).expand(
            N, 1, tgt_len, tgt_len
        )
        return tgt_mask.to(self.device)
    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(tgt, enc_src, src_mask, tgt_mask)
        return out
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
        device
    )
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(
        device
    )
    out = model(x, trg[:, :-1])
    print(out.shape)