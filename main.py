from position_encoder import EmbeddingsWithPositionalEncoding
from feed_forward import FeedForward
from model import Transformer, Encoder, Decoder, TransformerLayer
from mha import MultiHeadAttention
import torch

src = torch.zeros((2, 10), dtype=torch.long)
tgt = torch.zeros((2, 10), dtype=torch.long)
src_mask = torch.ones((2, 10, 10), dtype=torch.long)
tgt_mask = torch.zeros((2, 10, 10), dtype=torch.long)

encoder_mha = MultiHeadAttention(d_model=15, heads=3, d_k=10, bias=True)
encoder_ffn = FeedForward(d_model=15, d_ff=20)
decoder_mha = MultiHeadAttention(d_model=15, heads=3, d_k=10, bias=True)
decoder_ffn = FeedForward(d_model=15, d_ff=20)
decoder_src_mha = MultiHeadAttention(d_model=15, heads=3, d_k=10, bias=True)

encoder_trans_layer = TransformerLayer(d_model=15, self_attn=encoder_mha, feed_forward=encoder_ffn, dropout_prob=0.2)
decoder_trans_layer = TransformerLayer(d_model=15, self_attn=decoder_mha, feed_forward=decoder_ffn, dropout_prob=0.2,
                                       src_attn=decoder_src_mha)
encoder_position = EmbeddingsWithPositionalEncoding(d_model=15, n_vocab=100, max_len=10)
decoder_position = EmbeddingsWithPositionalEncoding(d_model=15, n_vocab=100, max_len=10)

encoder = Encoder(layer=encoder_trans_layer, n_layers=2)
decoder = Decoder(layer=decoder_trans_layer, n_layers=2)
model = Transformer(encoder=encoder, decoder=decoder, src_embed=encoder_position, tgt_embed=decoder_position,
                    generator=None)

result = model.forward(src=src, tgt=tgt, src_mask=src_mask, tgt_mask=tgt_mask)
