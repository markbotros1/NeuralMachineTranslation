import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from typing import List, Tuple

class Encoder(nn.Module):
    def __init__(self, embed_size, hidden_size, source_embeddings):
        """
        """
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embedding = source_embeddings
        ### TODO - Initialize the following variables:
        ###     self.encoder (Bidirectional RNN with bias)
        ###     self.h_projection (Linear Layer with no bias), called W_{h} above.
        self.encoder = nn.LSTM(embed_size, hidden_size, bidirectional=True)
        self.h_projection = nn.Linear(hidden_size*2, hidden_size, bias=False)
        self.c_projection = nn.Linear(hidden_size*2, hidden_size, bias=False)
        
    def forward(self, source_padded: torch.Tensor, source_lengths: List[int]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        """
        enc_hiddens, dec_init_state = None, None
        ### TODO:
        ###     1. Construct Tensor `X` of source sentences with shape (src_len, b, e) using the source model embeddings.
        ###         src_len = maximum source sentence length, b = batch size, e = embedding size. Note
        ###         that there is no initial hidden state or cell for the encoder.
        ###     2. Compute `enc_hiddens`, `last_hidden` by applying the encoder to `X`.
        ###         - Before you can apply the encoder, you need to apply the `pack_padded_sequence` function to X.
        ###         - After you apply the encoder, you need to apply the `pad_packed_sequence` function to enc_hiddens.
        ###         - Note that the shape of the tensor returned by the encoder is (src_len, b, h*2) and we want to
        ###           return a tensor of shape (b, src_len, h*2) as `enc_hiddens`.
        ###     3. Compute `dec_init_state` = init_decoder_hidden:
        ###         - `init_decoder_hidden`:
        ###             `last_hidden` is a tensor shape (2, b, h). The first dimension corresponds to forward and backwards.
        ###             Concatenate the forward and backward tensors to obtain a tensor shape (b, 2*h).
        ###             Apply the h_projection layer to this in order to compute init_decoder_hidden.
        ###             This is h_0^{dec} in above in the writeup. Here b = batch size, h = hidden size
        X = self.embedding(source_padded)

        X_packed = pack_padded_sequence(X, source_lengths)
        enc_hiddens, (last_hidden, last_cell) = self.encoder(X_packed)
        enc_hiddens,_ = pad_packed_sequence(enc_hiddens)

        init_decoder_hidden = self.h_projection(torch.cat((last_hidden[0], last_hidden[1]), dim=1))
        init_decoder_cell = self.c_projection(torch.cat((last_cell[0], last_cell[1]), dim=1))
        dec_init_state = (init_decoder_hidden, init_decoder_cell)

        enc_hiddens = enc_hiddens.permute(1, 0, 2)

        return enc_hiddens, dec_init_state

  
def generate_sent_masks(enc_hiddens: torch.Tensor, source_lengths: List[int], device: torch.device) -> torch.Tensor:
    """ Generate sentence masks for encoder hidden states.

    :param enc_hiddens: encodings of shape (b, src_len, 2*h), where b = batch size,
        src_len = max source length, h = hidden size.
    :type enc_hiddens: torch.Tensor
    :param source_lengths: List of actual lengths for each of the sentences in the batch.   
    :type source_lengths: List[int]
    :param device: Device on which to load the tensor, ie. CPU or GPU
    :type device: torch.device
    :returns: Tensor of sentence masks of shape (b, src_len),
        where src_len = max source length, h = hidden size.
    :rtype: torch.Tensor
    """
    enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
    for e_id, src_len in enumerate(source_lengths):
        enc_masks[e_id, src_len:] = 1
    return enc_masks.to(device)