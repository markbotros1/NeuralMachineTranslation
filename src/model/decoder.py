import torch
import torch.nn as nn
from typing import Tuple

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, target_embedding, device, d_rate=None):
        """
        """
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.device = device
        self.embedding = target_embedding
        output_vocab_size = self.embedding.weight.size(0)
        self.softmax = nn.Softmax(dim=1)
        self.d_rate = d_rate

        ### TODO:
        ###     self.decoder (RNN Cell with bias)
        ###     self.combined_output_projection (Linear Layer with no bias), called W_{v} above.
        ###     self.target_vocab_projection (Linear Layer with no bias), called W_{vocab} above.

        self.decoder = nn.LSTMCell(embed_size + hidden_size, hidden_size, bias=True)
        self.att_projection = nn.Linear(hidden_size*2, hidden_size, bias=False)
        self.combined_output_projection = nn.Linear(hidden_size*3, hidden_size, bias=False)
        self.target_vocab_projection = nn.Linear(hidden_size, output_vocab_size, bias=False)
        self.dropout = nn.Dropout(self.d_rate)
    
    def forward(self, enc_hiddens: torch.Tensor, enc_masks: torch.Tensor,
                dec_init_state: torch.Tensor, target_padded: torch.Tensor) -> torch.Tensor:
        """
        """
        # Chop off the <END> token for max length sentences.
        target_padded = target_padded[:-1]

        dec_state = dec_init_state

        # Initialize previous combined output vector o_{t-1} as zero
        batch_size = enc_hiddens.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

        # Initialize a list we will use to collect the combined output o_t on each step
        combined_outputs = []

        ### TODO:
        ###     1. Construct tensor `Y` of target sentences with shape (tgt_len, b, e) using the target model embeddings.
        ###         where tgt_len = maximum target sentence length, b = batch size, e = embedding size.
        ###     2. Use the torch.split function to iterate over the time dimension of Y.
        ###         Within the loop, this will give you Y_t of shape (1, b, e) where b = batch size, e = embedding size.
        ###             - Squeeze Y_t into a tensor of dimension (b, e). 
        ###             - Construct Ybar_t by concatenating Y_t with o_prev on their last dimension
        ###             - Use the step function to compute the the Decoder's next (cell, state) values
        ###               as well as the new combined output o_t.
        ###             - Append o_t to combined_outputs
        ###             - Update o_prev to the new o_t.
        ###     3. Use torch.stack to convert combined_outputs from a list length tgt_len of
        ###         tensors shape (b, h), to a single tensor shape (tgt_len, b, h)
        ###         where tgt_len = maximum target sentence length, b = batch size, h = hidden size.

        enc_hiddens_proj = self.att_projection(enc_hiddens)
        Y = self.embedding(target_padded)
        
        for Y_t in torch.split(Y, split_size_or_sections=1):
            Y_t = Y_t.squeeze(0)
            Ybar_t = torch.cat([Y_t, o_prev], dim=-1)
            dec_state, o_t = self.step(Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks)
            combined_outputs.append(o_t)
            o_prev = o_t

        combined_outputs = torch.stack(combined_outputs)

        return combined_outputs
    
    def step(self, Ybar_t: torch.Tensor,
                 dec_state: Tuple[torch.Tensor, torch.Tensor],
                 enc_hiddens: torch.Tensor, 
                 enc_hiddens_proj: torch.Tensor,
                 enc_masks: torch.Tensor) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        """ Compute one forward step of the LSTM decoder, including the attention computation.

        :param Ybar_t: Concatenated Tensor of [Y_t o_prev], with shape (b, e + h). The input for the decoder,
                                where b = batch size, e = embedding size, h = hidden size.
        :type Ybar_t: torch.Tensor
        :param dec_state: Tensors with shape (b, h), where b = batch size, h = hidden size.
                Tensor is decoder's prev hidden state
        :type dec_state: torch.Tensor
        :param enc_hiddens: Encoder hidden states Tensor, with shape (b, src_len, h * 2), where b = batch size,
                                    src_len = maximum source length, h = hidden size.
        :type enc_hiddens: torch.Tensor

        :returns dec_state: Tensors with shape (b, h), where b = batch size, h = hidden size.
                Tensor is decoder's new hidden state. For an LSTM, this should be a tuple
                of the hidden state and cell state.
        returns combined_output: Combined output Tensor at timestep t, shape (b, h), where b = batch size, h = hidden size.
        """
        combined_output = None

        ### TODO:
        ###     1. Apply the decoder to `Ybar_t` and `dec_state` to obtain the new dec_state.
        ###     2. Rename dec_state to dec_hidden

        dec_state = self.decoder(Ybar_t, dec_state)
        (dec_hidden, dec_cell) = dec_state
        e_t = torch.bmm(enc_hiddens_proj, dec_hidden.unsqueeze(2)).squeeze(2)

        ### TODO:
        ###     1. Apply the combined output projection layer to h^dec_t to compute tensor V_t
        ###     2. Compute tensor O_t by applying the Tanh function.

        alpha_t = self.softmax(e_t)
        a_t = torch.bmm(alpha_t.unsqueeze(1), enc_hiddens).squeeze(1)
        U_t = torch.cat([dec_hidden, a_t], 1)
        V_t = self.combined_output_projection(U_t)
        O_t = self.dropout(torch.tanh(V_t))

        combined_output = O_t

        return dec_state, combined_output