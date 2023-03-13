import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from collections import namedtuple
from encoder import Encoder, generate_sent_masks
from decoder import Decoder

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

class NMT(nn.Module):
    """ Simple Neural Machine Translation Model:
        - Bidrectional RNN Encoder
        - Unidirection RNN Decoder
    """
    def __init__(self, embed_size, hidden_size, src_vocab, tgt_vocab, d_rate,
                 device=torch.device("cpu"), pretrained_source=None,pretrained_target=None):
        """ Init NMT Model.

        :param embed_size: Embedding size (dimensionality)
        :type embed_size: int
        :param hidden_size: Hidden Size, the size of hidden states (dimensionality)
        :type hidden_size: int
        :param src_vocab: Vocabulary object containing src language
        :type src_vocab: Vocab
        :param tgt_vocab: Vocabulary object containing tgt language
        :type tgt_vocab: Vocab
        :param device: torch device to put all modules on
        :type device: torch.device
        :param pretrained_source: Matrix of pre-trained source word embeddings
        :type pretrained_source: Optional[torch.Tensor]
        :param pretrained_target: Matrix of pre-trained target word embeddings
        :type pretrained_target: Optional[torch.Tensor]
        """
        super(NMT, self).__init__()
        self.device=device
        self.embed_size = embed_size
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.d_rate = d_rate
        src_pad_token_idx = src_vocab['<pad>']
        tgt_pad_token_idx = tgt_vocab['<pad>']
        self.source_embedding = nn.Embedding(len(src_vocab), embed_size, padding_idx=src_pad_token_idx)
        self.target_embedding = nn.Embedding(len(tgt_vocab), embed_size, padding_idx=tgt_pad_token_idx)

        with torch.no_grad():
            if pretrained_source is not None:
                self.source_embedding.weight.data = pretrained_source
                # TODO: Decide if we want the embeddings to update as we train
                self.source_embedding.weight.requires_grad = True
        
            if pretrained_target is not None:
                self.target_embedding.weight.data = pretrained_target
                # TODO: Decide if we want the embeddings to update as we train
                self.target_embedding.weight.requires_grad = True
        
        self.hidden_size = hidden_size

        self.encoder = Encoder(
            embed_size=embed_size,
            hidden_size=hidden_size,
            source_embeddings=self.source_embedding,
        )
        self.decoder = Decoder(
            embed_size=embed_size,
            hidden_size=hidden_size,
            target_embedding=self.target_embedding,
            device=self.device,            
            d_rate = self.d_rate
        )


    def forward(self, source: List[List[str]], target: List[List[str]]) -> torch.Tensor:
        """ Take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences under the language models learned by the NMT system.

        :param source: list of source sentence tokens
        :type source: List[List[str]]
        :param target: list of target sentence tokens, wrapped by `<s>` and `</s>`
        :type target: List[List[str]]
        :returns scores: a variable/tensor of shape (b, ) representing the
                                    log-likelihood of generating the gold-standard target sentence for
                                    each example in the input batch. Here b = batch size.
        :rtype: torch.Tensor
        """
        # Compute sentence lengths
        source_lengths = [len(s) for s in source]

        # Convert list of lists into tensors
        source_padded = self.src_vocab.to_input_tensor(source, device=self.device)   # Tensor: (src_len, b)
        target_padded = self.tgt_vocab.to_input_tensor(target, device=self.device)   # Tensor: (tgt_len, b)
        
        ###     Run the network forward:
        ###     1. Apply the encoder to `source_padded` by calling `self.encode()`
        ###     2. Generate sentence masks for `source_padded` by calling `self.generate_sent_masks()`
        ###     3. Apply the decoder to compute combined-output by calling `self.decode()`
        ###     4. Compute log probability distribution over the target vocabulary using the
        ###        combined_outputs returned by the `self.decode()` function.

        enc_hiddens, dec_init_state = self.encode(source_padded, source_lengths)
        enc_masks = generate_sent_masks(enc_hiddens, source_lengths, self.device)
        combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init_state, target_padded)
        P = F.log_softmax(self.decoder.target_vocab_projection(combined_outputs), dim=-1)

        # Zero out, probabilities for which we have nothing in the target text
        target_masks = (target_padded != self.tgt_vocab['<pad>']).float()
        
        # Compute log probability of generating true target words
        target_gold_words_log_prob = torch.gather(P, index=target_padded[1:].unsqueeze(-1), dim=-1).squeeze(-1) * target_masks[1:]
        scores = target_gold_words_log_prob.sum(dim=0)
        return scores


    def encode(self, source_padded: torch.Tensor, source_lengths: List[int]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """ Apply the encoder to source sentences to obtain encoder hidden states.
            Additionally, take the final states of the encoder and project them to obtain initial states for decoder.

        :param source_padded: Tensor of padded source sentences with shape (src_len, b), where
            b = batch_size, src_len = maximum source sentence length. Note that these have
            already been sorted in order of longest to shortest sentence.
        :type source_padded: torch.Tensor
        :param source_lengths: List of actual lengths for each of the source sentences in the batch
        :type source_lengths: List[int]
        :returns: Tuple of two items. The first is Tensor of hidden units with shape (b, src_len, h*2),
            where b = batch size, src_len = maximum source sentence length, h = hidden size. The second is
            Tuple of tensors representing the decoder's initial hidden state and cell.
        :rtype: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        """
        return self.encoder(source_padded, source_lengths)

    def decode(self, enc_hiddens: torch.Tensor, enc_masks: torch.Tensor,
               dec_init_state: torch.Tensor, target_padded: torch.Tensor) -> torch.Tensor:
        """Compute combined output vectors for a batch.

        :param enc_hiddens (Tensor): Hidden states (b, src_len, h*2), where
                                     b = batch size, src_len = maximum source sentence length, h = hidden size.
        :param dec_init_state (tuple(Tensor, Tensor)): Initial state and cell for decoder
        :param target_padded: Gold-standard padded target sentences (tgt_len, b), where
                                       tgt_len = maximum target sentence length, b = batch size. 

        :returns combined_outputs: combined output tensor  (tgt_len, b,  h), where
                                    tgt_len = maximum target sentence length, b = batch_size,  h = hidden size
        :rtype: torch.Tensor
        """
        return self.decoder(enc_hiddens, enc_masks, dec_init_state, target_padded)

    def beam_search(self, src_sent: List[str], beam_size: int=5, max_decoding_time_step: int=70) -> List[Hypothesis]:
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
        :param src_sent: a single source sentence (words)
        :type src_sent: List[str]
        :param beam_size: beam size
        :type beam_size: int
        :param max_decoding_time_step: maximum number of time steps to unroll the decoding RNN
        :type max_decoding_time_step: int
        :returns hypotheses: a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        :rtype: List[Hypothesis]
        """
        src_sents_var = self.src_vocab.to_input_tensor([src_sent], self.device)

        src_encodings, dec_init_vec = self.encode(src_sents_var, [len(src_sent)])
        src_encodings_att_linear = self.decoder.att_projection(src_encodings)

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

        eos_id = self.tgt_vocab['</s>']

        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))
            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num, 
                                                                           src_encodings_att_linear.size(1), 
                                                                           src_encodings_att_linear.size(2))


            y_tm1 = torch.tensor([self.tgt_vocab[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
            y_t_embed = self.target_embedding(y_tm1)

            x = torch.cat([y_t_embed, att_tm1], dim=-1)

            h_t, att_t = self.decoder.step(x, h_tm1,
                                exp_src_encodings,
                                exp_src_encodings_att_linear, enc_masks=None)
            
            ## TODO: Uncomment the line below if this is an LSTM
            h_t, c_t = h_t

            # log probabilities over target words
            log_p_t = F.log_softmax(self.decoder.target_vocab_projection(att_t), dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = torch.div(top_cand_hyp_pos, len(self.tgt_vocab), rounding_mode='floor')
            hyp_word_ids = top_cand_hyp_pos % len(self.tgt_vocab)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.tgt_vocab.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '</s>':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)

            # h_tm1 = h_t[live_hyp_ids]
            ### TODO: Uncomment the below if it is an LSTM and comment out line
            # above. Otherwise leave.
            h_tm1 = h_t[live_hyp_ids], c_t[live_hyp_ids]
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses


    def greedy(self, src_sent: List[str], max_decoding_time_step: int=70) -> List[Hypothesis]:
        return self.beam_search(src_sent, beam_size=1, max_decoding_time_step=max_decoding_time_step)


    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NMT(
            src_vocab=params['vocab']['source'],
            tgt_vocab=params['vocab']['target'],
            d_rate=0.2, 
            **args
        )
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the model to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(embed_size=self.embed_size, hidden_size=self.hidden_size),
            'vocab': dict(source=self.src_vocab, target=self.tgt_vocab),
            'state_dict': self.state_dict()
        }

        torch.save(params, path)