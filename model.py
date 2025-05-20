import torch 
import torch.nn.functional as F
from torch import nn
import random
from utils import devnagri2int


EOS_TOKEN = "<EOS>"
SOS_TOKEN = "<SOS>"
class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, nonlinearity="tanh", dropout_p=0.1, layer="rnn"):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layer = layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        if layer == "rnn":
            self.cell = nn.RNN(embed_size, hidden_size, num_layers, nonlinearity, batch_first=True) 
        elif layer == "gru":   
            self.cell = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        elif layer == "lstm":
            self.cell = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input, input_lengths, hidden=None):
        embedded = self.dropout(self.embedding(input))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, input_lengths, batch_first=True, enforce_sorted=True
        )
        
        if self.layer == "lstm":
            output, (hidden, cell) = self.cell(packed, hidden)
        else:
            output, hidden = self.cell(packed, hidden)
            cell = None
            
        # Unpack sequence
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        
        return output, hidden, cell
    
class BeamSearchNode:
    def __init__(self, hidden_state, cell_state, prev_node, token_id, log_prob, length):
        self.hidden = hidden_state
        self.cell   = cell_state
        self.prev   = prev_node
        self.token  = token_id
        self.logp   = log_prob
        self.length = length

    def get_score(self, length_normalize=True):
        return (self.logp / (self.length + 1e-6)) if length_normalize else self.logp

    def __lt__(self, other):
        return self.get_score() < other.get_score()


class DecoderRNN(nn.Module):
    def __init__(self,vocab_size,embed_size,hidden_size,num_layers=1,nonlinearity="tanh",layer="rnn",pad_token_id=0):
        super().__init__()
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.layer        = layer
        self.embedding    = nn.Embedding(vocab_size, embed_size, padding_idx=pad_token_id)
        self.vocab_size   = vocab_size
        self.pad_token_id = pad_token_id

        if layer == "rnn":
            self.cell = nn.RNN(embed_size, hidden_size, num_layers,
                               nonlinearity, batch_first=True)
        elif layer == "gru":
            self.cell = nn.GRU(embed_size, hidden_size, num_layers,
                               batch_first=True)
        elif layer == "lstm":
            self.cell = nn.LSTM(embed_size, hidden_size, num_layers,
                                batch_first=True)

        self.fc = nn.Linear(hidden_size, vocab_size)


    def forward(self,encoder_outputs,encoder_hidden,encoder_cell, target_tensor=None,MAX_LENGTH=None,teacher_forcing_prob=0.5,beam_width=5):
        B = encoder_outputs.size(0)
        device = encoder_outputs.device

        # 1) init states
        if self.layer == "lstm":
            decoder_hidden = encoder_hidden.contiguous()
            decoder_cell   = encoder_cell.contiguous()
        else:
            decoder_hidden = encoder_hidden.contiguous()
            decoder_cell   = None

        # 2) training branch
        if target_tensor is not None:
            T = target_tensor.size(1)
            input_tok = torch.full((B, 1),
                                   devnagri2int[SOS_TOKEN],
                                   dtype=torch.long,
                                   device=device)
            outputs = []
            for t in range(T):
                emb = self.embedding(input_tok)
                if self.layer == "lstm":
                    h = decoder_hidden.contiguous()
                    c = decoder_cell.contiguous()
                    out, (decoder_hidden, decoder_cell) = self.cell(emb, (h, c))
                else:
                    hx = decoder_hidden.contiguous()
                    out, decoder_hidden = self.cell(emb, hx)

                logits = self.fc(out.squeeze(1))
                logp   = F.log_softmax(logits, dim=1)
                outputs.append(logp.unsqueeze(1))

                if random.random() < teacher_forcing_prob:
                    input_tok = target_tensor[:, t].unsqueeze(1)
                else:
                    input_tok = logp.argmax(1).unsqueeze(1)

            return torch.cat(outputs, dim=1), decoder_hidden, decoder_cell, None

        # 3) inference with **batched** beam search
        else:
            K = beam_width
            V = self.vocab_size
            max_len = MAX_LENGTH or 30
            sos = devnagri2int[SOS_TOKEN]
            eos = devnagri2int[EOS_TOKEN]

            # a) expand hidden/cell: (layers, B, H) → (layers, B*K, H)
            if self.layer == "lstm":
                h0 = encoder_hidden.contiguous().unsqueeze(2).repeat(1, 1, K, 1)
                c0 = encoder_cell.contiguous().unsqueeze(2).repeat(1, 1, K, 1)
                hidden = h0.view(self.num_layers, B*K, self.hidden_size)
                cell   = c0.view(self.num_layers, B*K, self.hidden_size)
            else:
                h0 = encoder_hidden.contiguous().unsqueeze(2).repeat(1, 1, K, 1)
                hidden = h0.view(self.num_layers, B*K, self.hidden_size)
                cell   = None

            # b) init scores & sequences
            scores = torch.zeros(B, K, device=device)
            scores[:,1:] = -1e9
            seqs = torch.full((B, K, max_len),
                              self.pad_token_id,
                              dtype=torch.long,
                              device=device)
            seqs[:,:,0] = sos
            input_tok = torch.full((B*K,1), sos, dtype=torch.long, device=device)

            # c) step time
            for t in range(1, max_len):
                emb = self.embedding(input_tok)  # (B*K,1,E)
                if self.layer == "lstm":
                    h_in, c_in = hidden.contiguous(), cell.contiguous()
                    out, (h_out, c_out) = self.cell(emb, (h_in, c_in))
                else:
                    h_in = hidden.contiguous()
                    out, h_out = self.cell(emb, h_in)
                    c_out = None

                logits   = self.fc(out.squeeze(1))            # (B*K, V)
                logp_all = F.log_softmax(logits, dim=-1).view(B, K, V)

                total_scores = scores.unsqueeze(2) + logp_all  # (B, K, V)
                flat = total_scores.view(B, -1)               # (B, K*V)
                top_scores, top_idx = flat.topk(K, dim=-1)    # (B, K)

                beam_idx  = top_idx // V                      # (B, K)
                token_idx = top_idx %  V                      # (B, K)

                # reorder hidden
                h_beams = h_out.view(self.num_layers, B, K, self.hidden_size)
                hidden  = h_beams.gather(
                    2,
                    beam_idx.unsqueeze(0).unsqueeze(-1)
                            .expand(self.num_layers, B, K, self.hidden_size)
                ).view(self.num_layers, B*K, self.hidden_size)

                if self.layer == "lstm":
                    c_beams = c_out.view(self.num_layers, B, K, self.hidden_size)
                    cell    = c_beams.gather(
                        2,
                        beam_idx.unsqueeze(0).unsqueeze(-1)
                                .expand(self.num_layers, B, K, self.hidden_size)
                    ).view(self.num_layers, B*K, self.hidden_size)

                scores = top_scores  # update

                # reorder & append seqs
                seqs = seqs.gather(
                    1,
                    beam_idx.unsqueeze(-1)
                            .expand(B, K, max_len)
                )
                seqs[:,:,t] = token_idx
                input_tok = token_idx.view(B*K,1)

                if (token_idx == eos).all():
                    break

            # d) select best beam
            best = scores.argmax(dim=-1)  # (B,)
            preds = seqs[torch.arange(B, device=device), best]  # (B, max_len)

            return preds, None, None, None
        

class BeamSearchNode:
    def __init__(self, hidden_state, cell_state, prev_node, token_id, log_prob, length):
        self.hidden = hidden_state
        self.cell   = cell_state
        self.prev   = prev_node
        self.token  = token_id
        self.logp   = log_prob
        self.length = length

    def get_score(self, length_normalize=True):
        return (self.logp / (self.length + 1e-6)) if length_normalize else self.logp

    def __lt__(self, other):
        return self.get_score() < other.get_score()


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Ua = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, decoder_hidden, encoder_outputs):
        h = decoder_hidden[0] if isinstance(decoder_hidden, tuple) else decoder_hidden
        query = h[-1].unsqueeze(1)  # (B, 1, H)
        energy = torch.tanh(self.Wa(query) + self.Ua(encoder_outputs))  # (B, T, H)
        scores = self.Va(energy).squeeze(-1)  # (B, T)
        weights = F.softmax(scores, dim=1).unsqueeze(1)  # (B,1,T)
        context = torch.bmm(weights, encoder_outputs)  # (B,1,H)
        return context, weights


class AttnDecoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size,
                 num_layers=1, nonlinearity="tanh", layer="lstm", pad_token_id=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layer = layer
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_token_id)
        self.attention = BahdanauAttention(hidden_size)
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        rnn_input_dim = embed_size + hidden_size
        cell_cls = {
            'lstm': nn.LSTM,
            'gru': nn.GRU,
            'rnn': lambda *args, **kwargs: nn.RNN(*args, nonlinearity=nonlinearity, **kwargs)
        }[layer]
        self.rnn = cell_cls(rnn_input_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, encoder_outputs, encoder_hidden, encoder_cell=None,
                target_tensor=None, MAX_LENGTH=None, teacher_forcing_prob=0.5,
                beam_width=5):
        B, T_enc, _ = encoder_outputs.size()
        device = encoder_outputs.device
        # init decoder state
        if self.layer == 'lstm':
            dec_hidden = encoder_hidden.contiguous()
            dec_cell = encoder_cell.contiguous()
        else:
            dec_hidden = encoder_hidden.contiguous()
            dec_cell = None

        if target_tensor is not None:
            # TRAINING with partial teacher forcing
            T = target_tensor.size(1)
            input_tok = torch.full((B,), devnagri2int[SOS_TOKEN], dtype=torch.long, device=device)
            outputs = []
            for t in range(T):
                emb = self.embedding(input_tok).unsqueeze(1)  # (B,1,E)
                context, _ = self.attention(
                    dec_hidden, encoder_outputs)
                rnn_input = torch.cat([emb, context], dim=2)

                if self.layer == 'lstm':
                    h, c = dec_hidden.contiguous(), dec_cell.contiguous()
                    out, (dec_hidden, dec_cell) = self.rnn(rnn_input, (h, c))
                else:
                    h = dec_hidden.contiguous()
                    out, dec_hidden = self.rnn(rnn_input, h)
                logits = self.fc(out.squeeze(1))
                logp = F.log_softmax(logits, dim=1)
                outputs.append(logp.unsqueeze(1))
                teacher = random.random() < teacher_forcing_prob
                top1 = logp.argmax(1)
                input_tok = target_tensor[:, t] if teacher else top1
            return torch.cat(outputs, dim=1), dec_hidden, dec_cell, None

        # INFERENCE with batched beam search
        else:
            K = beam_width
            max_len = MAX_LENGTH or 30
            sos = devnagri2int[SOS_TOKEN]; eos = devnagri2int[EOS_TOKEN]
            # expand states: (layers,B,H)->(layers,B*K,H)
            def expand(x): return x.unsqueeze(2).repeat(1,1,K,1)
            if self.layer=='lstm':
                h0, c0 = expand(dec_hidden), expand(dec_cell)
                hidden = h0.view(self.num_layers, B*K, self.hidden_size)
                cell = c0.view(self.num_layers, B*K, self.hidden_size)
            else:
                h0 = expand(dec_hidden)
                hidden, cell = h0.view(self.num_layers,B*K,self.hidden_size), None
            # beam data
            scores = torch.zeros(B, K, device=device);
            scores[:,1:] = -1e9
            seqs = torch.full((B,K,max_len), self.pad_token_id, device=device, dtype=torch.long)
            seqs[:,:,0] = sos
            input_tok = torch.full((B*K,), sos, dtype=torch.long, device=device)
            # time loop
            all_attn = []
            for t in range(1, max_len):
                emb = self.embedding(input_tok).unsqueeze(1)  # (B*K,1,E)
                # attention per beam
                h_layer = hidden.view(self.num_layers,B,K,self.hidden_size)[-1]
                h_flat = h_layer.view(B*K,self.hidden_size).unsqueeze(0)
                enc_flat = encoder_outputs.unsqueeze(1).repeat(1,K,1,1).view(B*K,T_enc,self.hidden_size)
                context, attn_weights = self.attention(h_flat, enc_flat)
                attn_w = attn_weights.squeeze()
                all_attn.append(attn_w)
                rnn_in = torch.cat([emb, context.view(B*K,1,self.hidden_size)], dim=2)
                # RNN step
                if self.layer=='lstm':
                    out,(h_new,c_new)=self.rnn(rnn_in,(hidden.contiguous(),cell.contiguous()))
                else:
                    out,h_new=self.rnn(rnn_in,hidden.contiguous()); c_new=None
                # scores
                logp = F.log_softmax(self.fc(out.squeeze(1)),dim=1).view(B,K,self.vocab_size)
                total = scores.unsqueeze(2) + logp
                flat = total.view(B,-1)
                top_scores, top_idx = flat.topk(K,dim=-1)
                beam_idx, token_idx = top_idx//self.vocab_size, top_idx%self.vocab_size
                # reorder hidden/cell
                def gather_beams(x):
                    xb = x.view(self.num_layers,B,K,self.hidden_size)
                    return xb.gather(2,beam_idx.unsqueeze(0).unsqueeze(-1)
                                    .expand(self.num_layers,B,K,self.hidden_size))
                hidden = gather_beams(h_new).view(self.num_layers,B*K,self.hidden_size)
                if cell is not None:
                    cell = gather_beams(c_new).view(self.num_layers,B*K,self.hidden_size)
                scores = top_scores
                seqs = seqs.gather(1,beam_idx.unsqueeze(-1).expand(B,K,max_len))
                seqs[:,:,t] = token_idx
                input_tok = token_idx.view(B*K)
                if (token_idx==eos).all(): break
            # after the beam loop
            best   = scores.argmax(dim=-1)
            preds  = seqs[torch.arange(B, device=device), best]

            # number of timesteps actually decoded
            Tt     = len(all_attn)
            # stack into (Tt, B*K, src_len)
            stacked = torch.stack(all_attn, dim=0)
            # permute to (B*K, Tt, src_len)
            perm    = stacked.permute(1, 0, 2)
            # reshape to (B, K, Tt, src_len)
            attn_beams = perm.view(B, K, Tt, T_enc)
            # select the top beam
            attn_top1 = attn_beams[torch.arange(B, device=device), best, :, :]  # (B, Tt, T_enc)

            return preds, None, None, attn_top1

