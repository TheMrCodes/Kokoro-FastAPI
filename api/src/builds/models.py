# https://github.com/yl4579/StyleTTS2/blob/main/models.py
import json
import os
import os.path as osp
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from munch import Munch
from torch.nn.utils import spectral_norm, weight_norm

from .istftnet import AdaIN1d, Decoder
from .plbert import load_plbert


try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    pass

def is_xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()



class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)

class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)
    
class TextEncoder(nn.Module):
    def __init__(self, channels, kernel_size, depth, n_symbols, actv=nn.LeakyReLU(0.2)):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)

        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList()
        for _ in range(depth):
            self.cnn.append(nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)),
                LayerNorm(channels),
                actv,
                nn.Dropout(0.2),
            ))
        # self.cnn = nn.Sequential(*self.cnn)

        self.lstm = nn.LSTM(channels, channels//2, 1, batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths, m):
        x = self.embedding(x)  # [B, T, emb]
        x = x.transpose(1, 2)  # [B, emb, T]
        m = m.to(input_lengths.device).unsqueeze(1)
        x.masked_fill_(m, 0.0)
        
        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)
            
        x = x.transpose(1, 2)  # [B, T, chn]

        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True, enforce_sorted=False)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)
                
        x = x.transpose(-1, -2)
        x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]])

        x_pad[:, :, :x.shape[-1]] = x
        x = x_pad.to(x.device)
        
        x.masked_fill_(m, 0.0)
        
        return x

    def inference(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        return x
    
    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask


class UpSample1d(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        else:
            return F.interpolate(x, scale_factor=2, mode='nearest')

class AdainResBlk1d(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, actv=nn.LeakyReLU(0.2),
                 upsample='none', dropout_p=0.0):
        super().__init__()
        self.actv = actv
        self.upsample_type = upsample
        self.upsample = UpSample1d(upsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)
        self.dropout = nn.Dropout(dropout_p)
        
        if upsample == 'none':
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(nn.ConvTranspose1d(dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in, padding=1, output_padding=1))
        
        
    def _build_weights(self, dim_in, dim_out, style_dim):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.pool(x)
        x = self.conv1(self.dropout(x))
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(self.dropout(x))
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) / np.sqrt(2)
        return out
    
class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.fc = nn.Linear(style_dim, channels*2)

    def forward(self, x, s):
        x = x.transpose(-1, -2)
        x = x.transpose(1, -1)
                
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma, beta = gamma.transpose(1, -1), beta.transpose(1, -1)
        
        
        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x.transpose(1, -1).transpose(-1, -2)

class ProsodyPredictor(nn.Module):

    def __init__(self, style_dim, d_hid, nlayers, max_dur=50, dropout=0.1):
        super().__init__() 
        
        self.text_encoder = DurationEncoder(sty_dim=style_dim, 
                                            d_model=d_hid,
                                            nlayers=nlayers, 
                                            dropout=dropout)

        self.lstm = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.duration_proj = LinearNorm(d_hid, max_dur)
        
        self.shared = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.F0 = nn.ModuleList()
        self.F0.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.F0.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout))
        self.F0.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))

        self.N = nn.ModuleList()
        self.N.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))
        
        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)


    def forward(self, texts, style, text_lengths, alignment, m):
        d = self.text_encoder(texts, style, text_lengths, m)
        
        batch_size = d.shape[0]
        text_size = d.shape[1]
        
        # predict duration
        input_lengths = text_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            d, input_lengths, batch_first=True, enforce_sorted=False)
        
        m = m.to(text_lengths.device).unsqueeze(1)
        
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)
        
        x_pad = torch.zeros([x.shape[0], m.shape[-1], x.shape[-1]])

        x_pad[:, :x.shape[1], :] = x
        x = x_pad.to(x.device)
                
        duration = self.duration_proj(nn.functional.dropout(x, 0.5, training=self.training))
        
        en = (d.transpose(-1, -2) @ alignment)

        return duration.squeeze(-1), en
    
    def F0Ntrain(self, x, s):
        x, _ = self.shared(x.transpose(-1, -2))
        
        F0 = x.transpose(-1, -2)
        for block in self.F0:
            F0 = block(F0, s)
        F0 = self.F0_proj(F0)

        N = x.transpose(-1, -2)
        for block in self.N:
            N = block(N, s)
        N = self.N_proj(N)
        
        return F0.squeeze(1), N.squeeze(1)
    
    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask

class DurationEncoder(nn.Module):

    def __init__(self, sty_dim, d_model, nlayers, dropout=0.1):
        super().__init__()
        self.lstms = nn.ModuleList()
        for _ in range(nlayers):
            self.lstms.append(nn.LSTM(d_model + sty_dim, 
                                 d_model // 2, 
                                 num_layers=1, 
                                 batch_first=True, 
                                 bidirectional=True, 
                                 dropout=dropout))
            self.lstms.append(AdaLayerNorm(sty_dim, d_model))
        
        
        self.dropout = dropout
        self.d_model = d_model
        self.sty_dim = sty_dim

    def forward(self, x, style, text_lengths, m):
        masks = m.to(text_lengths.device)
        
        x = x.permute(2, 0, 1)
        s = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, s], axis=-1)
        x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)
                
        x = x.transpose(0, 1)
        input_lengths = text_lengths.cpu().numpy()
        x = x.transpose(-1, -2)
        
        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, -1, 0)], axis=1)
                x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                x = x.transpose(-1, -2)
                x = nn.utils.rnn.pack_padded_sequence(
                    x, input_lengths, batch_first=True, enforce_sorted=False)
                block.flatten_parameters()
                x, _ = block(x)
                x, _ = nn.utils.rnn.pad_packed_sequence(
                    x, batch_first=True)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = x.transpose(-1, -2)
                
                x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]])

                x_pad[:, :, :x.shape[-1]] = x
                x = x_pad.to(x.device)
        
        return x.transpose(-1, -2)
    
    def inference(self, x, style):
        x = self.embedding(x.transpose(-1, -2)) * np.sqrt(self.d_model)
        style = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, style], axis=-1)
        src = self.pos_encoder(x)
        output = self.transformer_encoder(src).transpose(0, 1)
        return output
    
    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask

# https://github.com/yl4579/StyleTTS2/blob/main/utils.py
def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d
    
def length_to_mask(lengths):
    """Create attention mask from lengths - possibly optimized version"""
    max_len = lengths.max()
    # Create mask directly on the same device as lengths
    mask = torch.arange(max_len, device=lengths.device)[None, :].expand(
        lengths.shape[0], -1
    )
    # Avoid type_as by using the correct dtype from the start
    if lengths.dtype != mask.dtype:
        mask = mask.to(dtype=lengths.dtype)
    # Fuse operations  using broadcasting
    return mask + 1 > lengths[:, None]

class KokoroModel(nn.Module):
    bert: nn.Module
    bert_encoder: nn.Module
    predictor: nn.Module
    decoder: nn.Module
    text_encoder: nn.Module

    @classmethod
    def load_config(cls):
        config = Path(__file__).parent / 'config.json'
        assert config.exists(), f'Config path incorrect: config.json not found at {config}'
        with open(config, 'r') as r:
            args = recursive_munch(json.load(r))
        assert args.decoder.type == 'istftnet', f'Unknown decoder type: {args.decoder.type}'
        return args



    def __init__(self, device, *args, **kwargs):
        super().__init__()
        self.device = device
        args = self.load_config()
        self.text_encoder = TextEncoder(channels=args.hidden_dim, kernel_size=5, depth=args.n_layer, n_symbols=args.n_token)
        self.predictor = ProsodyPredictor(style_dim=args.style_dim, d_hid=args.hidden_dim, nlayers=args.n_layer, max_dur=args.max_dur, dropout=args.dropout)
        self.bert = load_plbert()
        self.decoder = Decoder(
            dim_in=args.hidden_dim, style_dim=args.style_dim, dim_out=args.n_mels,
            resblock_kernel_sizes = args.decoder.resblock_kernel_sizes,
            upsample_rates = args.decoder.upsample_rates,
            upsample_initial_channel=args.decoder.upsample_initial_channel,
            resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
            upsample_kernel_sizes=args.decoder.upsample_kernel_sizes,
            gen_istft_n_fft=args.decoder.gen_istft_n_fft, gen_istft_hop_size=args.decoder.gen_istft_hop_size
        )
        self.bert_encoder = nn.Linear(self.bert.config.hidden_size, args.hidden_dim)

        for parent in [self.bert, self.bert_encoder, self.predictor, self.decoder, self.text_encoder]:
            for child in parent.children():
                if isinstance(child, nn.RNNBase):
                    child.flatten_parameters()

    def load_from_file(self, path):
        for key, state_dict in torch.load(path, map_location=self.device, weights_only=True)['net'].items():
            assert key in ['bert', 'bert_encoder', 'predictor', 'decoder', 'text_encoder'], key
            try:
                getattr(self, key).load_state_dict(state_dict)
            except:
                state_dict = {k[7:]: v for k, v in state_dict.items()}
                getattr(self, key).load_state_dict(state_dict, strict=False)


    @classmethod
    def from_pretrained(cls, path, device):
        model = cls(device)
        model.load_from_file(path)
        
        # Optimizations
        model = model.to(device).eval()
        if device == 'xpu':
            model = ipex.optimize(model, conv_bn_folding=False, linear_bn_folding=False, split_master_weight_for_bf16=False)
        return model

    @torch.no_grad()
    def forward(model, tokens, ref_s, speed):
        """Forward pass through the model with moderate memory management"""
        device = ref_s.device
        
        try:
            # Initial tensor setup with proper device placement
            tokens = torch.LongTensor([[0, *tokens, 0]]).to(device)
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
            text_mask = length_to_mask(input_lengths).to(device)

            # Split and clone reference signals with explicit device placement
            s_content = ref_s[:, 128:].clone().to(device)
            s_ref = ref_s[:, :128].clone().to(device)

            # BERT and encoder pass
            bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

            # Predictor forward pass
            d = model.predictor.text_encoder(d_en, s_content, input_lengths, text_mask)
            x, _ = model.predictor.lstm(d)

            # Duration prediction
            duration = model.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1) / speed
            pred_dur = torch.round(duration).clamp(min=1).long()
            # Only cleanup large intermediates
            del duration, x

            # Alignment matrix construction
            pred_aln_trg = torch.zeros(input_lengths.item(), pred_dur.sum().item(), device=device)
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame : c_frame + pred_dur[0, i].item()] = 1
                c_frame += pred_dur[0, i].item()
            pred_aln_trg = pred_aln_trg.unsqueeze(0)

            # Matrix multiplications with selective cleanup
            en = d.transpose(-1, -2) @ pred_aln_trg
            del d  # Free large intermediate tensor
            
            F0_pred, N_pred = model.predictor.F0Ntrain(en, s_content)
            del en  # Free large intermediate tensor

            # Final text encoding and decoding
            t_en = model.text_encoder(tokens, input_lengths, text_mask)
            asr = t_en @ pred_aln_trg
            del t_en  # Free large intermediate tensor

            # Final decoding and transfer to CPU
            output = model.decoder(asr, F0_pred, N_pred, s_ref)
            result = output.squeeze().cpu().numpy()
            
            return result
            
        finally:
            # Let PyTorch handle most cleanup automatically
            # Only explicitly free the largest tensors
            del pred_aln_trg, asr

