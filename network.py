from logging import raiseExceptions
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import tinycudann as tcnn

class GegEmbedding(nn.Module):
    def __init__(self, dim_out, alpha=0.5):
        super(GegEmbedding, self).__init__()
        self.dim_out = dim_out
        self.alpha = alpha
    
    def forward(self, x):
        n = len(x)
        x_in = x.squeeze()
        c = torch.zeros((n, self.dim_out+1), device=x.device)
        c[..., 0] = 1.0
        c[..., 1] = 2.0 * self.alpha * x_in
        for i in range(2, self.dim_out+1):
            c[..., i] = ((2*i-2+2.0*self.alpha) * x_in * c[..., i-1] + (-i + 2 -2.0 * self.alpha) * c[..., i-2]) / i

        return c[..., 1:].contiguous().view(n, -1)

class PositionalEncoder(nn.Module):
    def __init__(self, dim_in, num_freqs, max_freq=-1, log_sampling=True, include_input=True):
        super(PositionalEncoder, self).__init__()

        if max_freq < 0:
            max_freq = num_freqs - 1

        self.dim_in = dim_in
        self.num_freqs = num_freqs
        self.max_freq = max_freq
        self.include_input = include_input
        self.log_sampling = log_sampling

        if log_sampling:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=num_freqs)
        else:
            freq_bands = torch.linspace(0., 2.**max_freq, steps=num_freqs)
        self.register_buffer('freq_bands', freq_bands, persistent=False)

        self.dim_out = num_freqs * self.dim_in * 2
        if self.include_input:
            self.dim_out += self.dim_in

    def embed_size(self):
        return self.dim_out

    def forward(self, x):
        """
        Args:
        x: [N_pts, 3] point coordinates
        Return:
        embedded results of out_size() dimension
        """
        freq_bands = self.freq_bands.expand(1, 1, self.num_freqs) # [1, 1, N_freqs]
        y = x[..., None] * freq_bands  # [N_pts, dim_in, N_freqs]
        y = y.view(x.shape[0], -1) # [N_pts, dim_in * N_freqs]
        if self.include_input:
            return torch.cat([x, torch.sin(y), torch.cos(y)], -1)
        else:
            return torch.cat([torch.sin(y), torch.cos(y)], -1)

class SineLayer(nn.Module):    
    def __init__(self, in_features, out_features, bias=False, is_first=False, is_res=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.is_res = is_res
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights(self.linear)
    
    def init_weights(self, layer):
        with torch.no_grad():
            if self.is_first:
                layer.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                layer.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,  np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        if self.is_res:
            return input + torch.sin(self.omega_0 * self.linear(input))
        else:
            return torch.sin(self.omega_0 * self.linear(input))

class LF_network(nn.Module):
    def __init__(self, encoding="DiscreteFourier", first_omega_0=30, hidden_omega_0=30., hidden_layers=8, alpha=0.5, in_feature_ratio=1, out_features=3, skips=[3], hidden_features=512, with_res=False, with_sigmoid=False, with_norm=False, multires_xy=10, multires_uv=6):
        super().__init__()

        
        self.with_res = with_res
        self.with_sigmoid = with_sigmoid
        self.D = hidden_layers + 2
        self.skips = skips
        self.encoding = encoding

        if encoding == "DiscreteFourier":
            self.uv_embedd =  PositionalEncoder(dim_in=1, num_freqs=multires_uv, include_input=False)
            self.N_uv = self.uv_embedd.embed_size() 
            self.xy_embedd =  PositionalEncoder(dim_in=1, num_freqs=multires_xy, include_input=False)
            self.N_xy = self.xy_embedd.embed_size() 
            in_features = self.N_xy * 2 + self.N_uv * 2
        elif encoding == "Gegenbauer":
            self.N_xy = int(in_feature_ratio * 240)
            self.xy_embedd = GegEmbedding(self.N_xy, alpha)
            self.N_uv = int(in_feature_ratio * 16)
            self.uv_embedd = GegEmbedding(self.N_uv, alpha)
            in_features = self.N_xy * 2 + self.N_uv * 2
        elif encoding == "Hashencoding":
            per_level_scale = np.exp2(np.log2(2048 * 1. / 16) / (16 - 1))
            self.encoder_xy = tcnn.Encoding(
                n_input_dims=2,
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": 16,
                    "n_features_per_level": 4,
                    "log2_hashmap_size": 19,
                    "base_resolution": 16,
                    "per_level_scale": per_level_scale,
                },
                dtype = torch.float32
            )
            self.encoder_uv = tcnn.Encoding(
                n_input_dims=2,
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": 16,
                    "n_features_per_level": 4,
                    "log2_hashmap_size": 19,
                    "base_resolution": 16,
                    "per_level_scale": per_level_scale,
                },
                dtype = torch.float32
            )
            in_features = 128
        else:
            # Notimplemented
            raiseExceptions(NotImplementedError)
        

        for i in range(hidden_layers+1):
            if i == 0:
                layer = SineLayer(in_features, hidden_features, bias=True, is_first=True, is_res=False, omega_0=first_omega_0)
            elif i in skips:
                layer = SineLayer(hidden_features + in_features, hidden_features, bias=True, is_first=False, is_res=False, omega_0=hidden_omega_0)
            else:
                layer = SineLayer(hidden_features, hidden_features, is_first=False, bias=True, is_res=self.with_res, omega_0=hidden_omega_0)
            if with_norm:
                layer = nn.Sequential(layer, nn.LayerNorm(hidden_features, elementwise_affine=True))
            setattr(self, f"encoding_{i+1}", layer)

        final_linear = nn.Linear(hidden_features, out_features, bias=True)
        
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,  np.sqrt(6 / hidden_features) / hidden_omega_0)
        setattr(self, f"encoding_{hidden_layers+2}", final_linear)
        

    def forward(self, x):
        # x: [B, 4]
        if self.encoding == "DiscreteFourier":
            x=x*2-1.
        if self.encoding != "Hashencoding":
            emb_x = torch.cat( [self.uv_embedd(x[:, 0]), self.uv_embedd(x[:, 1]), self.xy_embedd(x[:, 2]), self.xy_embedd(x[:, 3]) ], axis=1)
        else:
            emb_x = torch.cat( [self.encoder_uv(x[:, :2]), self.encoder_xy(x[:, 2:])], axis=1)
        out = emb_x
        for i in range(self.D):
            if i in self.skips:
                out = torch.cat([emb_x, out], -1)
            out = getattr(self, f"encoding_{i+1}")(out)
        return out
