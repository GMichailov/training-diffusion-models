import torch
import torch.nn as nn
import math

INV_SQRT2_SCALING = 0.70710678

class RoPEQK(nn.Module):
    def __init__(self, base=10000):
        super().__init__()
        self.base = base

    def _rotate_half(self, x):
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).reshape_as(x)
        
    def forward(self, q, k):
        device = q.device
        seq_len = q.shape[-2]
        dim = q.shape[-1]
        freqs = torch.exp(-math.log(self.base) * torch.arange(0, dim, 2, device=device) / dim)
        positions = torch.arange(seq_len, device=device)
        angles = positions[:, None] * freqs[None, :]
        sin = angles.sin()[None, None, :, :].to(dtype=q.dtype)
        cos = angles.cos()[None, None, :, :].to(dtype=q.dtype)
        q_rot = q * cos.repeat_interleave(2, dim=-1) + self._rotate_half(q) * sin.repeat_interleave(2, dim=-1)
        k_rot = k * cos.repeat_interleave(2, dim=-1) + self._rotate_half(k) * sin.repeat_interleave(2, dim=-1)
        return q_rot, k_rot

class RoPEQ(nn.Module):
    def __init__(self, base=10000):
        super().__init__()
        self.base = base

    def _rotate_half(self, x):
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).reshape_as(x)
        
    def forward(self, q):
        device = q.device
        seq_len = q.shape[-2]
        dim = q.shape[-1]
        freqs = torch.exp(-math.log(self.base) * torch.arange(0, dim, 2, device=device) / dim)
        positions = torch.arange(seq_len, device=device)
        angles = positions[:, None] * freqs[None, :]
        sin = angles.sin()[None, None, :, :].to(dtype=q.dtype)
        cos = angles.cos()[None, None, :, :].to(dtype=q.dtype)
        q_rot = q * cos.repeat_interleave(2, dim=-1) + self._rotate_half(q) * sin.repeat_interleave(2, dim=-1)
        return q_rot

class MHACrossAttention(nn.Module):
    def __init__(self, num_heads, d_model, d_text, dropout=0.0, dtype=torch.bfloat16):
        super().__init__()
        self.dtype = dtype
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q = nn.Linear(d_model, d_model, dtype=dtype)
        self.kv = nn.Linear(d_text, 2 * d_model, dtype=dtype)
        self._init()

    def forward(self, x, text_input, mask=None):
        batch, num_visual_tokens, d_model = x.shape
        num_text_tokens = text_input.shape[1]
        Q = self.q(x)
        KV = self.kv(text_input)
        K, V = KV.chunk(2, dim=-1)
        # Reshape from B, num_visual_tokens, d_model for Q and B, num_textual_tokens, d_model to B, num_heads, N, head_dim
        Q = Q.view(batch, num_visual_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, num_text_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, num_text_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        attn_out = nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, num_visual_tokens, d_model)
        return self.dropout(attn_out)
    
    def _init(self):
        nn.init.trunc_normal_(self.q.weight, std=0.02)
        nn.init.trunc_normal_(self.kv.weight, std=0.02)

class MQACrossAttention(nn.Module):
    def __init__(self, num_heads, num_kv_heads, d_model, d_text, dropout=0.0, dtype=torch.bfloat16):
        super().__init__()
        self.dtype = dtype
        self.text_norm = nn.LayerNorm(d_text, eps=1e-5)
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.kv_heads_dim = num_kv_heads * self.head_dim
        self.q = nn.Linear(d_model, d_model, bias=False, dtype=dtype)
        self.kv = nn.Linear(d_text, 2 * self.kv_heads_dim, bias=False, dtype=dtype)
        self.out = nn.Linear(d_model, d_model, dtype=dtype)
        self._init()

    def forward(self, x, text_input, mask=None):
        text_input = self.text_norm(text_input)
        batch, num_visual_tokens, d_model = x.shape
        num_text_tokens = text_input.shape[1]
        Q = self.q(x)
        KV = self.kv(text_input)
        K, V = KV.chunk(2, dim=-1)
        
        Q = Q.view(batch, num_visual_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, num_text_tokens, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, num_text_tokens, self.num_kv_heads, self.head_dim).transpose(1, 2)

        n_repeats = self.num_heads // self.num_kv_heads
        K = K.repeat_interleave(n_repeats, dim=1)
        V = V.repeat_interleave(n_repeats, dim=1)
        attn_out = nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, num_visual_tokens, d_model)
        attn_out = self.out(attn_out)
        return self.dropout(attn_out)
    
    def _init(self):
        nn.init.trunc_normal_(self.q.weight, std=0.02)
        nn.init.trunc_normal_(self.kv.weight, std=0.02)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        nn.init.ones_(self.text_norm.weight)

class MHA(nn.Module):
    def __init__(self, num_heads, d_model, dropout = 0.0) -> None:
        super().__init__()
        self.rope = RoPEQK()
        self.Wqkv = nn.Linear(d_model, 3*d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.dropout = dropout
        self._init()

    def forward(self, x):
        B, S, D = x.shape
        QKV = self.Wqkv(x)
        Q, K, V = QKV.chunk(3, dim=-1) # [B, S, d_model]
        Q = Q.view(B, S, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(B, S, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(B, S, self.num_heads, self.d_head).transpose(1, 2)
        Q, K = self.rope(Q, K)
        attn_out = nn.functional.scaled_dot_product_attention(Q, K, V, None, self.dropout)
        attn_out = attn_out.transpose(1,2).view(B, S, D)
        return self.Wo(attn_out)

    def _init(self):
        nn.init.trunc_normal_(self.Wqkv.weight)
        nn.init.trunc_normal_(self.Wo.weight)
        nn.init.zeros_(self.Wqkv.bias)
        nn.init.zeros_(self.Wo.bias)

class SiluFFN(nn.Module):
    def __init__(self, io_dim, intermediate_size, dtype=torch.bfloat16) -> None:
        super().__init__()
        self.dtype = dtype
        self.gate_proj = nn.Linear(io_dim, intermediate_size, dtype=dtype)
        self.up_proj = nn.Linear(io_dim, intermediate_size, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, io_dim, dtype=dtype)
        self._init()

    def forward(self, x):
        gate = nn.functional.silu(self.gate_proj(x))
        return self.down_proj(gate * self.up_proj(x))
    
    def _init(self):
        nn.init.trunc_normal_(self.gate_proj.weight, std=0.02)
        nn.init.trunc_normal_(self.up_proj.weight, std=0.02)
        nn.init.zeros_(self.down_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)
        nn.init.zeros_(self.up_proj.bias)
        nn.init.zeros_(self.down_proj.bias)

class TimestepEmbedding(nn.Module):
    def __init__(self, d_model, intermediate_size, dtype=torch.bfloat16) -> None:
        super().__init__()
        self.silu_ffn = SiluFFN(d_model, intermediate_size)
        self.dtype = dtype

    def forward(self, timesteps):
        """
        Takes a tensor of batch scalars representing the timestep.
        """
        half_dim = self.silu_ffn.gate_proj.in_features // 2
        device = timesteps.device
        # Rope
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim, device=device) / (half_dim - 1))
        args = timesteps.float()[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1).to(dtype=self.dtype)
        # MLP
        return self.silu_ffn(emb)

class AdaLN_Zero(nn.Module):
    def __init__(self, d_cond, d_model, triplets, dtype=torch.bfloat16) -> None:
        super().__init__()
        self.lin1 = nn.Linear(d_cond, d_model)
        self.act = nn.SiLU()
        self.output = nn.Linear(d_model, triplets * 3 * d_model)
        self.triplets = triplets
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        nn.init.trunc_normal_(self.lin1.weight)
        nn.init.zeros_(self.lin1.bias)
        
    def forward(self, cond):
        out = self.output(self.act(self.lin1(cond)))
        # Gate (alpha) , Shift (beta), Scale (gamma)
        return out.chunk(3 * self.triplets, dim=1)

class PixelPatchEmbed(nn.Module):
    def __init__(self, in_channels, d_model):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, d_model)
        self.act = nn.SiLU()
        self.out = nn.Linear (d_model, d_model)
        self.d_model = d_model
        nn.init.trunc_normal_(self.lin1.weight, std=0.02)
        nn.init.zeros_(self.lin1.bias)
        nn.init.trunc_normal_(self.out.weight, std=0.02)
        nn.init.zeros_(self.out.bias)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H*W).transpose(1, 2)
        return self.out(self.act(self.lin1(x))), H, W
    
class PixelUnpatchEmbed(nn.Module):
    def __init__(self, out_channels, d_model):
        super().__init__()
        self.lin1 = nn.Linear(d_model, out_channels)
        self.act = nn.SiLU()
        self.out = nn.Linear (out_channels, out_channels)
        self.out_channels = out_channels
        nn.init.trunc_normal_(self.lin1.weight, std=0.02)
        nn.init.zeros_(self.lin1.bias)
        nn.init.trunc_normal_(self.out.weight, std=0.02)
        nn.init.zeros_(self.out.bias)
    
    def forward(self, x, H, W):
        B, _, _ = x.shape
        x = self.out(self.act(self.lin1(x)))
        x = x.transpose(1,2).view(B, self.out_channels, H, W)
        return x

class UNetDownSampler(nn.Module):
    def __init__(self, input_channels, d_text, num_heads, num_kv_heads, d_timesteps, fan_out_factor, dtype=torch.bfloat16) -> None:
        super().__init__()
        self.dtype = dtype
        self.timestep_projector = nn.Linear(d_timesteps, input_channels)
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, stride=1, padding=1, dtype=dtype)
        self.norm1 = nn.RMSNorm(input_channels, eps=1e-5, dtype=torch.float32)
        self.silu_ffn1 = SiluFFN(input_channels, input_channels * fan_out_factor, dtype=dtype)
        self.conv2 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, stride=1, padding=1, dtype=dtype)
        self.norm2 = nn.RMSNorm(input_channels, eps=1e-5, dtype=torch.float32)
        self.attn = MQACrossAttention(num_heads=num_heads, d_text=d_text, num_kv_heads=num_kv_heads, d_model=input_channels, dtype=dtype)
        self.norm3 = nn.RMSNorm(input_channels, eps=1e-5, dtype=torch.float32)
        self.silu_ffn2 = SiluFFN(input_channels, input_channels * fan_out_factor, dtype=dtype)
        self.downsample_conv = nn.Conv2d(in_channels=input_channels, out_channels=input_channels * 2, kernel_size=2, stride=2, dtype=dtype)
        self._init()

    def _conv2d_to_tokens(self, image_input):
        """Converts (batch, channels, height, width) to (batch, H*W, channels)"""
        batch, channels, height, width = image_input.shape
        return image_input.reshape(batch, channels, -1).transpose(1, 2)
    
    def _tokens_to_conv2d(self, image_input, height, width):
        batch, tokens, channels = image_input.shape
        return image_input.transpose(1, 2).reshape(batch, channels, height, width)

    def forward(self, image_input, timestep_embeddings, text_input):
        timestep_embeddings = self.timestep_projector(timestep_embeddings)
        _, _, height, width = image_input.shape
        residual = image_input
        image_input = nn.functional.silu(self.conv1(image_input))
        image_input = self._conv2d_to_tokens(image_input)
        image_input_norm = self.norm1(image_input)
        image_input = image_input + self.silu_ffn1(image_input_norm)
        image_input = self._tokens_to_conv2d(image_input, height, width)
        image_input = nn.functional.silu(self.conv2(image_input))
        image_input = image_input + timestep_embeddings[:, :, None, None]
        image_input = self._conv2d_to_tokens(image_input)

        image_input_norm = self.norm2(image_input)
        image_input = image_input + self.attn(image_input_norm, text_input)
        image_input_norm = self.norm3(image_input)
        image_input = image_input + self.silu_ffn2(image_input_norm)
        image_input = self._tokens_to_conv2d(image_input, height, width)
        image_input = (image_input + residual) * INV_SQRT2_SCALING
        skip = image_input
        image_input = self.downsample_conv(image_input)
        return image_input, skip

    def _init(self):
        nn.init.trunc_normal_(self.timestep_projector.weight, std=0.02)
        nn.init.zeros_(self.timestep_projector.bias)
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="conv2d")
        nn.init.zeros_(self.conv1.bias) # type: ignore
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_out", nonlinearity="conv2d")
        nn.init.zeros_(self.conv2.bias) # type: ignore
        nn.init.kaiming_normal_(self.downsample_conv.weight, mode="fan_out", nonlinearity="conv2d")
        nn.init.zeros_(self.downsample_conv.bias) # type: ignore
        nn.init.ones_(self.norm1.weight)
        nn.init.ones_(self.norm2.weight)
        nn.init.ones_(self.norm3.weight)

class UNetDownSamplerV2(nn.Module):
    def __init__(self, input_channels, d_text, num_heads, num_kv_heads, d_timesteps, fan_out_factor, dtype=torch.bfloat16) -> None:
        """Uses AdaLnZero for timestamp + text embedding and assume concat rather than channel wise add and proj."""
        super().__init__()
        self.dtype = dtype
        self.AdaLnZero = AdaLN_Zero(d_text + d_timesteps, input_channels, 2)
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, stride=1, padding=1, dtype=dtype)
        self.norm1 = nn.RMSNorm(input_channels, eps=1e-5, dtype=torch.bfloat16)
        self.silu_ffn1 = SiluFFN(input_channels, input_channels * fan_out_factor, dtype=dtype)
        self.conv2 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, stride=1, padding=1, dtype=dtype)
        self.norm2 = nn.RMSNorm(input_channels, eps=1e-5, dtype=torch.bfloat16)
        self.attn = MQACrossAttention(num_heads=num_heads, d_text=d_text + d_timesteps, num_kv_heads=num_kv_heads, d_model=input_channels, dtype=dtype)
        self.silu_ffn2 = SiluFFN(input_channels, input_channels * fan_out_factor, dtype=dtype)
        self.downsample_conv = nn.Conv2d(in_channels=input_channels, out_channels=input_channels * 2, kernel_size=2, stride=2, dtype=dtype)
        self._init()

    def _conv2d_to_tokens(self, image_input):
        """Converts (batch, channels, height, width) to (batch, H*W, channels)"""
        batch, channels, height, width = image_input.shape
        return image_input.reshape(batch, channels, -1).transpose(1, 2)
    
    def _tokens_to_conv2d(self, image_input, height, width):
        batch, tokens, channels = image_input.shape
        return image_input.transpose(1, 2).reshape(batch, channels, height, width)

    def forward(self, image_input, cond):
        res_gate_alpha_1, shift_beta_1, scale_gamma_1, res_gate_alpha_2, shift_beta_2, scale_gamma_2 = self.AdaLnZero(cond)
        _, _, height, width = image_input.shape
        residual = image_input
        image_input = self.conv1(image_input)
        image_input = nn.functional.silu(image_input)
        image_input = self._conv2d_to_tokens(image_input)
        image_input = self.norm1(image_input)
        image_input = scale_gamma_1 * image_input + shift_beta_1
        image_input = self.silu_ffn1(image_input)
        image_input = image_input * res_gate_alpha_1 + residual
        residual = image_input

        image_input = self._tokens_to_conv2d(image_input, height, width)
        image_input = self.conv2(image_input)
        image_input = nn.functional.silu(image_input)
        image_input = self._conv2d_to_tokens(image_input)
        image_input = self.norm2(image_input)
        image_input = scale_gamma_2 * image_input + shift_beta_2
        image_input = self.attn(image_input, cond)
        image_input = self.silu_ffn2(image_input)
        image_input = self._tokens_to_conv2d(image_input, height, width)
        image_input = image_input * res_gate_alpha_2 + residual
        skip = image_input
        image_input = self.downsample_conv(image_input)
        return image_input, skip

    def _init(self):
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="conv2d")
        nn.init.zeros_(self.conv1.bias) # type: ignore
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_out", nonlinearity="conv2d")
        nn.init.zeros_(self.conv2.bias) # type: ignore
        nn.init.kaiming_normal_(self.downsample_conv.weight, mode="fan_out", nonlinearity="conv2d")
        nn.init.zeros_(self.downsample_conv.bias) # type: ignore
        nn.init.ones_(self.norm1.weight)
        nn.init.ones_(self.norm2.weight)

class UNetUpSampler(nn.Module):
    def __init__(self, input_channels, d_text, num_heads, num_kv_heads, d_timesteps, fan_out_factor, dtype=torch.bfloat16) -> None:
        super().__init__()
        self.dtype = dtype
        self.timestep_projector = nn.Linear(d_timesteps, input_channels)
        self.upsample_conv = nn.ConvTranspose2d(in_channels=input_channels * 2, out_channels=input_channels, kernel_size=2, stride=2, dtype=dtype)
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, stride=1, padding=1, dtype=dtype)
        self.norm1 = nn.RMSNorm(input_channels, eps=1e-5, dtype=torch.float32)
        self.silu_ffn1 = SiluFFN(input_channels, input_channels * fan_out_factor, dtype=dtype)
        self.conv2 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, stride=1, padding=1, dtype=dtype)
        self.norm2 = nn.RMSNorm(input_channels, eps=1e-5, dtype=torch.float32)
        self.attn = MQACrossAttention(num_heads=num_heads, num_kv_heads=num_kv_heads, d_model=input_channels, d_text=d_text, dtype=dtype)
        self.norm3 = nn.RMSNorm(input_channels, eps=1e-5, dtype=torch.float32)
        self.silu_ffn2 = SiluFFN(input_channels, input_channels * fan_out_factor, dtype=dtype)
        self._init()

    def _conv2d_to_tokens(self, image_input):
        """Converts (batch, channels, height, width) to (batch, H*W, channels)"""
        batch, channels, height, width = image_input.shape
        return image_input.reshape(batch, channels, -1).transpose(1, 2)
    
    def _tokens_to_conv2d(self, image_input, height, width):
        batch, tokens, channels = image_input.shape
        return image_input.transpose(1, 2).reshape(batch, channels, height, width)

    def _match_hw(self, image_input, skip):
        if skip is None:
            return None
        if skip.shape[-2:] != image_input.shape[-2:]:
            skip = nn.functional.interpolate(skip, image_input.shape[-2:], mode="nearest")
        return skip

    def forward(self, image_input, timestep_embeddings, text_input, skip=None):
        timestep_embeddings = self.timestep_projector(timestep_embeddings)
        image_input = self.upsample_conv(image_input)

        skip = self._match_hw(image_input, skip)
        if skip is not None:
            image_input = (image_input + skip) * INV_SQRT2_SCALING
        _, _, height, width = image_input.shape
        residual = image_input
        image_input = nn.functional.silu(self.conv1(image_input))
        image_input = self._conv2d_to_tokens(image_input)
        image_input_norm = self.norm1(image_input)
        image_input = image_input + self.silu_ffn1(image_input_norm)
        image_input = self._tokens_to_conv2d(image_input, height, width)
        image_input = nn.functional.silu(self.conv2(image_input))
        image_input = image_input + timestep_embeddings[:, :, None, None]
        image_input = self._conv2d_to_tokens(image_input)
        image_input_norm = self.norm2(image_input)
        image_input = image_input + self.attn(image_input_norm, text_input)
        image_input_norm = self.norm3(image_input)
        image_input = image_input + self.silu_ffn2(image_input_norm)
        image_input = self._tokens_to_conv2d(image_input, height, width)
        image_input = (image_input + residual) * INV_SQRT2_SCALING
        return image_input

    def _init(self):
        nn.init.trunc_normal_(self.timestep_projector.weight, std=0.02)
        nn.init.zeros_(self.timestep_projector.bias)
        nn.init.kaiming_normal_(self.upsample_conv.weight, mode="fan_out", nonlinearity="conv2d")
        nn.init.zeros_(self.upsample_conv.bias) # type: ignore
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="conv2d")
        nn.init.zeros_(self.conv1.bias) # type: ignore
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_out", nonlinearity="conv2d")
        nn.init.zeros_(self.conv2.bias) # type: ignore
        nn.init.ones_(self.norm1.weight)
        nn.init.ones_(self.norm2.weight)
        nn.init.ones_(self.norm3.weight)

class UNetModel(nn.Module):
    def __init__(self, num_layers, raw_input_channels, hidden_size_channels, d_text, num_heads, num_kv_heads, fan_out_factor, dtype=torch.bfloat16) -> None:
        super().__init__()
        self.dtype = dtype
        self.stem = nn.Conv2d(raw_input_channels, hidden_size_channels, kernel_size=3, padding=1)
        self.timestep_embedding = TimestepEmbedding(d_model=hidden_size_channels, intermediate_size=4 * hidden_size_channels, dtype=dtype)
        self.d_timesteps = hidden_size_channels
        self.channel_counts = []
        current_channels = hidden_size_channels
        for _ in range(num_layers):
            self.channel_counts.append(current_channels)
            current_channels *= 2
        self.downsampling_layers = nn.ModuleList([
            UNetDownSampler(
                input_channels=ch, 
                d_text=d_text, 
                num_heads=num_heads, 
                num_kv_heads=num_kv_heads, 
                d_timesteps = self.d_timesteps,dtype=dtype,
                fan_out_factor=fan_out_factor
            )
            for ch in self.channel_counts
        ])
        self.upsampling_layers = nn.ModuleList([
            UNetUpSampler(
                input_channels=ch, 
                d_text=d_text, 
                num_heads=num_heads, 
                num_kv_heads=num_kv_heads, 
                d_timesteps = self.d_timesteps,
                fan_out_factor=fan_out_factor, 
                dtype=dtype
            )
            for ch in reversed(self.channel_counts)
        ])
        self.out_norm = nn.GroupNorm(num_groups=1, num_channels=hidden_size_channels,eps=1e-5)
        self.out_conv = nn.Conv2d(hidden_size_channels, out_channels=raw_input_channels, kernel_size=3, padding=1, stride=1, dtype=dtype)
        self._init()

    def forward(self, image_input, timesteps, text_input):
        skip_connections = []
        timestep_embeddings = self.timestep_embedding(timesteps)
        image_input = self.stem(image_input)
        for downsampling_layer in self.downsampling_layers:
            image_input, skip = downsampling_layer(image_input, timestep_embeddings, text_input)
            skip_connections.append(skip)
        for upsampling_layer in self.upsampling_layers:
            skip = skip_connections.pop() if len(skip_connections) else None
            image_input = upsampling_layer(image_input, timestep_embeddings, text_input, skip=skip)
        return self.out_conv(self.out_norm(image_input))
    
    def _init(self):
        nn.init.kaiming_normal_(self.out_conv.weight, mode="fan_out", nonlinearity="conv2d")
        nn.init.zeros_(self.out_conv.bias) #type: ignore
        nn.init.ones_(self.out_norm.weight)
        nn.init.zeros_(self.out_norm.bias)

class PixelTransformerBlock(nn.Module):
    def __init__(self, num_heads, d_model, d_latent, d_cond, scale_factor=4) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(normalized_shape=d_model, eps=1e-6)
        self.AdaLnZero = AdaLN_Zero(d_cond, d_model, 2)
        self.linear_compress = nn.Linear(d_model, d_latent)
        self.mhsa = MHA(num_heads, d_latent)
        self.linear_expand = nn.Linear(d_latent, d_model)
        self.norm2 = nn.RMSNorm(normalized_shape=d_model, eps=1e-6)
        self.ffn1 = SiluFFN(d_model, scale_factor * d_model)

    def forward(self, x, condition_embedding):
        res = x
        res_gate_alpha_1, shift_beta_1, scale_gamma_1, res_gate_alpha_2, shift_beta_2, scale_gamma_2 = self.AdaLnZero(condition_embedding)
        x = self.norm1(x)
        x = x * scale_gamma_1 + shift_beta_1
        x = self.linear_compress(x)
        x = self.mhsa(x)
        x = self.linear_expand(x)
        x = res + res_gate_alpha_1 * x
        res = x
        x = self.norm2(x)
        x = x * scale_gamma_2 + shift_beta_2
        x = self.ffn1(x)
        x = res + res_gate_alpha_2 * x
        return x

class DiffusionTransformerBlock(nn.Module):
    def __init__(self, num_heads, d_model, d_latent, d_cond, scale_factor=4) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(normalized_shape=d_model, eps=1e-6)
        self.AdaLnZero = AdaLN_Zero(d_cond, d_model, 2)
        self.mhsa = MHA(num_heads, d_model)
        self.norm2 = nn.RMSNorm(normalized_shape=d_model, eps=1e-6)
        self.ffn1 = SiluFFN(d_model, scale_factor * d_model)

    def forward(self, x, condition_embeddings):
        res = x
        res_gate_alpha_1, shift_beta_1, scale_gamma_1, res_gate_alpha_2, shift_beta_2, scale_gamma_2 = self.AdaLnZero(condition_embeddings)
        x = self.norm1(x)
        x = x * scale_gamma_1 + shift_beta_1
        x = self.mhsa(x)
        x = self.ffn1(x)
        x = res + res_gate_alpha_1 * x
        res = x
        x = self.norm2(x)
        x = x * scale_gamma_2 + shift_beta_2
        x = self.ffn1(x)
        x = res + res_gate_alpha_2 * x
        return x

class PixelDit(nn.Module):
    def __init__(self, n, m, in_channels, num_heads, d_model, d_latent, d_cond, scale_factor=4) -> None:
        super().__init__()
        self.patch_embed = PixelPatchEmbed(in_channels, d_model)
        self.dits = nn.ModuleList([
            DiffusionTransformerBlock(num_heads, d_model, d_latent, d_cond)
            for _ in range(n)
        ])
        self.pits = nn.ModuleList([
            PixelTransformerBlock(num_heads, d_model, d_latent, d_cond)
            for _ in range(m)
        ])
        self.norm1 = nn.RMSNorm(normalized_shape=d_model, eps=1e-6)
        self.AdaLnZero = AdaLN_Zero(d_cond, d_model, 1)
        self.out_proj = PixelUnpatchEmbed(in_channels, d_model)

    def forward(self, x_image, cond):
        x, H, W = self.patch_embed(x_image)
        for dit in self.dits:
            x = dit(x, cond)
        for pit in self.pits:
            x = pit(x, cond)
        _, shift_beta, scale_gamma = self.AdaLnZero(cond)
        x = self.norm1(x)
        x = x * scale_gamma + shift_beta
        return x + self.out_proj(x, H, W)

       

