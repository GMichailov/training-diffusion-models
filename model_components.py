import torch
import torch.nn as nn
import math

class MHACrossAttention(nn.Module):
    def __init__(self, num_heads, d_model, d_text, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_text, 2 * d_model)
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
    def __init__(self, num_heads, num_kv_heads, d_model, d_text, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.kv_head_dim = d_model // num_kv_heads
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_text, 2 * d_model)
        self._init()

    def forward(self, x, text_input, mask=None):
        batch, num_visual_tokens, d_model = x.shape
        num_text_tokens = text_input.shape[1]
        Q = self.q(x)
        KV = self.kv(text_input)
        K, V = KV.chunk(2, dim=-1)
        
        Q = Q.view(batch, num_visual_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, num_text_tokens, self.num_kv_heads, self.kv_head_dim).transpose(1, 2)
        V = V.view(batch, num_text_tokens, self.num_kv_heads, self.kv_head_dim).transpose(1, 2)

        n_repeats = self.num_heads // self.num_kv_heads
        K = K.repeat_interleave(n_repeats, dim=1)
        V = V.repeat_interleave(n_repeats, dim=1)
        attn_out = nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, num_visual_tokens, d_model)
        return self.dropout(attn_out)
    
    def _init(self):
        nn.init.trunc_normal_(self.q.weight, std=0.02)
        nn.init.trunc_normal_(self.kv.weight, std=0.02)

class SiluFFN(nn.Module):
    def __init__(self, io_dim, intermediate_size) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(io_dim, intermediate_size)
        self.up_proj = nn.Linear(io_dim, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, io_dim)
        self._init()

    def forward(self, x):
        gate = nn.functional.silu(self.gate_proj(x))
        return self.down_proj(gate * self.up_proj(x))
    
    def _init(self):
        nn.init.trunc_normal_(self.gate_proj.weight, std=0.02)
        nn.init.trunc_normal_(self.up_proj.weight, std=0.02)
        nn.init.trunc_normal_(self.down_proj.weight, std=0.02)

class TimestepEmbedding(nn.Module):
    def __init__(self, d_model, intermediate_size) -> None:
        super().__init__()
        self.silu_ffn = SiluFFN(d_model, intermediate_size)

    def forward(self, timesteps):
        """
        Takes a tensor of batch scalars representing the timestep.
        """
        half_dim = self.silu_ffn.gate_proj.in_features // 2
        device = timesteps.device
        # Rope
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim, device=device) / (half_dim - 1))
        args = timesteps.float()[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        # MLP
        return self.silu_ffn(emb)

class AdaLN_Zero(nn.Module):
    def __init__(self, d_model) -> None:
        super().__init__()
        self.act = nn.SiLU()
        self.output = nn.Linear(d_model, 3 * d_model)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        
    def forward(self, timestep_embedding):
        out = self.output(self.act(timestep_embedding))
        # Shift (beta), Scale (gamma), Gate (alpha)
        return out.chunk(3, dim=1)

class UNetDownSampler(nn.Module):
    def __init__(self, input_channels, d_text, num_heads, num_kv_heads) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.RMSNorm(input_channels, eps=1e-8)
        self.silu_ffn1 = SiluFFN(input_channels, input_channels * 4)
        self.conv2 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.RMSNorm(input_channels, eps=1e-8)
        self.attn = MQACrossAttention(num_heads=num_heads, d_text=d_text, num_kv_heads=num_kv_heads, d_model=input_channels)
        self.silu_ffn2 = SiluFFN(input_channels, input_channels * 4)
        self.downsample_conv = nn.Conv2d(in_channels=input_channels, out_channels=input_channels * 2, kernel_size=2, stride=2)

    def _conv2d_to_tokens(self, image_input):
        """Converts (batch, channels, height, width) to (batch, H*W, channels)"""
        batch, channels, height, width = image_input.shape
        return image_input.view(batch, channels, -1).transpose(1, 2)
    
    def _tokens_to_conv2d(self, image_input, height, width):
        batch, tokens, channels = image_input.shape
        return image_input.transpose(1, 2).view(batch, channels, height, width)

    def forward(self, image_input, timestep_embeddings, text_input):
        _, _, height, width = image_input.shape
        residual = image_input
        image_input = self.conv1(image_input)
        image_input = nn.functional.silu(image_input)
        image_input = self._conv2d_to_tokens(image_input)
        image_input = self.norm1(image_input)
        image_input = self.silu_ffn1(image_input)
        image_input = self._tokens_to_conv2d(image_input, height, width)
        image_input = self.conv2(image_input)
        image_input = nn.functional.silu(image_input)
        image_input = image_input + timestep_embeddings[:, :, None, None]
        image_input = self._conv2d_to_tokens(image_input)
        image_input = self.norm2(image_input)
        image_input = self.attn(image_input, text_input)
        image_input = self.silu_ffn2(image_input)
        image_input = self._tokens_to_conv2d(image_input, height, width)
        image_input = image_input + residual
        skip = image_input
        image_input = self.downsample_conv(image_input)
        return image_input, skip

class UNetUpSampler(nn.Module):
    def __init__(self, input_channels, d_text, num_heads, num_kv_heads) -> None:
        super().__init__()
        self.upsample_conv = nn.ConvTranspose2d(in_channels=input_channels * 2, out_channels=input_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.RMSNorm(input_channels, eps=1e-8)
        self.silu_ffn1 = SiluFFN(input_channels, input_channels * 4)
        self.conv2 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.RMSNorm(input_channels, eps=1e-8)
        self.attn = MQACrossAttention(num_heads=num_heads, num_kv_heads=num_kv_heads, d_model=input_channels, d_text=d_text)
        self.silu_ffn2 = SiluFFN(input_channels, input_channels * 4)

    def _conv2d_to_tokens(self, image_input):
        """Converts (batch, channels, height, width) to (batch, H*W, channels)"""
        batch, channels, height, width = image_input.shape
        return image_input.view(batch, channels, -1).transpose(1, 2)
    
    def _tokens_to_conv2d(self, image_input, height, width):
        batch, tokens, channels = image_input.shape
        return image_input.transpose(1, 2).view(batch, channels, height, width)

    def _match_hw(self, image_input, skip):
        if skip is None:
            return None
        if skip.shape[-2:] != image_input.shape[-2:]:
            skip = nn.functional.interpolate(skip, image_input.shape[-2:], mode="nearest")
        return skip

    def forward(self, image_input, timestep_embeddings, text_input, skip=None):
        image_input = self.upsample_conv(image_input)

        skip = self._match_hw(image_input, skip)
        if skip is not None:
            image_input = image_input + skip
        _, _, height, width = image_input.shape
        residual = image_input

        image_input = self.conv1(image_input)
        image_input = nn.functional.silu(image_input)
        image_input = self._conv2d_to_tokens(image_input)
        image_input = self.norm1(image_input)
        image_input = self.silu_ffn1(image_input)
        image_input = self._tokens_to_conv2d(image_input, height, width)
        image_input = self.conv2(image_input)
        image_input = nn.functional.silu(image_input)
        image_input = image_input + timestep_embeddings[:, :, None, None]
        image_input = self._conv2d_to_tokens(image_input)
        image_input = self.norm2(image_input)
        image_input = self.attn(image_input, text_input)
        image_input = self.silu_ffn2(image_input)
        image_input = self._tokens_to_conv2d(image_input, height, width)
        image_input = image_input + residual
        return image_input

class UNetModel(nn.Module):
    def __init__(self, num_layers, input_channels, d_text, num_heads, num_kv_heads) -> None:
        super().__init__()
        self.timestep_embedding = TimestepEmbedding(d_model=input_channels, intermediate_size=4 * input_channels)
        self.channel_counts = []
        current_channels = input_channels
        for _ in range(num_layers):
            self.channel_counts.append(current_channels)
            current_channels *= 2
        self.downsampling_layers = nn.ModuleList([
            UNetDownSampler(input_channels=ch, d_text=d_text, num_heads=num_heads, num_kv_heads=num_kv_heads)
            for ch in self.channel_counts
        ])
        self.upsampling_layers = nn.ModuleList([
            UNetUpSampler(input_channels=ch, d_text=d_text, num_heads=num_heads, num_kv_heads=num_kv_heads)
            for ch in reversed(self.channel_counts)
        ])

    def forward(self, image_input, timesteps, text_input):
        skip_connections = []
        timestep_embeddings = self.timestep_embedding(timesteps)
        for downsampling_layer in self.downsampling_layers:
            image_input, skip = downsampling_layer(image_input, timestep_embeddings, text_input)
            skip_connections.append(skip)
        for upsampling_layer in self.upsampling_layers:
            skip = skip_connections.pop() if len(skip_connections) else None
            image_input = upsampling_layer(image_input, timestep_embeddings, text_input, skip=skip)
        return image_input

class PixelTransformerBlock(nn.Module):
    pass

class DiffusionTransformerBlock(nn.Module):
    pass

class PixelDit(nn.Module):
    pass