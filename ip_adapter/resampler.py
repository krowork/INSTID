# modified from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Configure logging
logger = logging.getLogger(__name__)

# FFN
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )
    
    
def reshape_tensor(x, heads):
    bs, length, width = x.shape
    #(bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)


    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)
        
        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        
        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1) # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v
        
        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)


class Resampler(nn.Module):
    """
    Resampler module with improved memory management and error handling.
    """
    
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
        max_seq_len=257,  # 256 + 1 for CLS token
    ):
        """Initialize Resampler with validation."""
        try:
            super().__init__()
            
            self.validate_inputs(dim, depth, dim_head, heads, num_queries, embedding_dim, output_dim, ff_mult)
            
            self.num_queries = num_queries
            self.embedding_dim = embedding_dim
            self.output_dim = output_dim
            
            self.queries = nn.Parameter(torch.randn(1, num_queries, dim))
            
            self.layers = nn.ModuleList([])
            for _ in range(depth):
                self.layers.append(
                    nn.ModuleList([
                        EfficientCrossAttention(dim=dim, dim_head=dim_head, heads=heads, max_seq_len=max_seq_len),
                        FeedForward(dim=dim, mult=ff_mult)
                    ])
                )
                
            self.proj_in = nn.Linear(embedding_dim, dim)
            self.proj_out = nn.Linear(dim, output_dim)
            self.norm_out = nn.LayerNorm(output_dim)
            
            logger.info(f"Initialized Resampler with {depth} layers")
        except Exception as e:
            logger.error(f"Error initializing Resampler: {str(e)}")
            raise RuntimeError(f"Failed to initialize Resampler: {str(e)}")
    
    def validate_inputs(self, dim, depth, dim_head, heads, num_queries, embedding_dim, output_dim, ff_mult):
        """Validate initialization parameters."""
        if not all(isinstance(x, int) for x in [dim, depth, dim_head, heads, num_queries, embedding_dim, output_dim, ff_mult]):
            raise ValueError("All dimension parameters must be integers")
        if any(x <= 0 for x in [dim, depth, dim_head, heads, num_queries, embedding_dim, output_dim, ff_mult]):
            raise ValueError("All dimension parameters must be positive")
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with improved error handling and memory management.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embedding_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_queries, output_dim)
        """
        try:
            # Input validation
            if not isinstance(x, torch.Tensor):
                raise ValueError("Input must be a torch.Tensor")
            if x.dim() != 3:
                raise ValueError(f"Expected 3D input tensor, got {x.dim()}D")
            if x.size(-1) != self.embedding_dim:
                raise ValueError(f"Expected input dimension {self.embedding_dim}, got {x.size(-1)}")
            
            # Project input
            x = self.proj_in(x)
            
            # Prepare queries
            queries = self.queries.expand(x.size(0), -1, -1)
            
            # Process through transformer layers
            for attn, ff in self.layers:
                queries = attn(queries, x) + queries
                queries = ff(queries) + queries
            
            # Project output
            output = self.proj_out(queries)
            output = self.norm_out(output)
            
            return output
            
        except Exception as e:
            logger.error(f"Error in Resampler forward pass: {str(e)}")
            raise RuntimeError(f"Forward pass failed: {str(e)}")

class EfficientCrossAttention(nn.Module):
    """
    Memory-efficient cross attention implementation.
    """
    
    def __init__(self, dim, dim_head=64, heads=8, max_seq_len=512):
        """Initialize attention module with validation."""
        try:
            super().__init__()
            
            self.validate_inputs(dim, dim_head, heads)
            
            self.scale = dim_head ** -0.5
            self.heads = heads
            self.max_seq_len = max_seq_len
            inner_dim = dim_head * heads
            
            self.norm = nn.LayerNorm(dim)
            self.to_q = nn.Linear(dim, inner_dim, bias=False)
            self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
            self.to_out = nn.Linear(inner_dim, dim, bias=False)
            
        except Exception as e:
            logger.error(f"Error initializing EfficientCrossAttention: {str(e)}")
            raise RuntimeError(f"Failed to initialize attention module: {str(e)}")
    
    def validate_inputs(self, dim, dim_head, heads):
        """Validate initialization parameters."""
        if not all(isinstance(x, int) for x in [dim, dim_head, heads]):
            raise ValueError("All dimension parameters must be integers")
        if any(x <= 0 for x in [dim, dim_head, heads]):
            raise ValueError("All dimension parameters must be positive")
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")
    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with memory optimization and error handling.
        """
        try:
            # Input validation
            if not isinstance(x, torch.Tensor) or not isinstance(context, torch.Tensor):
                raise ValueError("Inputs must be torch.Tensor")
            if x.dim() != 3 or context.dim() != 3:
                raise ValueError("Expected 3D tensors")
            
            batch_size, seq_len, _ = x.shape
            
            # Apply layer normalization
            context = self.norm(context)
            
            # Project queries, keys, and values
            q = self.to_q(x)
            k, v = self.to_kv(context).chunk(2, dim=-1)
            
            # Split heads
            q, k, v = map(
                lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads),
                (q, k, v)
            )
            
            # Compute attention with memory efficient implementation
            with torch.cuda.amp.autocast(enabled=False):
                q = q.float()
                k = k.float()
                
                sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
                
                # Apply attention
                attn = F.softmax(sim, dim=-1)
                out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
            
            # Combine heads and project
            out = rearrange(out, 'b h n d -> b n (h d)')
            out = self.to_out(out)
            
            return out
            
        except Exception as e:
            logger.error(f"Error in attention forward pass: {str(e)}")
            raise RuntimeError(f"Attention computation failed: {str(e)}")

class FeedForward(nn.Module):
    """
    Memory-efficient feed-forward network implementation.
    """
    
    def __init__(self, dim, mult=4):
        """Initialize feed-forward module with validation."""
        try:
            super().__init__()
            
            self.validate_inputs(dim, mult)
            
            self.norm = nn.LayerNorm(dim)
            self.ff = nn.Sequential(
                nn.Linear(dim, dim * mult),
                nn.GELU(),
                nn.Linear(dim * mult, dim)
            )
            
        except Exception as e:
            logger.error(f"Error initializing FeedForward: {str(e)}")
            raise RuntimeError(f"Failed to initialize feed-forward module: {str(e)}")
    
    def validate_inputs(self, dim, mult):
        """Validate initialization parameters."""
        if not isinstance(dim, int) or not isinstance(mult, int):
            raise ValueError("dim and mult must be integers")
        if dim <= 0 or mult <= 0:
            raise ValueError("dim and mult must be positive")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with memory optimization and error handling.
        """
        try:
            # Input validation
            if not isinstance(x, torch.Tensor):
                raise ValueError("Input must be a torch.Tensor")
            if x.dim() != 3:
                raise ValueError(f"Expected 3D input tensor, got {x.dim()}D")
            
            return self.ff(self.norm(x))
            
        except Exception as e:
            logger.error(f"Error in feed-forward forward pass: {str(e)}")
            raise RuntimeError(f"Feed-forward computation failed: {str(e)}")