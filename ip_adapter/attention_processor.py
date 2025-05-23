# modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

try:
    import xformers
    import xformers.ops
    xformers_available = True
except Exception as e:
    xformers_available = False

# Configure logging
logger = logging.getLogger(__name__)

class RegionControler(object):
    def __init__(self) -> None:
        self.prompt_image_conditioning = []
region_control = RegionControler()


class AttnProcessor(nn.Module):
    """
    Enhanced attention processor with improved memory management and error handling.
    """
    
    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0):
        """Initialize attention processor with validation."""
        try:
            super().__init__()
            
            self.validate_inputs(hidden_size, cross_attention_dim, scale)
            
            self.hidden_size = hidden_size
            self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else hidden_size
            self.scale = scale
            
            self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
            self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
            
        except Exception as e:
            logger.error(f"Error initializing AttnProcessor: {str(e)}")
            raise RuntimeError(f"Failed to initialize attention processor: {str(e)}")
    
    def validate_inputs(self, hidden_size, cross_attention_dim, scale):
        """Validate initialization parameters."""
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        if cross_attention_dim is not None and (not isinstance(cross_attention_dim, int) or cross_attention_dim <= 0):
            raise ValueError("cross_attention_dim must be a positive integer or None")
        if not isinstance(scale, (int, float)) or scale <= 0:
            raise ValueError("scale must be a positive number")
    
    def __call__(
        self,
        attn: nn.Module,
        hidden_states: torch.Tensor,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale=1.0
    ) -> torch.Tensor:
        """
        Process attention with memory optimization and error handling.
        
        Args:
            attn (nn.Module): Attention module
            hidden_states (torch.Tensor): Input hidden states
            encoder_hidden_states (torch.Tensor, optional): Encoder hidden states
            attention_mask (torch.Tensor, optional): Attention mask
            temb (torch.Tensor, optional): Time embedding
            scale (float): Attention scale factor
            
        Returns:
            torch.Tensor: Processed hidden states
        """
        try:
            # Input validation
            if not isinstance(hidden_states, torch.Tensor):
                raise ValueError("hidden_states must be a torch.Tensor")
            if encoder_hidden_states is not None and not isinstance(encoder_hidden_states, torch.Tensor):
                raise ValueError("encoder_hidden_states must be a torch.Tensor or None")
            if attention_mask is not None and not isinstance(attention_mask, torch.Tensor):
                raise ValueError("attention_mask must be a torch.Tensor or None")
            
            batch_size, sequence_length, _ = hidden_states.shape
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            
            # Memory-efficient attention computation
            head_dim = attn.heads
            
            # Project query, key, and value
            query = attn.to_q(hidden_states)
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
            
            inner_dim = key.shape[-1]
            
            # Split heads
            query = query.view(batch_size, -1, head_dim, inner_dim).transpose(1, 2)
            key = key.view(batch_size, -1, head_dim, inner_dim).transpose(1, 2)
            value = value.view(batch_size, -1, head_dim, inner_dim).transpose(1, 2)
            
            # Compute attention with memory optimization
            with torch.cuda.amp.autocast(enabled=False):
                # Convert to float32 for better numerical stability
                query = query.float()
                key = key.float()
                
                # Compute attention scores
                attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
                
                if attention_mask is not None:
                    attention_scores = attention_scores + attention_mask
                
                attention_probs = F.softmax(attention_scores, dim=-1)
                
                # Convert back to original dtype
                attention_probs = attention_probs.to(value.dtype)
                
                # Compute output
                hidden_states = torch.matmul(attention_probs, value)
            
            # Reshape and project output
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, head_dim * inner_dim)
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            
            # Clean up memory
            del query, key, value, attention_scores, attention_probs
            torch.cuda.empty_cache()
            
            return hidden_states
            
        except Exception as e:
            logger.error(f"Error in attention processing: {str(e)}")
            raise RuntimeError(f"Attention processing failed: {str(e)}")
    
    def clear_memory(self):
        """Clear CUDA memory cache."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            logger.warning(f"Error clearing CUDA memory: {str(e)}")

class IPAttnProcessor(AttnProcessor):
    """
    IP-Adapter attention processor with enhanced memory management.
    """
    
    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4):
        """Initialize IP attention processor with validation."""
        try:
            super().__init__(hidden_size, cross_attention_dim, scale)
            
            if not isinstance(num_tokens, int) or num_tokens <= 0:
                raise ValueError("num_tokens must be a positive integer")
            
            self.num_tokens = num_tokens
            
        except Exception as e:
            logger.error(f"Error initializing IPAttnProcessor: {str(e)}")
            raise RuntimeError(f"Failed to initialize IP attention processor: {str(e)}")
    
    def __call__(
        self,
        attn: nn.Module,
        hidden_states: torch.Tensor,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale=1.0,
        image_hidden_states=None
    ) -> torch.Tensor:
        """
        Process IP attention with memory optimization and error handling.
        """
        try:
            if image_hidden_states is None:
                return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, temb, scale)
            
            # Input validation
            if not isinstance(image_hidden_states, torch.Tensor):
                raise ValueError("image_hidden_states must be a torch.Tensor")
            
            batch_size, sequence_length, _ = hidden_states.shape
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            
            head_dim = attn.heads
            
            # Project query, key, and value for text features
            query = attn.to_q(hidden_states)
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
            
            # Project key and value for image features
            ip_key = self.to_k_ip(image_hidden_states)
            ip_value = self.to_v_ip(image_hidden_states)
            
            inner_dim = key.shape[-1]
            
            # Split heads
            query = query.view(batch_size, -1, head_dim, inner_dim).transpose(1, 2)
            key = key.view(batch_size, -1, head_dim, inner_dim).transpose(1, 2)
            value = value.view(batch_size, -1, head_dim, inner_dim).transpose(1, 2)
            ip_key = ip_key.view(batch_size, -1, head_dim, inner_dim).transpose(1, 2)
            ip_value = ip_value.view(batch_size, -1, head_dim, inner_dim).transpose(1, 2)
            
            # Compute attention with memory optimization
            with torch.cuda.amp.autocast(enabled=False):
                # Convert to float32 for better numerical stability
                query = query.float()
                key = key.float()
                ip_key = ip_key.float()
                
                # Compute text attention scores
                text_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
                
                # Compute image attention scores
                image_scores = torch.matmul(query, ip_key.transpose(-1, -2)) * self.scale
                
                # Combine attention scores
                attention_scores = torch.cat([text_scores, image_scores], dim=-1)
                
                if attention_mask is not None:
                    attention_scores = attention_scores + attention_mask
                
                attention_probs = F.softmax(attention_scores, dim=-1)
                
                # Split attention probabilities
                text_probs, image_probs = attention_probs.split([key.size(-2), ip_key.size(-2)], dim=-1)
                
                # Convert back to original dtype
                text_probs = text_probs.to(value.dtype)
                image_probs = image_probs.to(ip_value.dtype)
                
                # Compute output
                text_output = torch.matmul(text_probs, value)
                image_output = torch.matmul(image_probs, ip_value)
                hidden_states = text_output + image_output * scale
            
            # Reshape and project output
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, head_dim * inner_dim)
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            
            # Clean up memory
            del query, key, value, ip_key, ip_value, attention_scores, text_probs, image_probs
            torch.cuda.empty_cache()
            
            return hidden_states
            
        except Exception as e:
            logger.error(f"Error in IP attention processing: {str(e)}")
            raise RuntimeError(f"IP attention processing failed: {str(e)}")


class AttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """
    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states