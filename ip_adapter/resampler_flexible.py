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


class FlexibleResampler(nn.Module):
    """
    Flexible Resampler that can adapt to different checkpoint structures.
    """
    
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,  # Corregido: 16 en lugar de 12 para compatibilidad matemática
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
    ):
        """Initialize Flexible Resampler."""
        super().__init__()
        
        # Validar que dim sea divisible por heads
        if dim % heads != 0:
            # Buscar el número de heads más cercano que sea compatible
            valid_heads = []
            for h in [8, 16, 32, 64, 128]:
                if dim % h == 0:
                    valid_heads.append(h)
            
            if valid_heads:
                # Elegir el más cercano al solicitado
                heads = min(valid_heads, key=lambda x: abs(x - heads))
                logger.warning(f"Ajustando heads a {heads} para compatibilidad con dim={dim}")
            else:
                raise ValueError(f"No se puede encontrar un número de heads válido para dim={dim}")
        
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.num_queries = num_queries
        
        # Usar 'queries' como nombre del parámetro (compatible con checkpoint)
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)
        
        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )
        
        logger.info(f"Initialized FlexibleResampler: dim={dim}, heads={heads}, depth={depth}")

    def forward(self, x):
        """Forward pass."""
        latents = self.queries.repeat(x.size(0), 1, 1)
        x = self.proj_in(x)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        latents = self.proj_out(latents)
        return self.norm_out(latents)
    
    def load_state_dict_flexible(self, state_dict, strict=False):
        """
        Flexible state dict loading that can handle different parameter names.
        """
        try:
            # Primero intentar carga normal
            return super().load_state_dict(state_dict, strict=strict)
        except Exception as e:
            logger.warning(f"Carga normal falló: {str(e)}")
            logger.info("Intentando carga flexible con mapeo de parámetros...")
            
            # Mapeo de nombres alternativos
            name_mappings = {
                'latents': 'queries',  # Mapear latents a queries si es necesario
                'queries': 'queries',  # Mantener queries como queries
            }
            
            # Crear nuevo state_dict con nombres mapeados
            mapped_state_dict = {}
            for key, value in state_dict.items():
                new_key = key
                for old_name, new_name in name_mappings.items():
                    if key.startswith(old_name):
                        new_key = key.replace(old_name, new_name, 1)
                        break
                mapped_state_dict[new_key] = value
            
            # Intentar carga con nombres mapeados
            try:
                return super().load_state_dict(mapped_state_dict, strict=strict)
            except Exception as e2:
                logger.error(f"Carga flexible también falló: {str(e2)}")
                
                # Información de debug
                model_keys = set(self.state_dict().keys())
                checkpoint_keys = set(mapped_state_dict.keys())
                
                missing_in_checkpoint = model_keys - checkpoint_keys
                missing_in_model = checkpoint_keys - model_keys
                
                if missing_in_checkpoint:
                    logger.error(f"Parámetros faltantes en checkpoint: {list(missing_in_checkpoint)[:5]}...")
                if missing_in_model:
                    logger.error(f"Parámetros extra en checkpoint: {list(missing_in_model)[:5]}...")
                
                # Si strict=False, intentar cargar solo los parámetros que coinciden
                if not strict:
                    logger.info("Intentando carga parcial...")
                    partial_state_dict = {}
                    for key in model_keys:
                        if key in mapped_state_dict:
                            partial_state_dict[key] = mapped_state_dict[key]
                    
                    if partial_state_dict:
                        logger.info(f"Cargando {len(partial_state_dict)} parámetros de {len(model_keys)} totales")
                        return super().load_state_dict(partial_state_dict, strict=False)
                
                raise e2


# Alias para compatibilidad
Resampler = FlexibleResampler 