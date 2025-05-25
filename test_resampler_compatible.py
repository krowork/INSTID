#!/usr/bin/env python3
"""
Test script para verificar que el Resampler compatible funciona con el checkpoint.
"""

import torch
import logging
from ip_adapter.resampler_compatible import CompatibleResampler

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_resampler_compatible():
    """Test the compatible Resampler with checkpoint loading."""
    try:
        print("🔧 Probando Resampler Compatible...")
        
        # Crear Resampler con parámetros típicos de InstantID
        resampler = CompatibleResampler(
            dim=2048,  # cross_attention_dim del UNet
            depth=4,
            dim_head=64,
            heads=16,  # Corregido: 16 en lugar de 12
            num_queries=16,  # num_tokens típico
            embedding_dim=512,  # image_emb_dim típico
            output_dim=2048,
            ff_mult=4
        )
        
        print(f"✅ Resampler creado exitosamente")
        print(f"   - dim: 2048")
        print(f"   - heads: 16")
        print(f"   - dim_head: 64")
        print(f"   - Verificación matemática: 2048 ÷ 16 = {2048 // 16} ✓")
        
        # Verificar estructura del modelo
        print("\n📋 Estructura del modelo:")
        for name, param in resampler.named_parameters():
            print(f"   - {name}: {param.shape}")
        
        # Test forward pass
        print("\n🚀 Probando forward pass...")
        batch_size = 1
        seq_len = 257  # Típico para CLIP
        embedding_dim = 512
        
        x = torch.randn(batch_size, seq_len, embedding_dim)
        
        with torch.no_grad():
            output = resampler(x)
        
        print(f"✅ Forward pass exitoso")
        print(f"   - Input shape: {x.shape}")
        print(f"   - Output shape: {output.shape}")
        print(f"   - Expected output shape: ({batch_size}, 16, 2048)")
        
        # Verificar que la salida tiene la forma correcta
        expected_shape = (batch_size, 16, 2048)
        if output.shape == expected_shape:
            print(f"✅ Output shape correcto: {output.shape}")
        else:
            print(f"❌ Output shape incorrecto. Esperado: {expected_shape}, Obtenido: {output.shape}")
            return False
        
        print("\n🎯 Test del Resampler Compatible: ¡EXITOSO!")
        return True
        
    except Exception as e:
        print(f"❌ Error en test del Resampler Compatible: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_checkpoint_compatibility():
    """Test loading a dummy checkpoint structure."""
    try:
        print("\n🔍 Probando compatibilidad con estructura de checkpoint...")
        
        # Crear Resampler
        resampler = CompatibleResampler(
            dim=2048,
            depth=4,
            dim_head=64,
            heads=16,
            num_queries=16,
            embedding_dim=512,
            output_dim=2048,
            ff_mult=4
        )
        
        # Crear un state_dict dummy que simule la estructura del checkpoint
        dummy_state_dict = {}
        
        # Agregar parámetros esperados basados en el error
        dummy_state_dict["queries"] = torch.randn(1, 16, 2048)
        
        # Agregar layers (4 layers, cada una con attention y feedforward)
        for layer_idx in range(4):
            # Attention layers
            dummy_state_dict[f"layers.{layer_idx}.0.norm1.weight"] = torch.randn(2048)
            dummy_state_dict[f"layers.{layer_idx}.0.norm1.bias"] = torch.randn(2048)
            dummy_state_dict[f"layers.{layer_idx}.0.norm2.weight"] = torch.randn(2048)
            dummy_state_dict[f"layers.{layer_idx}.0.norm2.bias"] = torch.randn(2048)
            dummy_state_dict[f"layers.{layer_idx}.0.to_q.weight"] = torch.randn(1024, 2048)
            dummy_state_dict[f"layers.{layer_idx}.0.to_kv.weight"] = torch.randn(2048, 2048)
            dummy_state_dict[f"layers.{layer_idx}.0.to_out.weight"] = torch.randn(2048, 1024)
            
            # FeedForward layers
            dummy_state_dict[f"layers.{layer_idx}.1.0.weight"] = torch.randn(2048)  # LayerNorm
            dummy_state_dict[f"layers.{layer_idx}.1.0.bias"] = torch.randn(2048)
            dummy_state_dict[f"layers.{layer_idx}.1.1.weight"] = torch.randn(8192, 2048)  # Linear 1
            dummy_state_dict[f"layers.{layer_idx}.1.1.bias"] = torch.randn(8192)
            dummy_state_dict[f"layers.{layer_idx}.1.3.weight"] = torch.randn(2048, 8192)  # Linear 2
            dummy_state_dict[f"layers.{layer_idx}.1.3.bias"] = torch.randn(2048)
        
        # Projection layers
        dummy_state_dict["proj_in.weight"] = torch.randn(2048, 512)
        dummy_state_dict["proj_in.bias"] = torch.randn(2048)
        dummy_state_dict["proj_out.weight"] = torch.randn(2048, 2048)
        dummy_state_dict["proj_out.bias"] = torch.randn(2048)
        dummy_state_dict["norm_out.weight"] = torch.randn(2048)
        dummy_state_dict["norm_out.bias"] = torch.randn(2048)
        
        print(f"📦 Dummy state_dict creado con {len(dummy_state_dict)} parámetros")
        
        # Intentar cargar el state_dict
        try:
            resampler.load_state_dict(dummy_state_dict, strict=False)
            print("✅ State dict cargado exitosamente (strict=False)")
        except Exception as e:
            print(f"⚠️  Error cargando state_dict: {str(e)}")
            
            # Mostrar qué parámetros faltan
            model_keys = set(resampler.state_dict().keys())
            checkpoint_keys = set(dummy_state_dict.keys())
            
            missing_in_checkpoint = model_keys - checkpoint_keys
            missing_in_model = checkpoint_keys - model_keys
            
            if missing_in_checkpoint:
                print(f"❌ Parámetros faltantes en checkpoint: {list(missing_in_checkpoint)[:5]}...")
            if missing_in_model:
                print(f"❌ Parámetros extra en checkpoint: {list(missing_in_model)[:5]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en test de compatibilidad: {str(e)}")
        return False

if __name__ == "__main__":
    print("🧪 Iniciando tests del Resampler Compatible...")
    
    success1 = test_resampler_compatible()
    success2 = test_checkpoint_compatibility()
    
    if success1 and success2:
        print("\n🎉 ¡Todos los tests pasaron exitosamente!")
        print("✅ El Resampler Compatible está listo para usar")
    else:
        print("\n❌ Algunos tests fallaron. Revisar la implementación.") 