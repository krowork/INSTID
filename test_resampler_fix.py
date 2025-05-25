#!/usr/bin/env python3
"""
Script para verificar que la corrección del Resampler funciona correctamente.
"""

import sys
import torch

def test_resampler_compatibility():
    """Prueba la compatibilidad del Resampler con diferentes configuraciones."""
    
    print("🧪 Probando compatibilidad del Resampler...")
    
    try:
        # Importar el Resampler
        from ip_adapter.resampler import Resampler
        
        # Configuraciones a probar
        test_configs = [
            {
                "name": "Configuración Original (Problemática)",
                "dim": 2048,
                "heads": 12,
                "should_fail": True
            },
            {
                "name": "Configuración Corregida",
                "dim": 2048,
                "heads": 16,
                "should_fail": False
            },
            {
                "name": "Configuración por Defecto",
                "dim": 1024,
                "heads": 16,
                "should_fail": False
            },
            {
                "name": "Configuración Alternativa",
                "dim": 2048,
                "heads": 32,
                "should_fail": False
            }
        ]
        
        for config in test_configs:
            print(f"\n📋 Probando: {config['name']}")
            print(f"   dim={config['dim']}, heads={config['heads']}")
            
            try:
                resampler = Resampler(
                    dim=config['dim'],
                    heads=config['heads'],
                    depth=4,
                    dim_head=64,
                    num_queries=16,
                    embedding_dim=512,
                    output_dim=config['dim']
                )
                
                if config['should_fail']:
                    print(f"   ❌ ERROR: Debería haber fallado pero funcionó")
                    return False
                else:
                    print(f"   ✅ ÉXITO: Inicialización correcta")
                    
                    # Probar forward pass
                    batch_size = 2
                    seq_len = 257
                    embedding_dim = 512
                    
                    x = torch.randn(batch_size, seq_len, embedding_dim)
                    output = resampler(x)
                    
                    expected_shape = (batch_size, 16, config['dim'])
                    if output.shape == expected_shape:
                        print(f"   ✅ Forward pass correcto: {output.shape}")
                    else:
                        print(f"   ❌ Forward pass incorrecto: esperado {expected_shape}, obtenido {output.shape}")
                        return False
                        
            except Exception as e:
                if config['should_fail']:
                    print(f"   ✅ ESPERADO: Falló como se esperaba - {str(e)}")
                else:
                    print(f"   ❌ ERROR: No debería haber fallado - {str(e)}")
                    return False
        
        print(f"\n🎉 Todas las pruebas pasaron correctamente!")
        return True
        
    except Exception as e:
        print(f"❌ Error en las pruebas: {str(e)}")
        return False

def verify_divisibility():
    """Verifica que las configuraciones sean matemáticamente correctas."""
    
    print("\n🔢 Verificando divisibilidad matemática...")
    
    # Configuraciones comunes
    configs = [
        (1024, 8), (1024, 16), (1024, 32),
        (2048, 8), (2048, 16), (2048, 32), (2048, 64),
        (512, 8), (512, 16),
        (768, 8), (768, 12), (768, 16)
    ]
    
    for dim, heads in configs:
        is_divisible = dim % heads == 0
        status = "✅" if is_divisible else "❌"
        print(f"   {status} dim={dim}, heads={heads} -> {dim}/{heads} = {dim/heads}")
    
    print("\n💡 Recomendaciones:")
    print("   • Para dim=2048: usar heads=8, 16, 32, 64, 128, 256, 512, 1024, 2048")
    print("   • Para dim=1024: usar heads=8, 16, 32, 64, 128, 256, 512, 1024")
    print("   • Para dim=768: usar heads=8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 768")

if __name__ == "__main__":
    print("🔧 Verificación de la corrección del Resampler")
    print("=" * 50)
    
    # Verificar divisibilidad matemática
    verify_divisibility()
    
    # Probar el Resampler
    success = test_resampler_compatibility()
    
    if success:
        print("\n✅ La corrección del Resampler es exitosa!")
        print("🚀 El notebook debería funcionar correctamente ahora.")
        sys.exit(0)
    else:
        print("\n❌ Hay problemas con la corrección.")
        sys.exit(1) 