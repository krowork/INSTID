#!/usr/bin/env python3
"""
Test script para verificar que el import de BasicTransformerBlock funciona correctamente
"""

print("🔧 Probando imports...")

try:
    from diffusers.models.attention import BasicTransformerBlock
    print("✅ BasicTransformerBlock importado correctamente desde diffusers.models.attention")
except ImportError as e:
    print(f"❌ Error importando BasicTransformerBlock: {e}")

try:
    from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline
    print("✅ Pipeline importado correctamente")
except ImportError as e:
    print(f"❌ Error importando pipeline: {e}")

print("�� Test completado!") 