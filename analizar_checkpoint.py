#!/usr/bin/env python3
"""
Script para analizar la estructura del checkpoint de InstantID.
"""

import torch
import os
from pathlib import Path

def analizar_checkpoint(checkpoint_path):
    """Analiza la estructura de un checkpoint."""
    try:
        print(f"🔍 Analizando checkpoint: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            print(f"❌ Archivo no encontrado: {checkpoint_path}")
            return False
        
        # Cargar checkpoint
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        
        print(f"📦 Checkpoint cargado exitosamente")
        print(f"   - Número total de parámetros: {len(state_dict)}")
        
        # Analizar parámetros por prefijo
        prefijos = {}
        for key in state_dict.keys():
            prefijo = key.split('.')[0]
            if prefijo not in prefijos:
                prefijos[prefijo] = []
            prefijos[prefijo].append(key)
        
        print(f"\n📋 Análisis por prefijos:")
        for prefijo, keys in prefijos.items():
            print(f"   - {prefijo}: {len(keys)} parámetros")
            if len(keys) <= 5:
                for key in keys:
                    shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
                    print(f"     * {key}: {shape}")
            else:
                for key in keys[:3]:
                    shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
                    print(f"     * {key}: {shape}")
                print(f"     * ... y {len(keys)-3} más")
        
        # Analizar específicamente image_proj
        image_proj_keys = [k for k in state_dict.keys() if k.startswith("image_proj.")]
        if image_proj_keys:
            print(f"\n🎯 Análisis detallado de image_proj ({len(image_proj_keys)} parámetros):")
            for key in image_proj_keys:
                clean_key = key.replace("image_proj.", "")
                shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
                print(f"   - {clean_key}: {shape}")
        else:
            print(f"\n❌ No se encontraron parámetros image_proj en el checkpoint")
        
        return True
        
    except Exception as e:
        print(f"❌ Error analizando checkpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def buscar_checkpoints():
    """Busca checkpoints en directorios comunes."""
    posibles_paths = [
        "checkpoints/ip-adapter.bin",
        "models/ip-adapter.bin", 
        "ip-adapter.bin",
        "checkpoints/InstantID/ip-adapter.bin",
        "models/InstantID/ip-adapter.bin"
    ]
    
    print("🔍 Buscando checkpoints en directorios comunes...")
    
    encontrados = []
    for path in posibles_paths:
        if os.path.exists(path):
            encontrados.append(path)
            print(f"✅ Encontrado: {path}")
    
    if not encontrados:
        print("❌ No se encontraron checkpoints en ubicaciones comunes")
        print("💡 Ubicaciones buscadas:")
        for path in posibles_paths:
            print(f"   - {path}")
    
    return encontrados

if __name__ == "__main__":
    print("🧪 Analizador de Checkpoints InstantID")
    print("=" * 50)
    
    # Buscar checkpoints
    checkpoints = buscar_checkpoints()
    
    if checkpoints:
        print(f"\n📁 Analizando {len(checkpoints)} checkpoint(s) encontrado(s):")
        for checkpoint in checkpoints:
            print("\n" + "="*50)
            analizar_checkpoint(checkpoint)
    else:
        print("\n💡 Para usar este script:")
        print("   1. Coloca el archivo ip-adapter.bin en uno de estos directorios:")
        print("      - checkpoints/")
        print("      - models/")
        print("      - directorio actual")
        print("   2. O ejecuta: python3 analizar_checkpoint.py /ruta/al/checkpoint.bin")
        
        # Si se proporciona un argumento, analizarlo
        import sys
        if len(sys.argv) > 1:
            checkpoint_path = sys.argv[1]
            print(f"\n🔍 Analizando checkpoint proporcionado: {checkpoint_path}")
            analizar_checkpoint(checkpoint_path) 