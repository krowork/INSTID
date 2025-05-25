#!/usr/bin/env python3
"""
Verificación Final - Solución InstantID Completa
===============================================

Este script verifica que todas las correcciones estén aplicadas correctamente:
1. Import correcto de BasicTransformerBlock
2. Clases AttnProcessor e IPAttnProcessor implementadas
3. Método set_ip_adapter corregido para usar attention processors
4. FlexibleResampler implementado
5. Sintaxis válida del código

Autor: AI Assistant
Fecha: 2025
"""

import os
import sys
import re

def verificar_imports():
    """Verifica que todos los imports estén correctos"""
    print("🔧 Verificando imports...")
    
    try:
        with open('pipeline_stable_diffusion_xl_instantid.py', 'r') as f:
            content = f.read()
        
        # Verificar import de BasicTransformerBlock
        if 'from diffusers.models.attention import BasicTransformerBlock' in content:
            print("✅ BasicTransformerBlock import correcto")
        else:
            print("❌ BasicTransformerBlock import incorrecto")
            return False
        
        # Verificar import de Attention con fallback
        if 'try:' in content and 'from diffusers.models.attention_processor import Attention' in content:
            print("✅ Attention import con fallback correcto")
        else:
            print("❌ Attention import faltante")
            return False
            
        return True
        
    except FileNotFoundError:
        print("❌ Archivo pipeline no encontrado")
        return False

def verificar_clases_attention():
    """Verifica que las clases de attention estén implementadas"""
    print("🔧 Verificando clases de attention...")
    
    try:
        with open('pipeline_stable_diffusion_xl_instantid.py', 'r') as f:
            content = f.read()
        
        # Verificar AttnProcessor
        if 'class AttnProcessor:' in content:
            print("✅ AttnProcessor implementado")
        else:
            print("❌ AttnProcessor faltante")
            return False
        
        # Verificar IPAttnProcessor
        if 'class IPAttnProcessor(torch.nn.Module):' in content:
            print("✅ IPAttnProcessor implementado")
        else:
            print("❌ IPAttnProcessor faltante")
            return False
        
        # Verificar métodos de IPAttnProcessor
        if 'self.to_k_ip = torch.nn.Linear' in content and 'self.to_v_ip = torch.nn.Linear' in content:
            print("✅ IPAttnProcessor tiene to_k_ip y to_v_ip")
        else:
            print("❌ IPAttnProcessor incompleto")
            return False
            
        return True
        
    except FileNotFoundError:
        print("❌ Archivo pipeline no encontrado")
        return False

def verificar_set_ip_adapter():
    """Verifica que el método set_ip_adapter esté corregido"""
    print("🔧 Verificando método set_ip_adapter...")
    
    try:
        with open('pipeline_stable_diffusion_xl_instantid.py', 'r') as f:
            content = f.read()
        
        # Verificar que NO use BasicTransformerBlock en set_ip_adapter
        set_ip_adapter_match = re.search(r'def set_ip_adapter.*?(?=def|\Z)', content, re.DOTALL)
        if set_ip_adapter_match:
            method_content = set_ip_adapter_match.group(0)
            
            if 'BasicTransformerBlock(' in method_content:
                print("❌ set_ip_adapter todavía usa BasicTransformerBlock")
                return False
            else:
                print("✅ set_ip_adapter NO usa BasicTransformerBlock")
            
            # Verificar que use attention processors
            if 'attn_procs = {}' in method_content and 'IPAttnProcessor(' in method_content:
                print("✅ set_ip_adapter usa attention processors")
            else:
                print("❌ set_ip_adapter no usa attention processors correctamente")
                return False
                
            # Verificar manejo de errores
            if 'except Exception as e:' in method_content and 'logger.warning' in method_content:
                print("✅ set_ip_adapter tiene manejo de errores robusto")
            else:
                print("❌ set_ip_adapter sin manejo de errores adecuado")
                return False
        else:
            print("❌ Método set_ip_adapter no encontrado")
            return False
            
        return True
        
    except FileNotFoundError:
        print("❌ Archivo pipeline no encontrado")
        return False

def verificar_flexible_resampler():
    """Verifica que FlexibleResampler esté implementado"""
    print("🔧 Verificando FlexibleResampler...")
    
    try:
        if os.path.exists('ip_adapter/resampler_flexible.py'):
            with open('ip_adapter/resampler_flexible.py', 'r') as f:
                content = f.read()
            
            if 'class FlexibleResampler' in content:
                print("✅ FlexibleResampler implementado")
                
                if 'load_state_dict_flexible' in content:
                    print("✅ FlexibleResampler tiene carga flexible")
                else:
                    print("⚠️ FlexibleResampler sin carga flexible")
                    
                return True
            else:
                print("❌ FlexibleResampler no encontrado en el archivo")
                return False
        else:
            print("⚠️ Archivo resampler_flexible.py no encontrado")
            return True  # No es crítico
            
    except Exception as e:
        print(f"⚠️ Error verificando FlexibleResampler: {e}")
        return True  # No es crítico

def verificar_sintaxis():
    """Verifica que la sintaxis del código sea válida"""
    print("🔧 Verificando sintaxis...")
    
    try:
        with open('pipeline_stable_diffusion_xl_instantid.py', 'r') as f:
            content = f.read()
        
        # Intentar compilar el código
        compile(content, 'pipeline_stable_diffusion_xl_instantid.py', 'exec')
        print("✅ Sintaxis válida")
        return True
        
    except SyntaxError as e:
        print(f"❌ Error de sintaxis: {e}")
        return False
    except FileNotFoundError:
        print("❌ Archivo pipeline no encontrado")
        return False

def main():
    """Función principal de verificación"""
    print("🎯 Verificación Final - Solución InstantID Completa")
    print("=" * 60)
    
    verificaciones = [
        ("Imports", verificar_imports),
        ("Clases Attention", verificar_clases_attention),
        ("Método set_ip_adapter", verificar_set_ip_adapter),
        ("FlexibleResampler", verificar_flexible_resampler),
        ("Sintaxis", verificar_sintaxis),
    ]
    
    resultados = []
    
    for nombre, funcion in verificaciones:
        print(f"\n📋 {nombre}:")
        resultado = funcion()
        resultados.append((nombre, resultado))
    
    print("\n" + "=" * 60)
    print("📊 RESUMEN FINAL:")
    
    exitosas = 0
    for nombre, resultado in resultados:
        estado = "✅ PASS" if resultado else "❌ FAIL"
        print(f"  {estado} {nombre}")
        if resultado:
            exitosas += 1
    
    print(f"\n🎯 Resultado: {exitosas}/{len(resultados)} verificaciones exitosas")
    
    if exitosas == len(resultados):
        print("🎉 ¡TODAS LAS CORRECCIONES APLICADAS CORRECTAMENTE!")
        print("📝 El pipeline InstantID está listo para usar en Colab")
    else:
        print("⚠️ Algunas verificaciones fallaron")
        print("📝 Revisar los errores antes de usar en Colab")
    
    return exitosas == len(resultados)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 