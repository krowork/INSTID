#!/usr/bin/env python3
"""
Verificación Final Completa - InstantID
=======================================

Este script verifica todas las correcciones aplicadas al pipeline de InstantID:
1. Import correcto de BasicTransformerBlock
2. Corrección de self.up_blocks -> self.unet.up_blocks
3. Parámetros válidos en BasicTransformerBlock constructor
4. FlexibleResampler implementado
5. Sintaxis válida del código

Autor: AI Assistant
Fecha: 2025
"""

import os
import sys
import re

def verificar_import_basictransformerblock():
    """Verifica que el import de BasicTransformerBlock esté correcto"""
    print("🔧 Verificando import de BasicTransformerBlock...")
    
    try:
        with open('pipeline_stable_diffusion_xl_instantid.py', 'r') as f:
            content = f.read()
        
        # Verificar import correcto
        import_correcto = 'from diffusers.models.attention import BasicTransformerBlock'
        import_incorrecto = 'from diffusers.models.attention_processor import BasicTransformerBlock'
        
        if import_correcto in content:
            print("✅ Import correcto encontrado: diffusers.models.attention")
            return True
        elif import_incorrecto in content:
            print("❌ Import incorrecto encontrado: diffusers.models.attention_processor")
            return False
        else:
            print("❌ No se encontró ningún import de BasicTransformerBlock")
            return False
            
    except FileNotFoundError:
        print("❌ Archivo pipeline_stable_diffusion_xl_instantid.py no encontrado")
        return False

def verificar_correccion_up_blocks():
    """Verifica que la corrección de up_blocks esté aplicada"""
    print("🔧 Verificando corrección de up_blocks...")
    
    try:
        with open('pipeline_stable_diffusion_xl_instantid.py', 'r') as f:
            content = f.read()
        
        # Buscar la línea problemática
        linea_incorrecta = 'dim=self.up_blocks[i].resnets[1].out_channels'
        linea_correcta = 'dim=self.unet.up_blocks[i].resnets[1].out_channels'
        
        if linea_correcta in content and linea_incorrecta not in content:
            print("✅ Corrección de up_blocks aplicada correctamente")
            return True
        elif linea_incorrecta in content:
            print("❌ Línea incorrecta encontrada: self.up_blocks")
            return False
        else:
            print("❌ No se encontró la línea de configuración")
            return False
            
    except FileNotFoundError:
        print("❌ Archivo pipeline_stable_diffusion_xl_instantid.py no encontrado")
        return False

def verificar_parametros_basictransformerblock():
    """Verifica que los parámetros de BasicTransformerBlock sean válidos"""
    print("🔧 Verificando parámetros de BasicTransformerBlock...")
    
    try:
        with open('pipeline_stable_diffusion_xl_instantid.py', 'r') as f:
            content = f.read()
        
        # Buscar parámetros inválidos
        parametros_invalidos = [
            'post_attention_norm=True',
            'ff_inner_dim=None',
            'ff_bias=True'
        ]
        
        parametros_encontrados = []
        for param in parametros_invalidos:
            if param in content:
                parametros_encontrados.append(param)
        
        if not parametros_encontrados:
            print("✅ Parámetros de BasicTransformerBlock corregidos")
            return True
        else:
            print(f"❌ Parámetros inválidos encontrados: {parametros_encontrados}")
            return False
            
    except FileNotFoundError:
        print("❌ Archivo pipeline_stable_diffusion_xl_instantid.py no encontrado")
        return False

def verificar_flexible_resampler():
    """Verifica que el FlexibleResampler esté implementado"""
    print("🔧 Verificando FlexibleResampler...")
    
    try:
        if os.path.exists('ip_adapter/resampler_flexible.py'):
            print("✅ FlexibleResampler implementado")
            return True
        else:
            print("❌ FlexibleResampler no encontrado")
            return False
            
    except Exception as e:
        print(f"❌ Error verificando FlexibleResampler: {e}")
        return False

def verificar_sintaxis():
    """Verifica que la sintaxis del código sea válida"""
    print("🔧 Verificando sintaxis del código...")
    
    try:
        with open('pipeline_stable_diffusion_xl_instantid.py', 'r') as f:
            content = f.read()
        
        # Intentar compilar el código
        compile(content, 'pipeline_stable_diffusion_xl_instantid.py', 'exec')
        print("✅ Sintaxis del código válida")
        return True
        
    except SyntaxError as e:
        print(f"❌ Error de sintaxis: {e}")
        return False
    except FileNotFoundError:
        print("❌ Archivo pipeline_stable_diffusion_xl_instantid.py no encontrado")
        return False

def main():
    """Función principal que ejecuta todas las verificaciones"""
    print("🎯 Verificación Final Completa - InstantID")
    print("=" * 50)
    
    verificaciones = [
        ("Import BasicTransformerBlock", verificar_import_basictransformerblock),
        ("Corrección up_blocks", verificar_correccion_up_blocks),
        ("Parámetros BasicTransformerBlock", verificar_parametros_basictransformerblock),
        ("FlexibleResampler", verificar_flexible_resampler),
        ("Sintaxis del código", verificar_sintaxis),
    ]
    
    resultados = []
    
    for nombre, funcion in verificaciones:
        print(f"\n📋 {nombre}:")
        resultado = funcion()
        resultados.append((nombre, resultado))
    
    print("\n" + "=" * 50)
    print("📊 RESUMEN DE VERIFICACIONES:")
    print("=" * 50)
    
    exitosas = 0
    for nombre, resultado in resultados:
        estado = "✅ PASÓ" if resultado else "❌ FALLÓ"
        print(f"{estado} - {nombre}")
        if resultado:
            exitosas += 1
    
    print(f"\n🎯 RESULTADO FINAL: {exitosas}/{len(verificaciones)} verificaciones pasadas")
    
    if exitosas == len(verificaciones):
        print("🎉 ¡TODAS LAS CORRECCIONES APLICADAS CORRECTAMENTE!")
        print("💡 El pipeline de InstantID debería funcionar ahora en Colab")
    else:
        print("⚠️  Algunas correcciones necesitan atención")
        print("💡 Revisa los errores arriba para más detalles")
    
    return exitosas == len(verificaciones)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 