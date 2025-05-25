#!/usr/bin/env python3
"""
🧠 Optimizador de Memoria para InstantID en Google Colab
========================================================

Este script implementa múltiples estrategias para optimizar el uso de memoria
durante la carga de modelos de InstantID, especialmente útil para Google Colab
donde la RAM es limitada.

Estrategias implementadas:
1. Carga secuencial de modelos con limpieza entre pasos
2. Uso de CPU offloading y técnicas de memoria compartida
3. Configuración optimizada de PyTorch para memoria
4. Monitoreo en tiempo real del uso de memoria
5. Limpieza automática de caché y garbage collection
"""

import os
import gc
import sys
import torch
import psutil
import warnings
from typing import Optional, Dict, Any
from contextlib import contextmanager
from datetime import datetime

# Configurar warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

class MemoryOptimizer:
    """Optimizador de memoria para carga de modelos."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.initial_memory = self._get_memory_info()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Configurar optimizaciones de PyTorch
        self._configure_pytorch()
        
        if self.verbose:
            self._print_system_info()
    
    def _configure_pytorch(self):
        """Configura PyTorch para uso optimizado de memoria."""
        # Configurar variables de entorno para optimización
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        os.environ['TORCH_HOME'] = './torch_cache'
        os.environ['HF_HOME'] = './hf_cache'
        os.environ['TRANSFORMERS_CACHE'] = './transformers_cache'
        
        # Configurar PyTorch para memoria eficiente
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
    
    def _get_memory_info(self) -> Dict[str, float]:
        """Obtiene información actual de memoria."""
        memory_info = {}
        
        # Memoria del sistema
        vm = psutil.virtual_memory()
        memory_info['system_total'] = vm.total / (1024**3)
        memory_info['system_available'] = vm.available / (1024**3)
        memory_info['system_used'] = vm.used / (1024**3)
        memory_info['system_percent'] = vm.percent
        
        # Memoria GPU si está disponible
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.mem_get_info()
            memory_info['gpu_free'] = gpu_memory[0] / (1024**3)
            memory_info['gpu_total'] = gpu_memory[1] / (1024**3)
            memory_info['gpu_used'] = (gpu_memory[1] - gpu_memory[0]) / (1024**3)
        
        return memory_info
    
    def _print_system_info(self):
        """Imprime información del sistema."""
        print("🧠 Optimizador de Memoria para InstantID")
        print("=" * 50)
        print(f"📱 Dispositivo: {self.device}")
        print(f"🔢 Tipo de datos: {self.dtype}")
        
        memory = self._get_memory_info()
        print(f"💾 RAM Total: {memory['system_total']:.2f} GB")
        print(f"💾 RAM Disponible: {memory['system_available']:.2f} GB")
        print(f"💾 RAM Usada: {memory['system_used']:.2f} GB ({memory['system_percent']:.1f}%)")
        
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            print(f"🎮 VRAM Total: {memory['gpu_total']:.2f} GB")
            print(f"🎮 VRAM Libre: {memory['gpu_free']:.2f} GB")
            print(f"🎮 VRAM Usada: {memory['gpu_used']:.2f} GB")
        
        print("=" * 50)
    
    def cleanup_memory(self, aggressive: bool = False):
        """Limpia la memoria del sistema y GPU."""
        if self.verbose:
            print("🧹 Limpiando memoria...")
        
        # Garbage collection
        gc.collect()
        
        # Limpiar caché de PyTorch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            if aggressive:
                # Limpieza más agresiva
                torch.cuda.ipc_collect()
        
        if self.verbose:
            memory = self._get_memory_info()
            print(f"   💾 RAM disponible: {memory['system_available']:.2f} GB")
            if torch.cuda.is_available():
                print(f"   🎮 VRAM libre: {memory['gpu_free']:.2f} GB")
    
    @contextmanager
    def memory_monitor(self, step_name: str):
        """Context manager para monitorear memoria durante una operación."""
        if self.verbose:
            print(f"\n🔄 {step_name}...")
            
        memory_before = self._get_memory_info()
        start_time = datetime.now()
        
        try:
            yield
        finally:
            end_time = datetime.now()
            memory_after = self._get_memory_info()
            duration = (end_time - start_time).total_seconds()
            
            if self.verbose:
                ram_diff = memory_after['system_used'] - memory_before['system_used']
                print(f"   ✅ Completado en {duration:.1f}s")
                print(f"   📊 RAM usada: {ram_diff:+.2f} GB")
                print(f"   📊 RAM disponible: {memory_after['system_available']:.2f} GB")
                
                if torch.cuda.is_available():
                    vram_diff = memory_after['gpu_used'] - memory_before['gpu_used']
                    print(f"   📊 VRAM usada: {vram_diff:+.2f} GB")
                    print(f"   📊 VRAM libre: {memory_after['gpu_free']:.2f} GB")
    
    def check_memory_threshold(self, threshold_gb: float = 1.0) -> bool:
        """Verifica si hay suficiente memoria disponible."""
        memory = self._get_memory_info()
        available = memory['system_available']
        
        if available < threshold_gb:
            if self.verbose:
                print(f"⚠️  Memoria baja: {available:.2f} GB disponible (umbral: {threshold_gb} GB)")
            return False
        return True
    
    def optimize_model_loading(self):
        """Aplica optimizaciones específicas para carga de modelos."""
        if self.verbose:
            print("🔧 Aplicando optimizaciones para carga de modelos...")
        
        # Configurar límites de memoria más conservadores
        if torch.cuda.is_available():
            # Reservar menos memoria inicial
            torch.cuda.set_per_process_memory_fraction(0.8)
        
        # Configurar número de workers para DataLoader
        os.environ['NUMEXPR_MAX_THREADS'] = '4'
        
        if self.verbose:
            print("   ✅ Optimizaciones aplicadas")

def load_models_optimized(optimizer: MemoryOptimizer):
    """Carga los modelos de InstantID de forma optimizada."""
    
    # Verificar memoria inicial
    if not optimizer.check_memory_threshold(2.0):
        print("❌ Memoria insuficiente para continuar")
        return None, None, None
    
    # Aplicar optimizaciones
    optimizer.optimize_model_loading()
    
    # Importaciones necesarias
    with optimizer.memory_monitor("Importando dependencias"):
        import torch
        from huggingface_hub import hf_hub_download
        from diffusers.models import ControlNetModel
        from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline
        from insightface.app import FaceAnalysis
        from diffusers.utils import logging as diffusers_logging
        
        # Configurar logging silencioso
        diffusers_logging.set_verbosity_error()
    
    # Crear directorios
    with optimizer.memory_monitor("Creando directorios"):
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("checkpoints/ControlNetModel", exist_ok=True)
        os.makedirs("models", exist_ok=True)
    
    # Descargar modelos necesarios
    with optimizer.memory_monitor("Descargando modelos"):
        model_files = [
            {"filename": "ControlNetModel/config.json", "repo_id": "InstantX/InstantID", "desc": "Config ControlNet"},
            {"filename": "ControlNetModel/diffusion_pytorch_model.safetensors", "repo_id": "InstantX/InstantID", "desc": "Modelo ControlNet"},
            {"filename": "ip-adapter.bin", "repo_id": "InstantX/InstantID", "desc": "IP-Adapter"}
        ]
        
        for file_info in model_files:
            file_path = os.path.join("checkpoints", file_info["filename"])
            if not os.path.exists(file_path):
                print(f"   📥 Descargando {file_info['desc']}...")
                hf_hub_download(
                    repo_id=file_info['repo_id'],
                    filename=file_info['filename'],
                    local_dir="./checkpoints",
                    resume_download=True
                )
    
    # Limpiar memoria antes de cargar modelos pesados
    optimizer.cleanup_memory(aggressive=True)
    
    # Verificar memoria antes de continuar
    if not optimizer.check_memory_threshold(1.5):
        print("⚠️  Memoria baja, aplicando limpieza agresiva...")
        optimizer.cleanup_memory(aggressive=True)
    
    # Cargar analizador facial
    with optimizer.memory_monitor("Cargando analizador facial"):
        try:
            # Configurar modelo facial más ligero si hay poca memoria
            memory = optimizer._get_memory_info()
            if memory['system_available'] < 3.0:
                det_size = (320, 320)  # Tamaño más pequeño para ahorrar memoria
                print("   🔧 Usando configuración de memoria reducida")
            else:
                det_size = (640, 640)
            
            app = FaceAnalysis(
                name='buffalo_l', 
                root='./models', 
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            app.prepare(ctx_id=0, det_size=det_size)
            
        except Exception as e:
            print(f"   ❌ Error cargando analizador facial: {e}")
            return None, None, None
    
    # Limpiar memoria entre cargas
    optimizer.cleanup_memory()
    
    # Cargar ControlNet
    with optimizer.memory_monitor("Cargando ControlNet"):
        try:
            controlnet = ControlNetModel.from_pretrained(
                'checkpoints/ControlNetModel',
                torch_dtype=optimizer.dtype,
                use_safetensors=True,
                low_cpu_mem_usage=True  # Optimización de memoria
            )
        except Exception as e:
            print(f"   ❌ Error cargando ControlNet: {e}")
            return None, None, None
    
    # Limpiar memoria antes del pipeline principal
    optimizer.cleanup_memory()
    
    # Verificar memoria crítica
    if not optimizer.check_memory_threshold(1.0):
        print("❌ Memoria insuficiente para cargar el pipeline principal")
        return None, None, None
    
    # Cargar pipeline principal
    with optimizer.memory_monitor("Cargando pipeline principal"):
        try:
            pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                controlnet=controlnet,
                torch_dtype=optimizer.dtype,
                safety_checker=None,
                feature_extractor=None,
                variant="fp16" if optimizer.device == "cuda" else None,
                low_cpu_mem_usage=True,  # Optimización de memoria
                use_safetensors=True
            )
            
            # Aplicar optimizaciones de memoria específicas
            if optimizer.device == "cuda":
                print("   🔧 Aplicando optimizaciones GPU...")
                pipe.enable_model_cpu_offload()  # Mover modelos a CPU cuando no se usen
                pipe.enable_vae_slicing()        # Procesar VAE en chunks
                pipe.enable_vae_tiling()         # Procesar VAE en tiles
                pipe.enable_attention_slicing()  # Reducir memoria de atención
                
                # Configuración adicional para memoria muy limitada
                memory = optimizer._get_memory_info()
                if memory['system_available'] < 2.0:
                    pipe.enable_sequential_cpu_offload()  # Offload secuencial más agresivo
                    print("   🔧 Modo de memoria ultra-conservador activado")
            
        except Exception as e:
            print(f"   ❌ Error cargando pipeline: {e}")
            return None, None, None
    
    # Cargar IP-Adapter
    with optimizer.memory_monitor("Cargando IP-Adapter"):
        try:
            pipe.load_ip_adapter_instantid('checkpoints/ip-adapter.bin')
        except Exception as e:
            print(f"   ❌ Error cargando IP-Adapter: {e}")
            return None, None, None
    
    # Limpieza final
    optimizer.cleanup_memory()
    
    # Verificar estado final
    final_memory = optimizer._get_memory_info()
    print(f"\n🎉 ¡Modelos cargados exitosamente!")
    print(f"📊 Memoria final disponible: {final_memory['system_available']:.2f} GB")
    if torch.cuda.is_available():
        print(f"📊 VRAM final libre: {final_memory['gpu_free']:.2f} GB")
    
    return pipe, app, controlnet

def main():
    """Función principal para probar el optimizador."""
    print("🚀 Iniciando carga optimizada de modelos InstantID...")
    
    # Crear optimizador
    optimizer = MemoryOptimizer(verbose=True)
    
    # Cargar modelos
    pipe, app, controlnet = load_models_optimized(optimizer)
    
    if pipe is not None and app is not None:
        print("\n✅ ¡Carga completada exitosamente!")
        print("Los modelos están listos para usar.")
        
        # Guardar referencias globales
        globals()['pipe'] = pipe
        globals()['app'] = app
        globals()['controlnet'] = controlnet
        globals()['optimizer'] = optimizer
        
        return True
    else:
        print("\n❌ Error en la carga de modelos")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n💡 Sugerencias para resolver problemas de memoria:")
        print("1. Reinicia el runtime de Colab: Runtime → Restart runtime")
        print("2. Usa una instancia con más RAM (Colab Pro)")
        print("3. Cierra otras pestañas del navegador")
        print("4. Ejecuta el script en una sesión limpia") 