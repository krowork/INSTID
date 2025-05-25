#!/usr/bin/env python3
"""
Script completo para actualizar el notebook InstantID con:
1. Versiones compatibles con PyTorch 2.6+ en la celda de instalación
2. Optimizaciones de memoria en la celda de carga de modelos
"""

import json
import sys

def create_updated_installation_cell():
    """Crea la celda de instalación actualizada con versiones compatibles con PyTorch 2.6+."""
    
    cell_code = '''# 🔥 Instalación compatible con PyTorch 2.6+ y Colab actual
import torch
if not torch.cuda.is_available():
    raise RuntimeError("\\n❌ Este notebook requiere una GPU. Por favor, selecciona: Runtime -> Change runtime type -> GPU")

print("🔧 Iniciando instalación compatible con PyTorch 2.6+...")
print("✨ Usando versiones compatibles con el entorno actual de Colab")

# Verificar versiones actuales
print(f"\\n📋 PyTorch actual: {torch.__version__}")
print(f"📋 CUDA actual: {torch.version.cuda}")

# Configurar variables de entorno para evitar conflictos
import os
os.environ['TORCH_HOME'] = './torch_home'
os.environ['HF_HOME'] = './hf_home'
os.environ['TRANSFORMERS_CACHE'] = './transformers_cache'

# Crear directorios de caché
!mkdir -p ./torch_home ./hf_home ./transformers_cache

print("\\n1️⃣ Actualizando dependencias principales...")
# Usar versiones compatibles con PyTorch 2.6+
!pip install --quiet --no-warn-script-location transformers>=4.41.0
!pip install --quiet --no-warn-script-location diffusers>=0.30.0
!pip install --quiet --no-warn-script-location huggingface-hub>=0.25.0
!pip install --quiet --no-warn-script-location accelerate>=0.30.0

print("\\n2️⃣ Instalando dependencias de visión...")
!pip install --quiet --no-warn-script-location opencv-python
!pip install --quiet --no-warn-script-location Pillow
!pip install --quiet --no-warn-script-location safetensors

print("\\n3️⃣ Instalando dependencias de IA facial...")
!pip install --quiet --no-warn-script-location insightface
!pip install --quiet --no-warn-script-location onnx
!pip install --quiet --no-warn-script-location onnxruntime-gpu

print("\\n4️⃣ Instalando ControlNet y utilidades...")
!pip install --quiet --no-warn-script-location controlnet_aux

print("\\n5️⃣ Instalando interfaz web...")
!pip install --quiet --no-warn-script-location gradio

print("\\n6️⃣ Instalando dependencias adicionales...")
!pip install --quiet --no-warn-script-location psutil  # Para monitoreo de memoria

# Verificación final
print("\\n🔍 Verificando instalación...")
try:
    import transformers
    print(f"✅ Transformers: {transformers.__version__}")
    
    import diffusers
    print(f"✅ Diffusers: {diffusers.__version__}")
    
    import huggingface_hub
    print(f"✅ HuggingFace Hub: {huggingface_hub.__version__}")
    
    import accelerate
    print(f"✅ Accelerate: {accelerate.__version__}")
    
    import cv2
    print(f"✅ OpenCV: {cv2.__version__}")
    
    import insightface
    print(f"✅ InsightFace: {insightface.__version__}")
    
    import gradio
    print(f"✅ Gradio: {gradio.__version__}")
    
    import psutil
    print(f"✅ PSUtil: {psutil.__version__}")
    
    print(f"\\n🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"🎮 Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
except Exception as e:
    print(f"❌ Error en verificación: {e}")

print("\\n🎉 ¡Instalación completada exitosamente!")
print("\\n⚠️ IMPORTANTE: Reinicia el runtime ahora")
print("💡 Runtime → Restart runtime")
print("\\nDespués de reiniciar, ejecuta la siguiente celda para verificar.")'''

    return cell_code

def create_memory_optimized_model_cell():
    """Crea la celda de carga de modelos optimizada para memoria."""
    
    cell_code = '''# 🧠 Configuración de Modelos con Optimización de Memoria
print("🚀 Iniciando carga optimizada de modelos InstantID...")

# Configurar optimizaciones de memoria
import os
import gc
import torch
import psutil

# Configurar variables de entorno para optimización
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['TORCH_HOME'] = './torch_cache'
os.environ['HF_HOME'] = './hf_cache'
os.environ['TRANSFORMERS_CACHE'] = './transformers_cache'

def get_memory_info():
    """Obtiene información de memoria."""
    vm = psutil.virtual_memory()
    info = {
        'total': vm.total / (1024**3),
        'available': vm.available / (1024**3),
        'used': vm.used / (1024**3),
        'percent': vm.percent
    }
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.mem_get_info()
        info['gpu_free'] = gpu_memory[0] / (1024**3)
        info['gpu_total'] = gpu_memory[1] / (1024**3)
    return info

def cleanup_memory():
    """Limpia memoria del sistema y GPU."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# Mostrar información inicial
memory = get_memory_info()
print(f"💾 RAM Total: {memory['total']:.2f} GB")
print(f"💾 RAM Disponible: {memory['available']:.2f} GB")
print(f"💾 RAM Usada: {memory['used']:.2f} GB ({memory['percent']:.1f}%)")

if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"🎮 VRAM Total: {memory['gpu_total']:.2f} GB")
    print(f"🎮 VRAM Libre: {memory['gpu_free']:.2f} GB")

# Verificar memoria suficiente
if memory['available'] < 2.0:
    print("⚠️  Memoria baja. Considera reiniciar el runtime.")
    print("💡 Runtime → Restart runtime")

# Configurar PyTorch para memoria eficiente
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(0.8)  # Usar solo 80% de VRAM
    cleanup_memory()

print(f"📱 Dispositivo: {device}")
print(f"🔢 Tipo de datos: {dtype}")

print("\\n📁 Creando directorios...")
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("checkpoints/ControlNetModel", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("\\n📥 Descargando modelos necesarios...")
from huggingface_hub import hf_hub_download

model_files = [
    {"filename": "ControlNetModel/config.json", "repo_id": "InstantX/InstantID"},
    {"filename": "ControlNetModel/diffusion_pytorch_model.safetensors", "repo_id": "InstantX/InstantID"},
    {"filename": "ip-adapter.bin", "repo_id": "InstantX/InstantID"}
]

for file_info in model_files:
    print(f"Descargando {file_info['filename']}...")
    hf_hub_download(
        repo_id=file_info['repo_id'],
        filename=file_info['filename'],
        local_dir="./checkpoints",
        resume_download=True
    )

print("\\n🧠 Configurando analizador facial...")
import zipfile
from insightface.app import FaceAnalysis
import urllib.request
import shutil

# Descargar el modelo buffalo_l
model_dir = os.path.abspath("./models")
model_name = "buffalo_l"
model_file = os.path.join(model_dir, f"{model_name}.zip")

if not os.path.exists(os.path.join(model_dir, model_name)):
    print(f"Descargando modelo {model_name}...")
    url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
    
    temp_dir = os.path.join(model_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    urllib.request.urlretrieve(url, model_file)
    with zipfile.ZipFile(model_file, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    for file in os.listdir(temp_dir):
        src = os.path.join(temp_dir, file)
        dst = os.path.join(model_dir, file)
        if os.path.exists(dst):
            os.remove(dst)
        shutil.move(src, dst)
    
    os.remove(model_file)
    shutil.rmtree(temp_dir)
    print("Modelo descargado y configurado correctamente.")

# Limpiar memoria antes de cargar modelos
cleanup_memory()

# Configurar tamaño de detección según memoria disponible
memory = get_memory_info()
det_size = (320, 320) if memory['available'] < 3.0 else (640, 640)
if memory['available'] < 3.0:
    print("🔧 Usando configuración de memoria reducida")

print("\\nInicializando analizador facial...")
global app
app = FaceAnalysis(name=model_name, root=model_dir, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=det_size)
print("Analizador facial inicializado correctamente.")

# Limpiar memoria entre cargas
cleanup_memory()

print("\\n🎛️  Cargando ControlNet...")
from diffusers.models import ControlNetModel

global controlnet
controlnet = ControlNetModel.from_pretrained(
    'checkpoints/ControlNetModel',
    torch_dtype=dtype,
    use_safetensors=True,
    low_cpu_mem_usage=True  # Optimización de memoria
)

# Limpiar memoria antes del pipeline principal
cleanup_memory()

# Verificar memoria antes de cargar el pipeline principal
memory = get_memory_info()
if memory['available'] < 1.0:
    print("❌ Memoria insuficiente para cargar el pipeline principal")
    print("💡 Intenta reiniciar el runtime o usar Colab Pro")
else:
    print("\\n🚀 Cargando pipeline principal...")
    from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline

    global pipe
    pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
        feature_extractor=None,
        variant="fp16" if device == "cuda" else None,
        low_cpu_mem_usage=True,  # Optimización de memoria
        use_safetensors=True
    )

    # Aplicar optimizaciones de memoria específicas
    if device == "cuda":
        print("🔧 Aplicando optimizaciones GPU...")
        pipe.enable_model_cpu_offload()  # Mover modelos a CPU cuando no se usen
        pipe.enable_vae_slicing()        # Procesar VAE en chunks
        pipe.enable_vae_tiling()         # Procesar VAE en tiles
        pipe.enable_attention_slicing()  # Reducir memoria de atención
        
        # Configuración adicional para memoria muy limitada
        memory = get_memory_info()
        if memory['available'] < 2.0:
            pipe.enable_sequential_cpu_offload()  # Offload secuencial más agresivo
            print("🔧 Modo de memoria ultra-conservador activado")

    print("\\n🔌 Cargando IP-Adapter...")
    pipe.load_ip_adapter_instantid('checkpoints/ip-adapter.bin')

    # Limpieza final
    cleanup_memory()

    print("\\n🎉 ¡Todos los modelos han sido cargados correctamente!")
    
    # Mostrar estado final de memoria
    final_memory = get_memory_info()
    print(f"📊 Memoria final disponible: {final_memory['available']:.2f} GB")
    if torch.cuda.is_available():
        print(f"📊 VRAM final libre: {final_memory['gpu_free']:.2f} GB")
    
    print("\\nVerificando variables globales:")
    print(f"Pipeline disponible: {'pipe' in globals()}")
    print(f"Analizador facial disponible: {'app' in globals()}")
    print(f"ControlNet disponible: {'controlnet' in globals()}")
    
    print("\\n🎯 ¡Listo para generar imágenes!")'''

    return cell_code

def update_notebook_completely():
    """Actualiza el notebook completamente con ambas mejoras."""
    
    # Leer el notebook
    with open('InstantID_Gradio.ipynb', 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Buscar y actualizar la celda de instalación
    installation_cell_index = None
    for i, cell in enumerate(notebook['cells']):
        if (cell['cell_type'] == 'code' and 
            'source' in cell and 
            any('Instalación base' in line or 'torch==2.0.1' in line for line in cell['source'])):
            installation_cell_index = i
            break
    
    if installation_cell_index is not None:
        # Actualizar celda de instalación
        new_installation_code = create_updated_installation_cell()
        notebook['cells'][installation_cell_index]['source'] = new_installation_code.split('\n')
        print(f"✅ Celda de instalación actualizada (posición {installation_cell_index})")
    else:
        print("❌ No se encontró la celda de instalación")
        return False
    
    # Buscar y actualizar la celda de configuración de modelos
    model_config_cell_index = None
    for i, cell in enumerate(notebook['cells']):
        if (cell['cell_type'] == 'code' and 
            'source' in cell and 
            any('Configurando modelos' in line or 'Descargando modelos' in line for line in cell['source'])):
            model_config_cell_index = i
            break
    
    if model_config_cell_index is not None:
        # Actualizar celda de modelos
        new_model_code = create_memory_optimized_model_cell()
        notebook['cells'][model_config_cell_index]['source'] = new_model_code.split('\n')
        print(f"✅ Celda de configuración de modelos actualizada (posición {model_config_cell_index})")
    else:
        print("❌ No se encontró la celda de configuración de modelos")
        return False
    
    # Guardar el notebook actualizado
    with open('InstantID_Gradio.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print("✅ Notebook actualizado completamente")
    return True

if __name__ == "__main__":
    success = update_notebook_completely()
    if success:
        print("\\n🎉 Actualización completa exitosa!")
        print("📝 Cambios realizados:")
        print("   • Celda de instalación: Versiones compatibles con PyTorch 2.6+")
        print("   • Celda de modelos: Optimizaciones completas de memoria")
        print("   • Sin versiones fijas problemáticas")
        print("   • Configuración adaptativa automática")
        print("   • Monitoreo en tiempo real de memoria")
    else:
        print("\\n❌ Error en la actualización")
        sys.exit(1) 