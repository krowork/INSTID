import nbformat as nbf

# Crear un nuevo notebook
nb = nbf.v4.new_notebook()

# Celda de título y descripción
nb.cells.append(nbf.v4.new_markdown_cell('''# InstantID: Zero-shot Identity-Preserving Generation in Seconds

Este script implementa [InstantID](https://github.com/InstantX/InstantID), un método para generar imágenes 
que preservan la identidad de una persona en segundos.

⚠️ **Importante**: Asegúrate de seleccionar un entorno de ejecución con GPU: Runtime -> Change runtime type -> GPU'''))

# Celda para verificar GPU
nb.cells.append(nbf.v4.new_code_cell('''# Verificar que tenemos GPU disponible
!nvidia-smi

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"GPU disponible: {torch.cuda.is_available()}")
print(f"Dispositivo CUDA: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No hay GPU'}")

if not torch.cuda.is_available():
    raise RuntimeError("No se detectó GPU. Por favor, selecciona un entorno de ejecución con GPU: Runtime -> Change runtime type -> GPU")'''))

# Celda para clonar el repositorio
nb.cells.append(nbf.v4.new_code_cell('''# Clonar el repositorio
!git clone https://github.com/krowork/INSTID.git
%cd INSTID'''))

# Celda de instalación
nb.cells.append(nbf.v4.new_code_cell('''# Asegurar que estamos en un entorno con GPU
import torch
if not torch.cuda.is_available():
    raise RuntimeError("\\n❌ Este notebook requiere una GPU. Por favor, selecciona: Runtime -> Change runtime type -> GPU")

print("\\n🔧 Iniciando instalación optimizada de dependencias...")
print("Este proceso puede tomar varios minutos y mostrará algunos warnings - esto es normal.")

# Configurar variables de entorno para evitar conflictos
import os
os.environ['TORCH_HOME'] = './torch_home'
os.environ['HF_HOME'] = './hf_home'
os.environ['PIP_NO_WARN_SCRIPT_LOCATION'] = '1'
!mkdir -p ./torch_home ./hf_home

print("\\n1️⃣ Instalando PyTorch y dependencias base...")
# Instalar PyTorch primero con versiones específicas
!pip install --quiet --no-warn-script-location torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Verificar PyTorch inmediatamente
import torch
print(f"✅ PyTorch {torch.__version__} instalado correctamente")
print(f"✅ CUDA disponible: {torch.cuda.is_available()}")

print("\\n2️⃣ Instalando numpy con versión compatible...")
!pip install --quiet --no-warn-script-location numpy==1.26.0

print("\\n3️⃣ Instalando dependencias principales...")
# Instalar dependencias principales una por una para evitar conflictos
!pip install --quiet --no-warn-script-location huggingface_hub==0.19.4
!pip install --quiet --no-warn-script-location transformers==4.36.2
!pip install --quiet --no-warn-script-location diffusers==0.24.0
!pip install --quiet --no-warn-script-location accelerate==0.25.0
!pip install --quiet --no-warn-script-location safetensors==0.4.1

print("\\n4️⃣ Instalando dependencias de procesamiento de imágenes...")
!pip install --quiet --no-warn-script-location opencv-python==4.8.0.74
!pip install --quiet --no-warn-script-location Pillow

print("\\n5️⃣ Instalando dependencias de ML...")
!pip install --quiet --no-warn-script-location insightface==0.7.3
!pip install --quiet --no-warn-script-location onnx
!pip install --quiet --no-warn-script-location onnxruntime-gpu

print("\\n6️⃣ Instalando ControlNet y utilidades...")
!pip install --quiet --no-warn-script-location controlnet_aux==0.0.7

print("\\n7️⃣ Instalando dependencias de interfaz...")
!pip install --quiet --no-warn-script-location websockets==11.0.3
!pip install --quiet --no-warn-script-location gradio==4.19.2

print("\\n8️⃣ Verificación de instalación...")
try:
    # Verificar importaciones críticas
    import torch
    import numpy as np
    import cv2
    import transformers
    from diffusers import __version__ as diffusers_version
    import huggingface_hub
    import insightface
    import controlnet_aux
    import gradio as gr
    import accelerate
    import safetensors
    
    print("\\n✅ Verificación de versiones:")
    print(f"  • PyTorch: {torch.__version__}")
    print(f"  • NumPy: {np.__version__}")
    print(f"  • Transformers: {transformers.__version__}")
    print(f"  • Diffusers: {diffusers_version}")
    print(f"  • HuggingFace Hub: {huggingface_hub.__version__}")
    print(f"  • OpenCV: {cv2.__version__}")
    print(f"  • InsightFace: {insightface.__version__}")
    print(f"  • ControlNet Aux: {controlnet_aux.__version__}")
    print(f"  • Gradio: {gr.__version__}")
    
    # Verificar CUDA
    if torch.cuda.is_available():
        print(f"\\n🚀 GPU detectada: {torch.cuda.get_device_name(0)}")
        print(f"   Memoria total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Test básico de CUDA
        x = torch.rand(5, 3).cuda()
        y = torch.matmul(x, x.t())
        print("   ✅ Test de CUDA exitoso")
    else:
        raise RuntimeError("❌ CUDA no está disponible")
    
    print("\\n🎉 ¡Instalación completada exitosamente!")
    print("\\n⚠️  IMPORTANTE:")
    print("   1. DEBES reiniciar el entorno: Runtime → Restart runtime")
    print("   2. Después del reinicio, ejecuta la siguiente celda de verificación")
    print("   3. Los warnings sobre conflictos de dependencias son normales y no afectan el funcionamiento")
    
except ImportError as e:
    print(f"\\n❌ Error de importación: {e}")
    print("\\nIntenta ejecutar esta celda nuevamente. Si el problema persiste:")
    print("1. Reinicia el entorno: Runtime → Restart runtime")
    print("2. Ejecuta esta celda de nuevo")
    
except Exception as e:
    print(f"\\n❌ Error inesperado: {e}")
    print("\\nPor favor, reinicia el entorno y ejecuta la celda nuevamente.")'''))

# Celda de verificación (separada y mejorada)
nb.cells.append(nbf.v4.new_code_cell('''# ===== EJECUTAR ESTA CELDA DESPUÉS DE REINICIAR EL ENTORNO =====

print("🔍 Verificando instalación después del reinicio...")

# Suprimir warnings no críticos
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

try:
    # Verificar versiones base
    import sys
    import torch
    import numpy as np
    
    print("\\n📋 Información del sistema:")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   NumPy: {np.__version__}")
    
    # Verificar GPU
    if not torch.cuda.is_available():
        raise RuntimeError("❌ No se detectó GPU. Este notebook requiere una GPU para funcionar.")
    
    print(f"\\n🎮 GPU detectada: {torch.cuda.get_device_name(0)}")
    print(f"   Memoria total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"   Memoria disponible: {torch.cuda.mem_get_info()[0] / 1024**3:.1f} GB")
    print(f"   CUDA versión: {torch.version.cuda}")
    
    # Test de CUDA
    print("\\n🧪 Probando CUDA...")
    x = torch.rand(5, 3).cuda()
    y = torch.matmul(x, x.t())
    del x, y  # Liberar memoria
    torch.cuda.empty_cache()
    print("   ✅ CUDA funciona correctamente")
    
    # Verificar dependencias críticas
    print("\\n📦 Verificando dependencias críticas...")
    
    import huggingface_hub
    print(f"   ✅ huggingface_hub: {huggingface_hub.__version__}")
    
    import transformers
    print(f"   ✅ transformers: {transformers.__version__}")
    
    from diffusers import __version__ as diffusers_version
    print(f"   ✅ diffusers: {diffusers_version}")
    
    import cv2
    print(f"   ✅ opencv-python: {cv2.__version__}")
    
    import insightface
    print(f"   ✅ insightface: {insightface.__version__}")
    
    import controlnet_aux
    print(f"   ✅ controlnet_aux: {controlnet_aux.__version__}")
    
    import gradio as gr
    print(f"   ✅ gradio: {gr.__version__}")
    
    import accelerate
    print(f"   ✅ accelerate: {accelerate.__version__}")
    
    import safetensors
    print(f"   ✅ safetensors: {safetensors.__version__}")
    
    # Verificar que podemos importar componentes específicos
    print("\\n🔧 Verificando componentes específicos...")
    from diffusers.models import ControlNetModel
    print("   ✅ ControlNetModel importado")
    
    from diffusers.utils import load_image
    print("   ✅ load_image importado")
    
    from insightface.app import FaceAnalysis
    print("   ✅ FaceAnalysis importado")
    
    print("\\n🎉 ¡Verificación completada exitosamente!")
    print("\\n🚀 El entorno está listo para usar InstantID!")
    print("\\nPuedes continuar con la siguiente celda para configurar los modelos.")
    
except ImportError as e:
    print(f"\\n❌ Error de importación: {e}")
    print("\\n🔄 Soluciones:")
    print("1. Asegúrate de haber reiniciado el entorno después de la instalación")
    print("2. Si acabas de reiniciar, ejecuta la celda de instalación nuevamente")
    print("3. Verifica que seleccionaste un entorno con GPU: Runtime → Change runtime type → GPU")
    
except RuntimeError as e:
    print(f"\\n❌ Error de GPU: {e}")
    print("\\n🔄 Solución:")
    print("Asegúrate de seleccionar un entorno con GPU: Runtime → Change runtime type → GPU")
    
except Exception as e:
    print(f"\\n❌ Error inesperado: {e}")
    print("\\n🔄 Soluciones:")
    print("1. Reinicia el entorno: Runtime → Restart runtime")
    print("2. Ejecuta la celda de instalación nuevamente")
    print("3. Si el problema persiste, intenta con un nuevo notebook")'''))

# Celda de importaciones
nb.cells.append(nbf.v4.new_markdown_cell('''## Importar Dependencias
Ejecuta esta celda después de reiniciar el entorno de ejecución'''))

nb.cells.append(nbf.v4.new_code_cell('''import os
import cv2
import torch
import numpy as np
from PIL import Image
import gradio as gr
import transformers
from diffusers import __version__ as diffusers_version
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from insightface.app import FaceAnalysis
# Importar el pipeline desde el repositorio clonado
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps
from huggingface_hub import hf_hub_download

# Verificar que todo está correcto
print(f"PyTorch CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Dispositivo CUDA: {torch.cuda.get_device_name(0)}")
    print(f"Memoria GPU total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print(f"\\nVersiones de las dependencias:")
print(f"numpy: {np.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"diffusers: {diffusers_version}")
print(f"torch: {torch.__version__}")'''))

# Celda de configuración de modelos
nb.cells.append(nbf.v4.new_markdown_cell('''## Configuración de Modelos
En esta sección vamos a:
1. Descargar los modelos necesarios
2. Configurar el pipeline de InstantID
3. Preparar el analizador facial'''))

nb.cells.append(nbf.v4.new_code_cell('''print("🔧 Configurando modelos de InstantID...")

# Importaciones necesarias
import os
import zipfile
import urllib.request
import shutil
import torch
from huggingface_hub import hf_hub_download
from diffusers.models import ControlNetModel
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline
from insightface.app import FaceAnalysis
from diffusers.utils import logging
import warnings

# Configurar logging y warnings
logging.set_verbosity_error()
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Configurar CUDA y memoria
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

if device == "cuda":
    torch.cuda.empty_cache()
    print(f"\\n🎮 Usando GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memoria total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"   Memoria disponible inicial: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB")
else:
    print("\\n⚠️  Usando CPU (no recomendado)")

# Crear directorios necesarios
print("\\n📁 Creando directorios...")
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("checkpoints/ControlNetModel", exist_ok=True)
os.makedirs("models", exist_ok=True)
print("   ✅ Directorios creados")

# Descargar modelos necesarios
print("\\n📥 Descargando modelos de InstantID...")
model_files = [
    {"filename": "ControlNetModel/config.json", "repo_id": "InstantX/InstantID", "desc": "Configuración ControlNet"},
    {"filename": "ControlNetModel/diffusion_pytorch_model.safetensors", "repo_id": "InstantX/InstantID", "desc": "Modelo ControlNet"},
    {"filename": "ip-adapter.bin", "repo_id": "InstantX/InstantID", "desc": "IP-Adapter"}
]

for i, file_info in enumerate(model_files, 1):
    file_path = os.path.join("checkpoints", file_info["filename"])
    if not os.path.exists(file_path):
        print(f"   {i}/3 Descargando {file_info['desc']}...")
        try:
            hf_hub_download(
                repo_id=file_info['repo_id'],
                filename=file_info['filename'],
                local_dir="./checkpoints",
                resume_download=True
            )
            print(f"       ✅ {file_info['desc']} descargado")
        except Exception as e:
            print(f"       ❌ Error descargando {file_info['desc']}: {e}")
            raise
    else:
        print(f"   {i}/3 ✅ {file_info['desc']} ya existe")

# Configurar el analizador facial
print("\\n👤 Configurando analizador facial...")
model_dir = os.path.abspath("./models")
model_name = "buffalo_l"
model_path = os.path.join(model_dir, model_name)

if not os.path.exists(model_path):
    print("   📥 Descargando modelo facial buffalo_l...")
    model_file = os.path.join(model_dir, f"{model_name}.zip")
    url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
    
    try:
        # Crear directorio temporal
        temp_dir = os.path.join(model_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Descargar archivo
        urllib.request.urlretrieve(url, model_file)
        print("   📦 Extrayendo modelo...")
        
        # Extraer archivo
        with zipfile.ZipFile(model_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Mover archivos al directorio final
        for file in os.listdir(temp_dir):
            src = os.path.join(temp_dir, file)
            dst = os.path.join(model_dir, file)
            if os.path.exists(dst):
                os.remove(dst)
            shutil.move(src, dst)
        
        # Limpiar archivos temporales
        os.remove(model_file)
        shutil.rmtree(temp_dir)
        print("   ✅ Modelo facial descargado y configurado")
        
    except Exception as e:
        print(f"   ❌ Error descargando modelo facial: {e}")
        raise
else:
    print("   ✅ Modelo facial ya existe")

# Inicializar el analizador facial
print("\\n🧠 Inicializando analizador facial...")
try:
    app = FaceAnalysis(name=model_name, root=model_dir, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("   ✅ Analizador facial inicializado")
except Exception as e:
    print(f"   ❌ Error inicializando analizador facial: {e}")
    raise

if device == "cuda":
    print(f"   Memoria disponible después del analizador: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB")

# Cargar ControlNet
print("\\n🎛️  Cargando ControlNet...")
try:
    controlnet = ControlNetModel.from_pretrained(
        'checkpoints/ControlNetModel',
        torch_dtype=dtype,
        use_safetensors=True
    )
    print("   ✅ ControlNet cargado")
except Exception as e:
    print(f"   ❌ Error cargando ControlNet: {e}")
    raise

if device == "cuda":
    print(f"   Memoria disponible después de ControlNet: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB")

# Cargar el pipeline principal
print("\\n🚀 Cargando pipeline principal...")
try:
    pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
        feature_extractor=None,
        variant="fp16" if device == "cuda" else None,
    )
    print("   ✅ Pipeline base cargado")
    
    # Aplicar optimizaciones de memoria para GPU
    if device == "cuda":
        print("   🔧 Aplicando optimizaciones de memoria...")
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        print("   ✅ Optimizaciones aplicadas")
        
except Exception as e:
    print(f"   ❌ Error cargando pipeline: {e}")
    raise

if device == "cuda":
    print(f"   Memoria disponible después del pipeline: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB")

# Cargar IP-Adapter
print("\\n🔗 Cargando IP-Adapter...")
try:
    pipe.load_ip_adapter_instantid('checkpoints/ip-adapter.bin')
    print("   ✅ IP-Adapter cargado")
except Exception as e:
    print(f"   ❌ Error cargando IP-Adapter: {e}")
    raise

# Limpieza final de memoria
if device == "cuda":
    torch.cuda.empty_cache()
    print(f"\\n🧹 Memoria final disponible: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB")

print("\\n🎉 ¡Configuración completada exitosamente!")
print("\\n📋 Resumen de componentes cargados:")
print("   ✅ Analizador facial (InsightFace)")
print("   ✅ ControlNet para InstantID")
print("   ✅ Pipeline Stable Diffusion XL")
print("   ✅ IP-Adapter")
print("\\n🚀 ¡Listo para generar imágenes!")

# Verificar que las variables globales están disponibles
print("\\n🔍 Verificando variables globales:")
print(f"   Pipeline disponible: {'pipe' in globals() and pipe is not None}")
print(f"   Analizador facial disponible: {'app' in globals() and app is not None}")
print(f"   ControlNet disponible: {'controlnet' in globals() and controlnet is not None}")

if not all(['pipe' in globals(), 'app' in globals(), 'controlnet' in globals()]):
    raise RuntimeError("❌ Error: No todas las variables globales están disponibles")

print("\\n✅ Todas las variables globales están correctamente configuradas")'''))

# Celda de función de generación
nb.cells.append(nbf.v4.new_markdown_cell('## Función de Generación de Imágenes'))

nb.cells.append(nbf.v4.new_code_cell('''def generate_image(face_image_path, prompt, negative_prompt=None, num_steps=30, identitynet_strength_ratio=0.80, adapter_strength_ratio=0.80):
    """Genera una imagen usando InstantID."""
    if negative_prompt is None:
        negative_prompt = "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry"
    
    print("Cargando imagen...")
    face_image = load_image(face_image_path)
    face_image_cv2 = cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR)
    
    print("Detectando rostro...")
    face_info = app.get(face_image_cv2)
    if len(face_info) == 0:
        raise ValueError("No se detectó ningún rostro en la imagen")
    
    face_info = face_info[-1]
    face_emb = face_info['embedding']
    face_kps = draw_kps(face_image, face_info['kps'])
    
    print("Configurando parámetros...")
    pipe.set_ip_adapter_scale(adapter_strength_ratio)
    
    print("Generando imagen...")
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_embeds=face_emb,
        image=face_kps,
        controlnet_conditioning_scale=float(identitynet_strength_ratio),
        num_inference_steps=num_steps,
        guidance_scale=5.0
    ).images[0]
    
    print("¡Generación completada!")
    return image'''))

# Celda de interfaz Gradio
nb.cells.append(nbf.v4.new_markdown_cell('## Interfaz Web con Gradio'))

nb.cells.append(nbf.v4.new_code_cell('''# Verificar que tenemos las variables necesarias
if 'pipe' not in globals():
    raise RuntimeError("El pipeline no está inicializado. Asegúrate de ejecutar la celda de configuración de modelos primero.")
if 'app' not in globals():
    raise RuntimeError("El analizador facial no está inicializado. Asegúrate de ejecutar la celda de configuración de modelos primero.")

# Importaciones necesarias
import os
import cv2
import torch
import numpy as np
from PIL import Image
import gradio as gr
from diffusers.utils import load_image
import logging
import traceback
import sys
from datetime import datetime

# Configurar logging para mostrar en el notebook
class NotebookHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            print(f"{datetime.now().strftime('%H:%M:%S')} - {msg}")
            sys.stdout.flush()
        except Exception:
            self.handleError(record)

# Configurar logger
logger = logging.getLogger('InstantID')
logger.setLevel(logging.INFO)
# Eliminar handlers existentes para evitar duplicados
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
handler = NotebookHandler()
formatter = logging.Formatter('%(levelname)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Verificar el estado inicial
logger.info("=== Verificando estado inicial ===")
logger.info(f"Pipeline disponible: {pipe is not None}")
logger.info(f"Analizador facial disponible: {app is not None}")
logger.info(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"Memoria GPU disponible: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB")

def generate_image(face_image_path, prompt, negative_prompt=None, num_steps=30, identitynet_strength_ratio=0.80, adapter_strength_ratio=0.80):
    """Genera una imagen usando InstantID."""
    try:
        logger.info("Iniciando generación de imagen...")
        logger.info(f"Parámetros: prompt='{prompt}', steps={num_steps}, identity_strength={identitynet_strength_ratio}, adapter_strength={adapter_strength_ratio}")
        
        if negative_prompt is None:
            negative_prompt = "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry"
        
        logger.info("Cargando imagen...")
        face_image = load_image(face_image_path)
        logger.info("Imagen cargada correctamente")
        
        logger.info("Convirtiendo imagen para detección facial...")
        face_image_cv2 = cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR)
        
        logger.info("Detectando rostro...")
        face_info = app.get(face_image_cv2)
        if len(face_info) == 0:
            raise ValueError("No se detectó ningún rostro en la imagen")
        logger.info("Rostro detectado correctamente")
        
        face_info = face_info[-1]
        face_emb = face_info['embedding']
        face_kps = draw_kps(face_image, face_info['kps'])
        logger.info("Puntos faciales extraídos correctamente")
        
        logger.info("Configurando parámetros del pipeline...")
        pipe.set_ip_adapter_scale(adapter_strength_ratio)
        
        logger.info("Generando imagen...")
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_embeds=face_emb,
            image=face_kps,
            controlnet_conditioning_scale=float(identitynet_strength_ratio),
            num_inference_steps=num_steps,
            guidance_scale=5.0
        ).images[0]
        
        logger.info("¡Generación completada!")
        return image
    
    except Exception as e:
        logger.error(f"Error en generate_image: {str(e)}")
        logger.error(f"Traza completa:\\n{traceback.format_exc()}")
        raise

def process_image(image, prompt, num_steps, identitynet_strength, adapter_strength):
    """Procesa la imagen para la interfaz Gradio."""
    try:
        logger.info("=== Iniciando nuevo procesamiento de imagen ===")
        logger.info(f"Tipo de imagen recibida: {type(image)}")
        
        # Verificar variables globales
        if 'pipe' not in globals() or pipe is None:
            raise RuntimeError("Pipeline no disponible")
        if 'app' not in globals() or app is None:
            raise RuntimeError("Analizador facial no disponible")
        
        logger.info("Verificando estado de variables globales...")
        logger.info(f"Pipeline disponible: {pipe is not None}")
        logger.info(f"Analizador facial disponible: {app is not None}")
        logger.info(f"CUDA disponible: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"Memoria GPU disponible: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB")
        
        # Guardar imagen temporal
        temp_path = "temp_face.png"
        if isinstance(image, str):
            temp_path = image
            logger.info(f"Usando ruta de imagen existente: {temp_path}")
        else:
            logger.info("Guardando imagen temporal...")
            image.save(temp_path)
            logger.info(f"Imagen guardada en: {temp_path}")
        
        try:
            logger.info("Llamando a generate_image...")
            result = generate_image(
                face_image_path=temp_path,
                prompt=prompt,
                num_steps=int(num_steps),
                identitynet_strength_ratio=float(identitynet_strength),
                adapter_strength_ratio=float(adapter_strength)
            )
            logger.info("Imagen generada exitosamente")
            return result
        
        except Exception as e:
            error_msg = f"Error durante la generación: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traza completa:\\n{traceback.format_exc()}")
            return gr.Image.update(value=None, label=error_msg)
        
    except Exception as e:
        error_msg = f"Error en el procesamiento: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Traza completa:\\n{traceback.format_exc()}")
        return gr.Image.update(value=None, label=error_msg)
        
    finally:
        # Limpiar archivo temporal
        if isinstance(image, Image.Image) and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info("Archivo temporal eliminado")
            except Exception as e:
                logger.warning(f"No se pudo eliminar el archivo temporal: {str(e)}")

# Crear la interfaz
demo = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="pil", label="Imagen del rostro"),
        gr.Textbox(
            label="Prompt",
            value="analog film photo of a person in a cyberpunk city, neon lights, cinematic lighting"
        ),
        gr.Slider(minimum=20, maximum=100, value=30, step=1, label="Número de pasos"),
        gr.Slider(minimum=0.0, maximum=1.5, value=0.8, step=0.05, label="Fuerza IdentityNet"),
        gr.Slider(minimum=0.0, maximum=1.5, value=0.8, step=0.05, label="Fuerza Adapter")
    ],
    outputs=gr.Image(type="pil", label="Imagen generada"),
    title="InstantID - Generación de Imágenes",
    description="Sube una imagen con un rostro claro y visible, ajusta los parámetros y genera una nueva imagen manteniendo la identidad."
)

logger.info("Interfaz Gradio inicializada. Lista para procesar imágenes.")

# Lanzar la interfaz
demo.launch(debug=True)'''))

# Celda de limpieza
nb.cells.append(nbf.v4.new_markdown_cell('## Limpieza de Memoria'))

nb.cells.append(nbf.v4.new_code_cell('''if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("Memoria GPU liberada")'''))

# Guardar el notebook
with open('InstantID_Gradio.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f) 