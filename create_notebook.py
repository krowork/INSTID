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

# Desinstalar paquetes conflictivos del sistema
print("\\n1. Limpiando entorno...")
!pip uninstall -y torch torchvision torchaudio transformers diffusers accelerate safetensors numpy websockets insightface opencv-python gradio controlnet-aux huggingface_hub onnx onnxruntime timm datasets tsfresh dask-cudf-cu12 raft-dask-cu12

# Limpiar la caché y archivos temporales
print("\\n2. Limpiando caché y archivos temporales...")
!pip cache purge
!rm -rf ~/.cache/pip
!rm -rf /tmp/pip-*
!rm -rf ~/.cache/huggingface
!rm -rf ~/.cache/torch
!rm -rf ~/.cache/clip

# Crear un entorno limpio
print("\\n3. Configurando entorno...")
import os
os.environ['TORCH_HOME'] = './torch_home'
os.environ['HF_HOME'] = './hf_home'
!mkdir -p ./torch_home ./hf_home

# Instalación base
print("\\n4. Instalando dependencias base...")
!pip install --no-cache-dir torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
!pip install --no-cache-dir numpy==1.26.0

# Verificar instalación de PyTorch
print("\\n5. Verificando instalación de PyTorch...")
import torch
import numpy as np
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"NumPy version: {np.__version__}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA no está disponible después de la instalación")

# Instalación de dependencias principales
print("\\n6. Instalando dependencias principales...")
!pip install --no-cache-dir -U huggingface_hub==0.19.4
!pip install --no-cache-dir -U transformers==4.36.2
!pip install --no-cache-dir -U diffusers==0.24.0
!pip install --no-cache-dir -U accelerate==0.25.0
!pip install --no-cache-dir -U safetensors==0.4.1

# Instalación de dependencias adicionales
print("\\n7. Instalando dependencias adicionales...")
!pip install --no-cache-dir websockets==11.0.3
!pip install --no-cache-dir opencv-python==4.8.0.74
!pip install --no-cache-dir insightface==0.7.3
!pip install --no-cache-dir controlnet_aux==0.0.7
!pip install --no-cache-dir onnx
!pip install --no-cache-dir onnxruntime-gpu
!pip install --no-cache-dir gradio==4.19.2

# Verificación final
print("\\n8. Verificación final de instalaciones...")
try:
    import huggingface_hub
    print(f"✓ huggingface_hub: {huggingface_hub.__version__}")
    import transformers
    print(f"✓ transformers: {transformers.__version__}")
    from diffusers import __version__ as diffusers_version
    print(f"✓ diffusers: {diffusers_version}")
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - Dispositivo: {torch.cuda.get_device_name(0)}")
        print(f"  - Memoria total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
except Exception as e:
    print(f"✗ Error en verificación: {str(e)}")

print("\\n⚠️ ¡IMPORTANTE!")
print("1. ES NECESARIO reiniciar el entorno de ejecución: Runtime -> Restart runtime")
print("2. Después de reiniciar, ejecuta la siguiente celda para verificar la instalación")'''))

# Celda de verificación (separada)
nb.cells.append(nbf.v4.new_code_cell('''# Suprimir advertencias no críticas
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

print("Verificando instalación...")

# Verificar versiones base
import sys
import torch
import numpy as np

print("\\nVersiones básicas:")
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"NumPy: {np.__version__}")

# Verificar GPU
if not torch.cuda.is_available():
    raise RuntimeError("\\n❌ No se detectó GPU. Este notebook requiere una GPU para funcionar.")

print(f"\\n✅ GPU detectada: {torch.cuda.get_device_name(0)}")
print(f"Memoria GPU total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"Memoria GPU disponible: {torch.cuda.mem_get_info()[0] / 1024**3:.1f} GB")
print(f"CUDA versión: {torch.version.cuda}")
print(f"cuDNN versión: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'No disponible'}")

# Verificar CUDA
print("\\nVerificando CUDA...")
try:
    x = torch.rand(5,3).cuda()
    y = torch.matmul(x, x.t())
    print("✅ CUDA está funcionando correctamente")
except Exception as e:
    print(f"❌ Error al usar CUDA: {e}")

# Verificar dependencias críticas
print("\\nVerificando dependencias críticas...")
import huggingface_hub
print(f"✓ huggingface_hub: {huggingface_hub.__version__}")

import onnxruntime
print(f"✓ onnxruntime: {onnxruntime.__version__}")

# Verificar dependencias principales
print("\\nVerificando dependencias principales...")
from diffusers import __version__ as diffusers_version
print(f"✓ diffusers: {diffusers_version}")

import transformers
print(f"✓ transformers: {transformers.__version__}")

import cv2
print(f"✓ opencv-python: {cv2.__version__}")

import insightface
print(f"✓ insightface: {insightface.__version__}")

import controlnet_aux
print(f"✓ controlnet_aux: {controlnet_aux.__version__}")

import gradio as gr
print(f"✓ gradio: {gr.__version__}")

print("\\nVerificación completada.")

# Verificar que las versiones sean las esperadas
expected_versions = {
    'huggingface_hub': '0.19.4',
    'transformers': '4.36.2',
    'diffusers': '0.24.0',
    'torch': '2.0.1+cu118',
    'numpy': '1.26.0',
    'opencv-python': '4.8.0',  # Actualizado para ser más flexible
    'insightface': '0.7.3',
    'controlnet_aux': '0.0.7',
    'gradio': '4.19.2'
}

def version_matches(current, expected):
    """Compara versiones con cierta flexibilidad."""
    if current == expected:
        return True
    # Para opencv, comparamos solo los primeros tres números de versión
    if 'opencv' in current:
        current_parts = current.split('.')[:3]
        expected_parts = expected.split('.')[:3]
        return current_parts == expected_parts
    return False

print("\\nVerificando versiones esperadas:")
all_correct = True
for package, expected_version in expected_versions.items():
    if package == 'torch':
        current_version = torch.__version__
    elif package == 'numpy':
        current_version = np.__version__
    elif package == 'opencv-python':
        current_version = cv2.__version__
    elif package == 'huggingface_hub':
        current_version = huggingface_hub.__version__
    elif package == 'transformers':
        current_version = transformers.__version__
    elif package == 'diffusers':
        current_version = diffusers_version
    elif package == 'insightface':
        current_version = insightface.__version__
    elif package == 'controlnet_aux':
        current_version = controlnet_aux.__version__
    elif package == 'gradio':
        current_version = gr.__version__

    if not version_matches(current_version, expected_version):
        print(f"⚠️ {package}: versión actual {current_version}, esperada {expected_version}")
        all_correct = False
    else:
        print(f"✅ {package}: {current_version}")

if all_correct:
    print("\\n✅ Todas las versiones son correctas!")
    print("\\n🚀 El entorno está listo para usar InstantID!")
else:
    print("\\n⚠️ Algunas versiones no coinciden con las esperadas.")
    print("Si todo funciona correctamente, puedes continuar. Si encuentras problemas, considera reinstalar las dependencias.")'''))

# Celda para descargar el pipeline
nb.cells.append(nbf.v4.new_markdown_cell('''## Descarga del Pipeline
Primero necesitamos descargar el archivo del pipeline de InstantID.'''))

nb.cells.append(nbf.v4.new_code_cell('''# Descargar el archivo del pipeline
!wget https://raw.githubusercontent.com/InstantID/InstantID/main/pipeline_stable_diffusion_xl_instantid.py'''))

# Celda de importaciones
nb.cells.append(nbf.v4.new_markdown_cell('''## Importar Dependencias
Ejecuta esta celda después de reiniciar el entorno de ejecución'''))

nb.cells.append(nbf.v4.new_code_cell('''import os
import cv2
import torch
import numpy as np
from PIL import Image
import gradio as gr
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from insightface.app import FaceAnalysis
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
print(f"diffusers: {diffusers.__version__}")
print(f"torch: {torch.__version__}")'''))

# Celda de configuración de modelos
nb.cells.append(nbf.v4.new_markdown_cell('''## Configuración de Modelos
En esta sección vamos a:
1. Descargar los modelos necesarios
2. Configurar el pipeline de InstantID
3. Preparar el analizador facial'''))

nb.cells.append(nbf.v4.new_code_cell('''# Crear directorios para checkpoints
import os
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("checkpoints/ControlNetModel", exist_ok=True)

# Descargar modelos necesarios
print("Descargando modelos necesarios...")
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

# Configurar el analizador facial
print("\\nConfigurando analizador facial...")
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Cargar ControlNet
print("\\nCargando ControlNet...")
import torch
from diffusers.models import ControlNetModel

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

controlnet = ControlNetModel.from_pretrained(
    'checkpoints/ControlNetModel',
    torch_dtype=dtype,
    use_safetensors=True
)

# Cargar el pipeline principal
print("\\nCargando pipeline principal...")
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline

pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=dtype,
    safety_checker=None,
    feature_extractor=None
)

if device == "cuda":
    pipe.cuda()
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_tiling()

# Cargar IP-Adapter
print("\\nCargando IP-Adapter...")
pipe.load_ip_adapter_instantid('checkpoints/ip-adapter.bin')

print("\\n✅ ¡Todos los modelos han sido cargados correctamente!")
print("Ahora puedes proceder a generar imágenes.")'''))

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

nb.cells.append(nbf.v4.new_code_cell('''def process_image(image, prompt, num_steps, identitynet_strength, adapter_strength):
    """Procesa la imagen para la interfaz Gradio."""
    temp_path = "temp_face.png"
    if isinstance(image, str):
        temp_path = image
    else:
        image.save(temp_path)
    
    try:
        result = generate_image(
            face_image_path=temp_path,
            prompt=prompt,
            num_steps=int(num_steps),
            identitynet_strength_ratio=float(identitynet_strength),
            adapter_strength_ratio=float(adapter_strength)
        )
        return result
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        if isinstance(image, Image.Image) and os.path.exists(temp_path):
            os.remove(temp_path)

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

# Lanzar la interfaz
demo.launch(share=True)'''))

# Celda de limpieza
nb.cells.append(nbf.v4.new_markdown_cell('## Limpieza de Memoria'))

nb.cells.append(nbf.v4.new_code_cell('''if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("Memoria GPU liberada")'''))

# Guardar el notebook
with open('InstantID_Gradio.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f) 