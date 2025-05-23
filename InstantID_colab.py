# -*- coding: utf-8 -*-
"""
# InstantID: Zero-shot Identity-Preserving Generation in Seconds

Este script implementa [InstantID](https://github.com/InstantX/InstantID), un método para generar imágenes 
que preservan la identidad de una persona en segundos.
"""

# Instalación de dependencias
!pip install -q torch==2.0.1 torchvision==0.15.2 diffusers==0.33.1 transformers==4.38.2 accelerate==0.28.0
!pip install -q safetensors==0.4.2 einops==0.7.0 onnxruntime==1.17.1 omegaconf==2.3.0 peft==0.9.0
!pip install -q huggingface-hub==0.21.4 opencv-python==4.9.0.80 insightface==0.7.3

# Clonar nuestro repositorio
!git clone https://github.com/krowork/INSTID.git
%cd INSTID

"""## Descargar Modelos Necesarios"""

import os
from huggingface_hub import hf_hub_download

# Crear directorio para checkpoints
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("checkpoints/ControlNetModel", exist_ok=True)

# Descargar modelos necesarios
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

"""## Importar Dependencias y Configurar el Modelo"""

import cv2
import torch
import numpy as np
from PIL import Image
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from insightface.app import FaceAnalysis
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps

# Configurar dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Usando dispositivo: {device}")

# Inicializar el analizador facial
print("Inicializando analizador facial...")
app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Cargar ControlNet
print("Cargando ControlNet...")
controlnet = ControlNetModel.from_pretrained(
    'checkpoints/ControlNetModel',
    torch_dtype=torch_dtype,
    use_safetensors=True
)

# Cargar el pipeline
print("Cargando pipeline principal...")
pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch_dtype,
    safety_checker=None,
    feature_extractor=None
)

if device == "cuda":
    pipe.cuda()
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_tiling()

# Cargar IP-Adapter
print("Cargando IP-Adapter...")
pipe.load_ip_adapter_instantid('checkpoints/ip-adapter.bin')

print("¡Configuración completada!")

"""## Función de Generación de Imágenes"""

def generate_image(face_image_path, prompt, negative_prompt=None, num_steps=30, identitynet_strength_ratio=0.80, adapter_strength_ratio=0.80):
    """Genera una imagen usando InstantID.
    
    Args:
        face_image_path (str): Ruta a la imagen del rostro
        prompt (str): Descripción de la imagen a generar
        negative_prompt (str, opcional): Prompt negativo
        num_steps (int): Número de pasos de inferencia
        identitynet_strength_ratio (float): Fuerza de IdentityNet (0-1)
        adapter_strength_ratio (float): Fuerza del adaptador (0-1)
    
    Returns:
        PIL.Image: Imagen generada
    """
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
    return image

"""## Ejemplo de Uso"""

# Código para usar en Colab:
from google.colab import files
import ipywidgets as widgets

print("Por favor, sube una imagen con un rostro claro y visible...")
uploaded = files.upload()
image_path = next(iter(uploaded.keys()))

# Mostrar la imagen subida
display(Image.open(image_path))

"""### Configurar Parámetros y Generar Imagen"""

# Widgets para configurar parámetros
prompt_widget = widgets.Text(
    value='analog film photo of a person in a cyberpunk city, neon lights, cinematic lighting',
    description='Prompt:',
    style={'description_width': 'initial'},
    layout={'width': '100%'}
)

steps_widget = widgets.IntSlider(
    value=30,
    min=20,
    max=100,
    step=1,
    description='Pasos:',
)

identity_strength_widget = widgets.FloatSlider(
    value=0.80,
    min=0.0,
    max=1.5,
    step=0.05,
    description='Fuerza IdentityNet:',
)

adapter_strength_widget = widgets.FloatSlider(
    value=0.80,
    min=0.0,
    max=1.5,
    step=0.05,
    description='Fuerza Adapter:',
)

display(prompt_widget, steps_widget, identity_strength_widget, adapter_strength_widget)

# Generar imagen con los parámetros configurados
generated_image = generate_image(
    face_image_path=image_path,
    prompt=prompt_widget.value,
    num_steps=steps_widget.value,
    identitynet_strength_ratio=identity_strength_widget.value,
    adapter_strength_ratio=adapter_strength_widget.value
)

# Mostrar la imagen generada
display(generated_image)

"""## Limpieza de Memoria"""

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("Memoria GPU liberada") 