import os
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

def setup_models():
    """Configura y carga todos los modelos necesarios."""
    print("Configurando modelos...")
    
    # Configurar dispositivo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Crear directorios para checkpoints
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
    
    return pipe, app

def generate_image(face_image_path, prompt, negative_prompt=None, num_steps=30, identitynet_strength_ratio=0.80, adapter_strength_ratio=0.80, pipe=None, app=None):
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
    return image

def process_image(image, prompt, num_steps, identitynet_strength, adapter_strength, pipe, app):
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
            adapter_strength_ratio=float(adapter_strength),
            pipe=pipe,
            app=app
        )
        return result
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        if isinstance(image, Image.Image) and os.path.exists(temp_path):
            os.remove(temp_path)

def launch_interface():
    """Lanza la interfaz web de Gradio."""
    # Configurar modelos
    pipe, app = setup_models()
    
    # Crear la interfaz
    demo = gr.Interface(
        fn=lambda img, prompt, steps, id_strength, adapter_strength: process_image(
            img, prompt, steps, id_strength, adapter_strength, pipe, app
        ),
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
    demo.launch(share=True)

if __name__ == "__main__":
    launch_interface() 