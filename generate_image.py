import cv2
import numpy as np
from PIL import Image
from diffusers.utils import load_image
from pipeline_stable_diffusion_xl_instantid import draw_kps

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
    from __main__ import pipe, app  # Importamos las variables globales del notebook
    
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