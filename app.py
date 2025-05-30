import os
import cv2
import math
#import spaces
import torch
import random
import numpy as np
import argparse
import logging
from pathlib import Path

import PIL
from PIL import Image

from diffusers import StableDiffusionXLControlNetPipeline
from diffusers.utils import load_image
from diffusers.models import ControlNetModel

import insightface
from insightface.app import FaceAnalysis

from style_template import styles
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline

import gradio as gr

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import LocalEntryNotFoundError, RepositoryNotFoundError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# global variable
MAX_SEED = np.iinfo(np.int32).max
#device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16

# Device configuration with memory optimization settings
def setup_device():
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    elif torch.cuda.is_available():
        # Check available GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # Convert to GB
        logger.info(f"Available GPU memory: {gpu_memory:.2f} GB")
        return "cuda", torch.float16
    else:
        return "cpu", torch.float32

device, torch_dtype = setup_device()
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Watercolor"

# Create checkpoints directory if it doesn't exist
CHECKPOINTS_DIR = Path("./checkpoints")
CHECKPOINTS_DIR.mkdir(exist_ok=True)

def download_model_files():
    """Download required model files with error handling and progress tracking."""
    model_files = [
        {"filename": "ControlNetModel/config.json", "desc": "ControlNet config"},
        {"filename": "ControlNetModel/diffusion_pytorch_model.safetensors", "desc": "ControlNet model"},
        {"filename": "ip-adapter.bin", "desc": "IP-Adapter weights"}
    ]
    
    for file_info in model_files:
        file_path = CHECKPOINTS_DIR / file_info["filename"]
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
        try:
            if not file_path.exists():
                logger.info(f"Downloading {file_info['desc']}...")
                hf_hub_download(
                    repo_id="InstantX/InstantID",
                    filename=file_info["filename"],
                    local_dir="./checkpoints",
                    resume_download=True
                )
                logger.info(f"Successfully downloaded {file_info['desc']}")
            else:
                logger.info(f"{file_info['desc']} already exists, skipping download")
        except (LocalEntryNotFoundError, RepositoryNotFoundError) as e:
            logger.error(f"Error downloading {file_info['desc']}: {str(e)}")
            raise RuntimeError(f"Failed to download required model file: {file_info['filename']}")
        except Exception as e:
            logger.error(f"Unexpected error downloading {file_info['desc']}: {str(e)}")
            raise

# Download model files
download_model_files()

# Load face encoder with error handling
def setup_face_analyzer():
    try:
        app = FaceAnalysis(name='antelopev2', root='./', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        return app
    except Exception as e:
        logger.error(f"Error initializing face analyzer: {str(e)}")
        raise RuntimeError("Failed to initialize face analyzer")

app = setup_face_analyzer()

# Path to InstantID models
face_adapter = str(CHECKPOINTS_DIR / 'ip-adapter.bin')
controlnet_path = str(CHECKPOINTS_DIR / 'ControlNetModel')

# Load pipeline with error handling
def load_controlnet():
    try:
        return ControlNetModel.from_pretrained(
            controlnet_path,
            torch_dtype=torch_dtype,
            safety_checker=None,
            use_safetensors=True
        )
    except Exception as e:
        logger.error(f"Error loading ControlNet model: {str(e)}")
        raise RuntimeError("Failed to load ControlNet model")

controlnet = load_controlnet()

# Memory optimization utilities
def enable_model_cpu_offload(pipe):
    """Enable CPU offloading for better memory management."""
    if device == "cuda":
        pipe.enable_model_cpu_offload()
        pipe.enable_sequential_cpu_offload()
    return pipe

def enable_attention_slicing(pipe):
    """Enable attention slicing for memory efficiency."""
    if device in ["cuda", "mps"]:
        pipe.enable_attention_slicing(slice_size="auto")
    return pipe

def enable_vae_tiling(pipe):
    """Enable VAE tiling for processing large images."""
    if device == "cuda":
        pipe.enable_vae_tiling()
    return pipe

def optimize_pipeline(pipe):
    """Apply all memory optimization techniques to the pipeline."""
    pipe = enable_model_cpu_offload(pipe)
    pipe = enable_attention_slicing(pipe)
    pipe = enable_vae_tiling(pipe)
    return pipe

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def swap_to_gallery(images):
    return gr.update(value=images, visible=True), gr.update(visible=True), gr.update(visible=False)

def upload_example_to_gallery(images, prompt, style, negative_prompt):
    return gr.update(value=images, visible=True), gr.update(visible=True), gr.update(visible=False)

def remove_back_to_files():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

def remove_tips():
    return gr.update(visible=False)

def get_example():
    case = [
        [
            ['./examples/yann-lecun_resize.jpg'],
            "a man",
            "Snow",
            "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
        ],
        [
            ['./examples/musk_resize.jpeg'],
            "a man",
            "Mars",
            "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
        ],
        [
            ['./examples/sam_resize.png'],
            "a man",
            "Jungle",
            "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, gree",
        ],
        [
            ['./examples/schmidhuber_resize.png'],
            "a man",
            "Neon",
            "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
        ],
        [
            ['./examples/kaifu_resize.png'],
            "a man",
            "Vibrant Color",
            "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
        ],
    ]
    return case

def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def draw_kps(image_pil, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil

def resize_img(input_image, max_side=820, min_side=678, size=None, 
              pad_to_max_side=False, mode=PIL.Image.BILINEAR, base_pixel_number=64):
    
    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image

def apply_style(style_name: str, positive: str, negative: str = "") -> tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive), n + ' ' + negative

#@spaces.GPU
def generate_image(face_image, pose_image, prompt, negative_prompt, style_name, enhance_face_region, num_steps, identitynet_strength_ratio, adapter_strength_ratio, guidance_scale, seed, progress=gr.Progress(track_tqdm=True)):
    try:
        if face_image is None:
            raise gr.Error("Cannot find any input face image! Please upload the face image")
        
        if prompt is None:
            prompt = "a person"
        
        # apply the style template
        prompt, negative_prompt = apply_style(style_name, prompt, negative_prompt)
        
        face_image = load_image(face_image[0])
        face_image = resize_img(face_image)
        face_image_cv2 = convert_from_image_to_cv2(face_image)
        height, width, _ = face_image_cv2.shape
        
        # Extract face features with error handling
        face_info = app.get(face_image_cv2)
        
        if len(face_info) == 0:
            raise gr.Error("Cannot find any face in the image! Please upload another person image")
        
        face_info = face_info[-1]
        face_emb = face_info['embedding']
        face_kps = draw_kps(convert_from_cv2_to_image(face_image_cv2), face_info['kps'])
        
        if pose_image is not None:
            pose_image = load_image(pose_image[0])
            pose_image = resize_img(pose_image)
            pose_image_cv2 = convert_from_image_to_cv2(pose_image)
            
            face_info = app.get(pose_image_cv2)
            
            if len(face_info) == 0:
                raise gr.Error("Cannot find any face in the reference image! Please upload another person image")
            
            face_info = face_info[-1]
            face_kps = draw_kps(pose_image, face_info['kps'])
            
            width, height = face_kps.size
        
        if enhance_face_region:
            control_mask = np.zeros([height, width, 3])
            x1, y1, x2, y2 = face_info['bbox']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            control_mask[y1:y2, x1:x2] = 255
            control_mask = Image.fromarray(control_mask.astype(np.uint8))
        else:
            control_mask = None
        
        generator = torch.Generator(device=device).manual_seed(seed)
        
        logger.info("Starting inference...")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Negative Prompt: {negative_prompt}")
        
        # Memory optimization before generation
        torch.cuda.empty_cache()
        
        pipe.set_ip_adapter_scale(adapter_strength_ratio)
        with torch.inference_mode():
            images = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image_embeds=face_emb,
                image=face_kps,
                control_mask=control_mask,
                controlnet_conditioning_scale=float(identitynet_strength_ratio),
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator,
                cross_attention_kwargs={"scale": 1.0}
            ).images

        # Clean up memory after generation
        torch.cuda.empty_cache()
        
        return images, gr.update(visible=True)
    
    except Exception as e:
        logger.error(f"Error during image generation: {str(e)}")
        raise gr.Error(str(e))

def clear_cuda_cache():
    """Enhanced memory cleanup function."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as e:
        logger.error(f"Error clearing CUDA cache: {str(e)}")

### Description
title = r"""
<h1 align="center">InstantID: Zero-shot Identity-Preserving Generation in Seconds</h1>
"""

description = r"""
<b>Official 🤗 Gradio demo</b> for <a href='https://github.com/InstantID/InstantID' target='_blank'><b>InstantID: Zero-shot Identity-Preserving Generation in Seconds</b></a>.<br>

How to use:<br>
1. Upload a person image. For multiple person images, we will only detect the biggest face. Make sure face is not too small and not significantly blocked or blurred.
2. (Optionally) upload another person image as reference pose. If not uploaded, we will use the first person image to extract landmarks. If you use a cropped face at step1, it is recommeneded to upload it to extract a new pose.
3. Enter a text prompt as done in normal text-to-image models.
4. Click the <b>Submit</b> button to start customizing.
5. Share your customizd photo with your friends, enjoy😊!
"""

article = r"""
---
📝 **Citation**
<br>
If our work is helpful for your research or applications, please cite us via:
```bibtex
@article{wang2024instantid,
  title={InstantID: Zero-shot Identity-Preserving Generation in Seconds},
  author={Wang, Qixun and Bai, Xu and Wang, Haofan and Qin, Zekui and Chen, Anthony},
  journal={arXiv preprint arXiv:2401.07519},
  year={2024}
}
```
📧 **Contact**
<br>
If you have any questions, please feel free to open an issue or directly reach us out at <b>haofanwang.ai@gmail.com</b>.
"""

tips = r"""
### Usage tips of InstantID
1. If you're unsatisfied with the similarity, increase the weight of controlnet_conditioning_scale (IdentityNet) and ip_adapter_scale (Adapter).
2. If the generated image is over-saturated, decrease the ip_adapter_scale. If not work, decrease controlnet_conditioning_scale.
3. If text control is not as expected, decrease ip_adapter_scale.
4. Find a good base model always makes a difference.
"""

css = '''
.gradio-container {width: 85% !important}
'''
with gr.Blocks(css=css) as demo:

    # description
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        with gr.Column():
            
            # upload face image
            face_files = gr.Files(
                        label="Upload a photo of your face",
                        file_types=["image"]
                    )
            uploaded_faces = gr.Gallery(label="Your images", visible=False, columns=1, rows=1, height=512)
            with gr.Column(visible=False) as clear_button_face:
                remove_and_reupload_faces = gr.ClearButton(value="Remove and upload new ones", components=face_files, size="sm")
            
            # optional: upload a reference pose image
            pose_files = gr.Files(
                        label="Upload a reference pose image (optional)",
                        file_types=["image"]
                    )
            uploaded_poses = gr.Gallery(label="Your images", visible=False, columns=1, rows=1, height=512)
            with gr.Column(visible=False) as clear_button_pose:
                remove_and_reupload_poses = gr.ClearButton(value="Remove and upload new ones", components=pose_files, size="sm")
            
            # prompt
            prompt = gr.Textbox(label="Prompt",
                       info="Give simple prompt is enough to achieve good face fedility",
                       placeholder="A photo of a person",
                       value="")
            
            submit = gr.Button("Submit", variant="primary")
            
            style = gr.Dropdown(label="Style template", choices=STYLE_NAMES, value=DEFAULT_STYLE_NAME)
            
            # strength
            identitynet_strength_ratio = gr.Slider(
                label="IdentityNet strength (for fedility)",
                minimum=0,
                maximum=1.5,
                step=0.05,
                value=0.80,
            )
            adapter_strength_ratio = gr.Slider(
                label="Image adapter strength (for detail)",
                minimum=0,
                maximum=1.5,
                step=0.05,
                value=0.80,
            )
            
            with gr.Accordion(open=False, label="Advanced Options"):
                negative_prompt = gr.Textbox(
                    label="Negative Prompt", 
                    placeholder="low quality",
                    value="(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
                )
                num_steps = gr.Slider( 
                    label="Number of sample steps",
                    minimum=20,
                    maximum=100,
                    step=1,
                    value=30,
                )
                guidance_scale = gr.Slider(
                    label="Guidance scale",
                    minimum=0.1,
                    maximum=10.0,
                    step=0.1,
                    value=5,
                )
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=42,
                )
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                enhance_face_region = gr.Checkbox(label="Enhance non-face region", value=True)

        with gr.Column():
            gallery = gr.Gallery(label="Generated Images")
            usage_tips = gr.Markdown(label="Usage tips of InstantID", value=tips ,visible=False)

        face_files.upload(fn=swap_to_gallery, inputs=face_files, outputs=[uploaded_faces, clear_button_face, face_files])
        pose_files.upload(fn=swap_to_gallery, inputs=pose_files, outputs=[uploaded_poses, clear_button_pose, pose_files])

        remove_and_reupload_faces.click(fn=remove_back_to_files, outputs=[uploaded_faces, clear_button_face, face_files])
        remove_and_reupload_poses.click(fn=remove_back_to_files, outputs=[uploaded_poses, clear_button_pose, pose_files])

        submit.click(
            fn=remove_tips,
            outputs=usage_tips,            
        ).then(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            queue=False,
            api_name=False,
        ).then(
            fn=generate_image,
            inputs=[face_files, pose_files, prompt, negative_prompt, style, enhance_face_region, num_steps, identitynet_strength_ratio, adapter_strength_ratio, guidance_scale, seed],
            outputs=[gallery, usage_tips]
        ).then(
            fn=clear_cuda_cache
        )
    
    gr.Examples(
        examples=get_example(),
        inputs=[face_files, prompt, style, negative_prompt],
        run_on_click=True,
        fn=upload_example_to_gallery,
        outputs=[uploaded_faces, clear_button_face, face_files],
    )
    
    gr.Markdown(article)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inbrowser', action='store_true', help='Open in browser')
    parser.add_argument('--server_port', type=int, default=7860, help='Server port')
    parser.add_argument('--share', action='store_true', help='Share the Gradio UI')
    parser.add_argument('--model_path', type=str, default='RunDiffusion/Juggernaut-XL-v8', help='Base model path')
    parser.add_argument('--medvram', action='store_true', help='Medium VRAM settings')
    parser.add_argument('--lowvram', action='store_true', help='Low VRAM settings')

    args = parser.parse_args()
    
    # Adjust settings based on VRAM availability
    if args.lowvram:
        max_side, min_side = 832, 640
        logger.info("Using low VRAM settings")
    elif args.medvram:
        max_side, min_side = 1024, 832
        logger.info("Using medium VRAM settings")
    else:
        max_side, min_side = 1280, 1024
        logger.info("Using standard VRAM settings")

    logger.info(f"Current resolution settings: max_side = {max_side}, min_side = {min_side}")

    # Initialize the pipeline with optimizations
    try:
        pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
            args.model_path,
            controlnet=controlnet,
            torch_dtype=torch_dtype,
            safety_checker=None,
            feature_extractor=None,
        )
        
        # Apply memory optimizations
        pipe = optimize_pipeline(pipe)
        
        if device == 'mps':
            pipe.to("mps", torch_dtype)
        elif device == 'cuda':
            pipe.cuda()
        
        pipe.load_ip_adapter_instantid(face_adapter)
        
        if device in ['mps', 'cuda']:
            pipe.image_proj_model.to(device)
            pipe.unet.to(device)
        
        logger.info("Pipeline initialized successfully")
        
        demo.launch(inbrowser=args.inbrowser, server_port=args.server_port, share=args.share)
        
    except Exception as e:
        logger.error(f"Error initializing pipeline: {str(e)}")
        raise
