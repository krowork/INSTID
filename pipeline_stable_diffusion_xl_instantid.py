# Copyright 2024 The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import math
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from transformers import CLIPTokenizer

from diffusers.image_processor import PipelineImageInput
from diffusers.models import ControlNetModel
from diffusers.utils import (
    deprecate,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
from diffusers import StableDiffusionXLControlNetPipeline
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.utils.import_utils import is_xformers_available

from ip_adapter.resampler_flexible import FlexibleResampler as Resampler
from ip_adapter.attention_processor import AttnProcessor, IPAttnProcessor
from ip_adapter.attention_processor import region_control

import os
from PIL import Image, ImageDraw

from diffusers.models.attention import BasicTransformerBlock

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> # !pip install opencv-python transformers accelerate insightface
        >>> import diffusers
        >>> from diffusers.utils import load_image
        >>> from diffusers.models import ControlNetModel

        >>> import cv2
        >>> import torch
        >>> import numpy as np
        >>> from PIL import Image
        
        >>> from insightface.app import FaceAnalysis
        >>> from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps

        >>> # download 'antelopev2' under ./models
        >>> app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        >>> app.prepare(ctx_id=0, det_size=(640, 640))
        
        >>> # download models under ./checkpoints
        >>> face_adapter = f'./checkpoints/ip-adapter.bin'
        >>> controlnet_path = f'./checkpoints/ControlNetModel'
        
        >>> # load IdentityNet
        >>> controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
        
        >>> pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, torch_dtype=torch.float16
        ... )
        >>> pipe.cuda()
        
        >>> # load adapter
        >>> pipe.load_ip_adapter_instantid(face_adapter)

        >>> prompt = "analog film photo of a man. faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage, masterpiece, best quality"
        >>> negative_prompt = "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured (lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch,deformed, mutated, cross-eyed, ugly, disfigured"

        >>> # load an image
        >>> image = load_image("your-example.jpg")
        
        >>> face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))[-1]
        >>> face_emb = face_info['embedding']
        >>> face_kps = draw_kps(face_image, face_info['kps'])
        
        >>> pipe.set_ip_adapter_scale(0.8)

        >>> # generate image
        >>> image = pipe(
        ...     prompt, image_embeds=face_emb, image=face_kps, controlnet_conditioning_scale=0.8
        ... ).images[0]
        ```
"""


from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline
class LongPromptWeight(object):
    
    """
    Copied from https://github.com/huggingface/diffusers/blob/main/examples/community/lpw_stable_diffusion_xl.py
    """
    
    def __init__(self) -> None:
        pass

    def parse_prompt_attention(self, text):
        """
        Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
        Accepted tokens are:
        (abc) - increases attention to abc by a multiplier of 1.1
        (abc:3.12) - increases attention to abc by a multiplier of 3.12
        [abc] - decreases attention to abc by a multiplier of 1.1
        \( - literal character '('
        \[ - literal character '['
        \) - literal character ')'
        \] - literal character ']'
        \\ - literal character '\'
        anything else - just text

        >>> parse_prompt_attention('normal text')
        [['normal text', 1.0]]
        >>> parse_prompt_attention('an (important) word')
        [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
        >>> parse_prompt_attention('(unbalanced')
        [['unbalanced', 1.1]]
        >>> parse_prompt_attention('\(literal\]')
        [['(literal]', 1.0]]
        >>> parse_prompt_attention('(unnecessary)(parens)')
        [['unnecessaryparens', 1.1]]
        >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
        [['a ', 1.0],
        ['house', 1.5730000000000004],
        [' ', 1.1],
        ['on', 1.0],
        [' a ', 1.1],
        ['hill', 0.55],
        [', sun, ', 1.1],
        ['sky', 1.4641000000000006],
        ['.', 1.1]]
        """
        import re

        re_attention = re.compile(
            r"""
                \\\(|\\\)|\\\[|\\]|\\\\|\\|\(|\[|:([+-]?[.\d]+)\)|
                \)|]|[^\\()\[\]:]+|:
            """,
            re.X,
        )

        re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)

        res = []
        round_brackets = []
        square_brackets = []

        round_bracket_multiplier = 1.1
        square_bracket_multiplier = 1 / 1.1

        def multiply_range(start_position, multiplier):
            for p in range(start_position, len(res)):
                res[p][1] *= multiplier

        for m in re_attention.finditer(text):
            text = m.group(0)
            weight = m.group(1)

            if text.startswith("\\"):
                res.append([text[1:], 1.0])
            elif text == "(":
                round_brackets.append(len(res))
            elif text == "[":
                square_brackets.append(len(res))
            elif weight is not None and len(round_brackets) > 0:
                multiply_range(round_brackets.pop(), float(weight))
            elif text == ")" and len(round_brackets) > 0:
                multiply_range(round_brackets.pop(), round_bracket_multiplier)
            elif text == "]" and len(square_brackets) > 0:
                multiply_range(square_brackets.pop(), square_bracket_multiplier)
            else:
                parts = re.split(re_break, text)
                for i, part in enumerate(parts):
                    if i > 0:
                        res.append(["BREAK", -1])
                    res.append([part, 1.0])

        for pos in round_brackets:
            multiply_range(pos, round_bracket_multiplier)

        for pos in square_brackets:
            multiply_range(pos, square_bracket_multiplier)

        if len(res) == 0:
            res = [["", 1.0]]

        # merge runs of identical weights
        i = 0
        while i + 1 < len(res):
            if res[i][1] == res[i + 1][1]:
                res[i][0] += res[i + 1][0]
                res.pop(i + 1)
            else:
                i += 1

        return res

    def get_prompts_tokens_with_weights(self, clip_tokenizer: CLIPTokenizer, prompt: str):
        """
        Get prompt token ids and weights, this function works for both prompt and negative prompt

        Args:
            pipe (CLIPTokenizer)
                A CLIPTokenizer
            prompt (str)
                A prompt string with weights

        Returns:
            text_tokens (list)
                A list contains token ids
            text_weight (list)
                A list contains the correspodent weight of token ids

        Example:
            import torch
            from transformers import CLIPTokenizer

            clip_tokenizer = CLIPTokenizer.from_pretrained(
                "stablediffusionapi/deliberate-v2"
                , subfolder = "tokenizer"
                , dtype = torch.float16
            )

            token_id_list, token_weight_list = get_prompts_tokens_with_weights(
                clip_tokenizer = clip_tokenizer
                ,prompt = "a (red:1.5) cat"*70
            )
        """
        texts_and_weights = self.parse_prompt_attention(prompt)
        text_tokens, text_weights = [], []
        for word, weight in texts_and_weights:
            # tokenize and discard the starting and the ending token
            token = clip_tokenizer(word, truncation=False).input_ids[1:-1]  # so that tokenize whatever length prompt
            # the returned token is a 1d list: [320, 1125, 539, 320]

            # merge the new tokens to the all tokens holder: text_tokens
            text_tokens = [*text_tokens, *token]

            # each token chunk will come with one weight, like ['red cat', 2.0]
            # need to expand weight for each token.
            chunk_weights = [weight] * len(token)

            # append the weight back to the weight holder: text_weights
            text_weights = [*text_weights, *chunk_weights]
        return text_tokens, text_weights

    def group_tokens_and_weights(self, token_ids: list, weights: list, pad_last_block=False):
        """
        Produce tokens and weights in groups and pad the missing tokens

        Args:
            token_ids (list)
                The token ids from tokenizer
            weights (list)
                The weights list from function get_prompts_tokens_with_weights
            pad_last_block (bool)
                Control if fill the last token list to 75 tokens with eos
        Returns:
            new_token_ids (2d list)
            new_weights (2d list)

        Example:
            token_groups,weight_groups = group_tokens_and_weights(
                token_ids = token_id_list
                , weights = token_weight_list
            )
        """
        bos, eos = 49406, 49407

        # this will be a 2d list
        new_token_ids = []
        new_weights = []
        while len(token_ids) >= 75:
            # get the first 75 tokens
            head_75_tokens = [token_ids.pop(0) for _ in range(75)]
            head_75_weights = [weights.pop(0) for _ in range(75)]

            # extract token ids and weights
            temp_77_token_ids = [bos] + head_75_tokens + [eos]
            temp_77_weights = [1.0] + head_75_weights + [1.0]

            # add 77 token and weights chunk to the holder list
            new_token_ids.append(temp_77_token_ids)
            new_weights.append(temp_77_weights)

        # padding the left
        if len(token_ids) >= 0:
            padding_len = 75 - len(token_ids) if pad_last_block else 0

            temp_77_token_ids = [bos] + token_ids + [eos] * padding_len + [eos]
            new_token_ids.append(temp_77_token_ids)

            temp_77_weights = [1.0] + weights + [1.0] * padding_len + [1.0]
            new_weights.append(temp_77_weights)

        return new_token_ids, new_weights

    def get_weighted_text_embeddings_sdxl(
        self,
        pipe: StableDiffusionXLPipeline,
        prompt: str = "",
        prompt_2: str = None,
        neg_prompt: str = "",
        neg_prompt_2: str = None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        extra_emb=None,
        extra_emb_alpha=0.6,
    ):
        """
        This function can process long prompt with weights, no length limitation
        for Stable Diffusion XL

        Args:
            pipe (StableDiffusionPipeline)
            prompt (str)
            prompt_2 (str)
            neg_prompt (str)
            neg_prompt_2 (str)
        Returns:
            prompt_embeds (torch.Tensor)
            neg_prompt_embeds (torch.Tensor)
        """
        # 
        if prompt_embeds is not None and \
            negative_prompt_embeds is not None and \
            pooled_prompt_embeds is not None and \
            negative_pooled_prompt_embeds is not None:
            return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

        if prompt_2:
            prompt = f"{prompt} {prompt_2}"

        if neg_prompt_2:
            neg_prompt = f"{neg_prompt} {neg_prompt_2}"

        eos = pipe.tokenizer.eos_token_id

        # tokenizer 1
        prompt_tokens, prompt_weights = self.get_prompts_tokens_with_weights(pipe.tokenizer, prompt)
        neg_prompt_tokens, neg_prompt_weights = self.get_prompts_tokens_with_weights(pipe.tokenizer, neg_prompt)

        # tokenizer 2
        # prompt_tokens_2, prompt_weights_2 = self.get_prompts_tokens_with_weights(pipe.tokenizer_2, prompt)
        # neg_prompt_tokens_2, neg_prompt_weights_2 = self.get_prompts_tokens_with_weights(pipe.tokenizer_2, neg_prompt)
        # tokenizer 2 遇到 !! !!!! 等多感叹号和tokenizer 1的效果不一致
        prompt_tokens_2, prompt_weights_2 = self.get_prompts_tokens_with_weights(pipe.tokenizer, prompt)
        neg_prompt_tokens_2, neg_prompt_weights_2 = self.get_prompts_tokens_with_weights(pipe.tokenizer, neg_prompt)

        # padding the shorter one for prompt set 1
        prompt_token_len = len(prompt_tokens)
        neg_prompt_token_len = len(neg_prompt_tokens)

        if prompt_token_len > neg_prompt_token_len:
            # padding the neg_prompt with eos token
            neg_prompt_tokens = neg_prompt_tokens + [eos] * abs(prompt_token_len - neg_prompt_token_len)
            neg_prompt_weights = neg_prompt_weights + [1.0] * abs(prompt_token_len - neg_prompt_token_len)
        else:
            # padding the prompt
            prompt_tokens = prompt_tokens + [eos] * abs(prompt_token_len - neg_prompt_token_len)
            prompt_weights = prompt_weights + [1.0] * abs(prompt_token_len - neg_prompt_token_len)

        # padding the shorter one for token set 2
        prompt_token_len_2 = len(prompt_tokens_2)
        neg_prompt_token_len_2 = len(neg_prompt_tokens_2)

        if prompt_token_len_2 > neg_prompt_token_len_2:
            # padding the neg_prompt with eos token
            neg_prompt_tokens_2 = neg_prompt_tokens_2 + [eos] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
            neg_prompt_weights_2 = neg_prompt_weights_2 + [1.0] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        else:
            # padding the prompt
            prompt_tokens_2 = prompt_tokens_2 + [eos] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
            prompt_weights_2 = prompt_weights + [1.0] * abs(prompt_token_len_2 - neg_prompt_token_len_2)

        embeds = []
        neg_embeds = []

        prompt_token_groups, prompt_weight_groups = self.group_tokens_and_weights(prompt_tokens.copy(), prompt_weights.copy())

        neg_prompt_token_groups, neg_prompt_weight_groups = self.group_tokens_and_weights(
            neg_prompt_tokens.copy(), neg_prompt_weights.copy()
        )

        prompt_token_groups_2, prompt_weight_groups_2 = self.group_tokens_and_weights(
            prompt_tokens_2.copy(), prompt_weights_2.copy()
        )

        neg_prompt_token_groups_2, neg_prompt_weight_groups_2 = self.group_tokens_and_weights(
            neg_prompt_tokens_2.copy(), neg_prompt_weights_2.copy()
        )

        # get prompt embeddings one by one is not working.
        for i in range(len(prompt_token_groups)):
            # get positive prompt embeddings with weights
            token_tensor = torch.tensor([prompt_token_groups[i]], dtype=torch.long, device=pipe.device)
            weight_tensor = torch.tensor(prompt_weight_groups[i], dtype=torch.float16, device=pipe.device)

            token_tensor_2 = torch.tensor([prompt_token_groups_2[i]], dtype=torch.long, device=pipe.device)

            # use first text encoder
            prompt_embeds_1 = pipe.text_encoder(token_tensor.to(pipe.device), output_hidden_states=True)
            prompt_embeds_1_hidden_states = prompt_embeds_1.hidden_states[-2]

            # use second text encoder
            prompt_embeds_2 = pipe.text_encoder_2(token_tensor_2.to(pipe.device), output_hidden_states=True)
            prompt_embeds_2_hidden_states = prompt_embeds_2.hidden_states[-2]
            pooled_prompt_embeds = prompt_embeds_2[0]

            prompt_embeds_list = [prompt_embeds_1_hidden_states, prompt_embeds_2_hidden_states]
            token_embedding = torch.concat(prompt_embeds_list, dim=-1).squeeze(0)

            for j in range(len(weight_tensor)):
                if weight_tensor[j] != 1.0:
                    token_embedding[j] = (
                        token_embedding[-1] + (token_embedding[j] - token_embedding[-1]) * weight_tensor[j]
                    )

            token_embedding = token_embedding.unsqueeze(0)
            embeds.append(token_embedding)

            # get negative prompt embeddings with weights
            neg_token_tensor = torch.tensor([neg_prompt_token_groups[i]], dtype=torch.long, device=pipe.device)
            neg_token_tensor_2 = torch.tensor([neg_prompt_token_groups_2[i]], dtype=torch.long, device=pipe.device)
            neg_weight_tensor = torch.tensor(neg_prompt_weight_groups[i], dtype=torch.float16, device=pipe.device)

            # use first text encoder
            neg_prompt_embeds_1 = pipe.text_encoder(neg_token_tensor.to(pipe.device), output_hidden_states=True)
            neg_prompt_embeds_1_hidden_states = neg_prompt_embeds_1.hidden_states[-2]

            # use second text encoder
            neg_prompt_embeds_2 = pipe.text_encoder_2(neg_token_tensor_2.to(pipe.device), output_hidden_states=True)
            neg_prompt_embeds_2_hidden_states = neg_prompt_embeds_2.hidden_states[-2]
            negative_pooled_prompt_embeds = neg_prompt_embeds_2[0]

            neg_prompt_embeds_list = [neg_prompt_embeds_1_hidden_states, neg_prompt_embeds_2_hidden_states]
            neg_token_embedding = torch.concat(neg_prompt_embeds_list, dim=-1).squeeze(0)

            for z in range(len(neg_weight_tensor)):
                if neg_weight_tensor[z] != 1.0:
                    neg_token_embedding[z] = (
                        neg_token_embedding[-1] + (neg_token_embedding[z] - neg_token_embedding[-1]) * neg_weight_tensor[z]
                    )

            neg_token_embedding = neg_token_embedding.unsqueeze(0)
            neg_embeds.append(neg_token_embedding)

        prompt_embeds = torch.cat(embeds, dim=1)
        negative_prompt_embeds = torch.cat(neg_embeds, dim=1)

        if extra_emb is not None:
            extra_emb = extra_emb.to(prompt_embeds.device, dtype=prompt_embeds.dtype) * extra_emb_alpha
            prompt_embeds = torch.cat([prompt_embeds, extra_emb], 1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, torch.zeros_like(extra_emb)], 1)
            print(f'fix prompt_embeds, extra_emb_alpha={extra_emb_alpha}')

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def get_prompt_embeds(self, *args, **kwargs):
        prompt_embeds, negative_prompt_embeds, _, _ = self.get_weighted_text_embeddings_sdxl(*args, **kwargs)
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        return prompt_embeds

    
class StableDiffusionXLInstantIDPipeline(StableDiffusionXLControlNetPipeline):
    
    def cuda(self, dtype=torch.float16, use_xformers=False):
        """Move the pipeline to CUDA with memory optimizations."""
        try:
            super().to("cuda", dtype)
            if use_xformers and is_xformers_available():
                self.enable_xformers_memory_efficient_attention()
                logger.info("Enabled xformers memory efficient attention")
            return self
        except Exception as e:
            logger.error(f"Error moving pipeline to CUDA: {str(e)}")
            raise

    def load_ip_adapter_instantid(self, model_ckpt, image_emb_dim=512, num_tokens=16, scale=0.5):
        """Load IP-Adapter with error handling and logging."""
        try:
            logger.info("Loading IP-Adapter InstantID...")
            self.set_image_proj_model(model_ckpt, image_emb_dim, num_tokens)
            self.set_ip_adapter(model_ckpt, num_tokens, scale)
            logger.info("Successfully loaded IP-Adapter InstantID")
        except Exception as e:
            logger.error(f"Error loading IP-Adapter InstantID: {str(e)}")
            raise RuntimeError("Failed to load IP-Adapter InstantID")

    def set_image_proj_model(self, model_ckpt, image_emb_dim=512, num_tokens=16):
        """Set up image projection model with error handling."""
        try:
            self.image_proj_model = Resampler(
                dim=self.unet.config.cross_attention_dim,
                depth=4,
                dim_head=64,
                heads=16,
                num_queries=num_tokens,
                embedding_dim=image_emb_dim,
                output_dim=self.unet.config.cross_attention_dim,
                ff_mult=4
            )
            
            state_dict = torch.load(model_ckpt, map_location="cpu")
            image_proj_dict = {}
            
            # Extraer solo los parámetros del image_proj
            for key in state_dict.keys():
                if key.startswith("image_proj."):
                    new_key = key.replace("image_proj.", "")
                    image_proj_dict[new_key] = state_dict[key]
            
            # Debug: mostrar qué parámetros tenemos
            logger.info(f"Parámetros encontrados en checkpoint: {list(image_proj_dict.keys())[:10]}...")
            logger.info(f"Parámetros esperados en modelo: {list(self.image_proj_model.state_dict().keys())[:10]}...")
            
            # Intentar carga con el método flexible
            if hasattr(self.image_proj_model, 'load_state_dict_flexible'):
                missing_keys, unexpected_keys = self.image_proj_model.load_state_dict_flexible(image_proj_dict, strict=False)
            else:
                missing_keys, unexpected_keys = self.image_proj_model.load_state_dict(image_proj_dict, strict=False)
            
            if missing_keys:
                logger.warning(f"Parámetros faltantes en checkpoint: {missing_keys[:5]}...")
            if unexpected_keys:
                logger.warning(f"Parámetros extra en checkpoint: {unexpected_keys[:5]}...")
            
            logger.info("Successfully loaded image projection model (with potential missing parameters)")
            
        except Exception as e:
            logger.error(f"Error setting up image projection model: {str(e)}")
            
            # Información adicional para debug
            try:
                state_dict = torch.load(model_ckpt, map_location="cpu")
                image_proj_keys = [k for k in state_dict.keys() if k.startswith("image_proj.")]
                logger.error(f"Checkpoint contiene {len(image_proj_keys)} parámetros image_proj")
                logger.error(f"Primeros parámetros: {image_proj_keys[:5]}")
                
                model_keys = list(self.image_proj_model.state_dict().keys())
                logger.error(f"Modelo espera {len(model_keys)} parámetros")
                logger.error(f"Primeros parámetros del modelo: {model_keys[:5]}")
            except:
                pass
                
            raise RuntimeError("Failed to set up image projection model")

    def set_ip_adapter(self, model_ckpt, num_tokens=16, scale=0.5):
        """Set up IP-Adapter with error handling and memory optimization."""
        try:
            state_dict = torch.load(model_ckpt, map_location="cpu")
            ip_layers = torch.nn.ModuleList([])
            
            for i in range(len(self.unet.up_blocks)):
                layer = BasicTransformerBlock(
                    dim=self.unet.up_blocks[i].resnets[1].out_channels,
                    num_attention_heads=8,
                    attention_head_dim=64,
                    cross_attention_dim=self.unet.config.cross_attention_dim,
                    ff_inner_dim=None,
                    ff_bias=True,
                    post_attention_norm=True,
                )
                ip_layers.append(layer)
            
            self.ip_layers = ip_layers
            ip_dict = {}
            
            for key in state_dict.keys():
                if key.startswith("ip_adapter."):
                    ip_dict[key.replace("ip_adapter.", "")] = state_dict[key]
            
            self.ip_layers.load_state_dict(ip_dict)
            self.gradient_checkpointing = False
            self.ip_scale = scale
            logger.info(f"Successfully set up IP-Adapter with scale {scale}")
        except Exception as e:
            logger.error(f"Error setting up IP-Adapter: {str(e)}")
            raise RuntimeError("Failed to set up IP-Adapter")

    def set_ip_adapter_scale(self, scale):
        """Set IP-Adapter scale with validation."""
        try:
            if not isinstance(scale, (int, float)) or scale < 0:
                raise ValueError("Scale must be a non-negative number")
            self.ip_scale = scale
            logger.info(f"IP-Adapter scale set to {scale}")
        except Exception as e:
            logger.error(f"Error setting IP-Adapter scale: {str(e)}")
            raise

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        original_size: Tuple[int, int] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Tuple[int, int] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        control_mask = None,
        **kwargs,
    ):
        """Enhanced pipeline call with better error handling and memory management.

        Examples:
        """
        try:
            # Memory optimization before generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Input validation
            if prompt is None and prompt_embeds is None:
                raise ValueError("Either `prompt` or `prompt_embeds` must be provided")
            if image is None:
                raise ValueError("Image input cannot be None")

            # Process controlnet
            if isinstance(self.controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
                controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(self.controlnet.nets)

            # Set up cross attention kwargs
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}

            # Process the inputs
            device = self._execution_device
            do_classifier_free_guidance = guidance_scale > 1.0

            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                prompt_2=prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                lora_scale=cross_attention_kwargs.get("scale", None),
                clip_skip=clip_skip,
            )

            # Process image input
            if isinstance(self.controlnet, MultiControlNetModel) and isinstance(image, list):
                images = []
                for image_ in image:
                    images.append(self.prepare_image(
                        image=image_,
                        width=width,
                        height=height,
                        batch_size=prompt_embeds.shape[0] * num_images_per_prompt,
                        num_images_per_prompt=num_images_per_prompt,
                        device=device,
                        dtype=self.controlnet.dtype,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        guess_mode=guess_mode,
                    ))
                image = images
            else:
                image = self.prepare_image(
                    image=image,
                    width=width,
                    height=height,
                    batch_size=prompt_embeds.shape[0] * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=self.controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )

            # Process control mask if provided
            if control_mask is not None:
                control_mask = self.prepare_control_mask(
                    control_mask=control_mask,
                    width=width,
                    height=height,
                    batch_size=prompt_embeds.shape[0] * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=self.controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )

            # Prepare timesteps
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

            # Prepare latents
            num_channels_latents = self.unet.config.in_channels
            latents = self.prepare_latents(
                batch_size=prompt_embeds.shape[0],
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                dtype=prompt_embeds.dtype,
                device=device,
                generator=generator,
                latents=latents,
            )

            # Prepare extra step kwargs
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            # Process image embeddings
            if image_embeds is not None:
                image_embeds = self.prepare_image_embeds(image_embeds, device, do_classifier_free_guidance)

            # Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    # Expand latents for classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                    # Scale latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # ControlNet(s) inference
                    control_model_input = latent_model_input
                    controlnet_prompt_embeds = prompt_embeds
                    
                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        control_model_input,
                        t,
                        encoder_hidden_states=controlnet_prompt_embeds,
                        controlnet_cond=image,
                        conditioning_scale=controlnet_conditioning_scale,
                        guess_mode=guess_mode,
                        return_dict=False,
                    )

                    # Predict noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        return_dict=False,
                    )[0]

                    # Perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # Compute previous noisy sample
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                    # Update progress
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback_on_step_end is not None:
                            callback_kwargs = {}
                            for k in callback_on_step_end_tensor_inputs:
                                callback_kwargs[k] = locals()[k]
                            callback_on_step_end(i, t, callback_kwargs)

            # Post-processing
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if not output_type == "latent":
                image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=[True] * image.shape[0])

            # Offload all models
            self.maybe_free_model_hooks()

            if not return_dict:
                return (image, has_nsfw_concept)

            return StableDiffusionXLPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

        except Exception as e:
            logger.error(f"Error during pipeline execution: {str(e)}")
            raise RuntimeError(f"Pipeline execution failed: {str(e)}")

    def prepare_image_embeds(self, image_embeds, device, do_classifier_free_guidance):
        """Prepare image embeddings with validation."""
        try:
            if not isinstance(image_embeds, torch.Tensor):
                raise ValueError("Image embeddings must be a torch.Tensor")
            
            image_embeds = image_embeds.to(device=device, dtype=self.text_encoder.dtype)
            if do_classifier_free_guidance:
                image_embeds = torch.cat([image_embeds] * 2)
            
            return image_embeds
        except Exception as e:
            logger.error(f"Error preparing image embeddings: {str(e)}")
            raise

    def prepare_control_mask(self, control_mask, width, height, batch_size, num_images_per_prompt, device, dtype, do_classifier_free_guidance, guess_mode):
        """Prepare control mask with validation and error handling."""
        try:
            if not isinstance(control_mask, (PIL.Image.Image, np.ndarray, torch.Tensor)):
                raise ValueError("Control mask must be a PIL Image, numpy array, or torch.Tensor")
            
            if isinstance(control_mask, PIL.Image.Image):
                control_mask = np.array(control_mask)
            
            if isinstance(control_mask, np.ndarray):
                control_mask = torch.from_numpy(control_mask)
            
            control_mask = control_mask.to(device=device, dtype=dtype)
            control_mask = control_mask.repeat(batch_size * num_images_per_prompt, 1, 1, 1)
            
            if do_classifier_free_guidance and not guess_mode:
                control_mask = torch.cat([control_mask] * 2)
            
            return control_mask
        except Exception as e:
            logger.error(f"Error preparing control mask: {str(e)}")
            raise

def draw_kps(image: Image.Image, kps: np.ndarray, color: tuple = (255, 0, 0), size: int = 4) -> Image.Image:
    """Dibuja puntos clave (keypoints) en una imagen.
    
    Args:
        image: Imagen PIL sobre la que dibujar
        kps: Array numpy con los keypoints en formato [[x1,y1], [x2,y2], ...]
        color: Color de los puntos (R,G,B)
        size: Tamaño de los puntos
        
    Returns:
        Imagen PIL con los keypoints dibujados
    """
    # Crear una copia de la imagen para no modificar la original
    result = image.copy()
    draw = ImageDraw.Draw(result)
    
    # Dibujar cada keypoint
    for x, y in kps:
        x1, y1 = int(x - size), int(y - size)
        x2, y2 = int(x + size), int(y + size)
        draw.ellipse([x1, y1, x2, y2], fill=color)
        
    return result