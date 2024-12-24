import base64
from functools import partial
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor


class Qwen2VL_Captioner:
    def __init__(
        self,
        model_path: str,
        min_pixels: int = 256,
        max_pixels: int = 1280,
        device: torch.device = torch.device("cuda:0"),
    ):
        self.device = device

        self.load_model_func = partial(
            Qwen2VLForConditionalGeneration.from_pretrained,
            model_path,
            torch_dtype="auto",
            device_map=torch.device("cpu"),
        )
        self.load_proc_func = partial(
            Qwen2VLProcessor.from_pretrained,
            model_path,
            min_pixels=min_pixels * 28 * 28,
            max_pixels=max_pixels * 28 * 28,
        )

    def set_message(self, img: str):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "Caption the image in short, no more than 30 words."},
                ],
            }
        ]

    def caption(self, img: str | np.ndarray) -> str:
        if not hasattr(self, "model"):
            self.model: Qwen2VLForConditionalGeneration = self.load_model_func().eval()
            self.processor: Qwen2VLProcessor = self.load_proc_func()

        model = self.model.to(self.device)

        if isinstance(img, np.ndarray):
            with BytesIO() as buf:
                Image.fromarray(img).save(buf, format="png")
                img = f"data:image;base64,{base64.b64encode(buf.getbuffer()).decode()}"

        messages = self.set_message(img)

        # Preparation for inference
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to(self.device)

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text, *_ = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text


if __name__ == "__main__":
    # model_path = "/mnt/nfs/data/pretrained_models/Qwen2-VL-7B-Instruct-GPTQ-Int8"
    model_path = "/mnt/nfs/data/pretrained_models/Qwen2-VL-7B-Instruct-AWQ"
    captioner = Qwen2VL_Captioner(model_path)
