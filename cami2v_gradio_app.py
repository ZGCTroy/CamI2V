import argparse
import json
import socket

import gradio as gr
import torch

from demo.cami2v_test import Image2Video
from demo.qwen2vl import Qwen2VL_Captioner


def load_example():
    key_order = ["input_image", "input_text"]
    with open(args.example_meta_path, "r") as f:
        data = json.load(f)

    return [[example.get(x, None) for x in key_order] for example in data]


def load_model_name(w: int, h: int):
    with open(args.model_meta_path, "r") as f:
        data = json.load(f)

    return [k for k, v in data.items() if w == v["width"] and h == v["height"]]


def load_camera_pose_type():
    with open(args.camera_pose_meta_path, "r") as f:
        data = json.load(f)

    return list(data.keys())


def get_tab(image2video: Image2Video, resolution: tuple[int, int], captioner: Qwen2VL_Captioner = None):
    def caption(*inputs):
        image2video.offload_cpu()
        return captioner.caption(*inputs)

    def generate(*inputs):
        if args.use_qwen2vl_captioner:
            captioner.offload_cpu()
        return image2video.get_image(*inputs)

    with gr.Tab("x".join(map(str, resolution))) as tab:
        with gr.Row():
            input_image = gr.Image(label="Input Image")
            output_3d = gr.Model3D(label="Camera Trajectory", clear_color=[1.0, 1.0, 1.0, 1.0])

        with gr.Row():
            output_video1 = gr.Video(label="New Generated Video", elem_id="output_vid", interactive=False, autoplay=True, loop=True)
            output_video2 = gr.Video(label="Previous Generated Video", elem_id="output_vid", interactive=False, autoplay=True, loop=True)

        with gr.Row(equal_height=True):
            input_text = gr.Textbox(label='Prompts', scale=4)

            if args.use_qwen2vl_captioner:
                caption_btn = gr.Button("Caption")
                caption_btn.click(fn=caption, inputs=[input_image], outputs=[input_text])

        with gr.Row():
            negative_prompt = gr.Textbox(label='Negative Prompts', value="Fast movement, jittery motion, abrupt transitions, distorted body, missing limbs, unnatural posture, blurry, cropped, extra limbs, bad anatomy, deformed, glitchy motion, artifacts.")

        with gr.Row():
            with gr.Column():
                model_name = gr.Dropdown(label='Model Name', choices=load_model_name(*resolution))
                camera_pose_type = gr.Dropdown(label='Camera Pose Type', choices=load_camera_pose_type())
                trace_extract_ratio = gr.Slider(label="Trace Extract Ratio", minimum=0, maximum=1.0, step=0.1, value=0.1)
                trace_scale_factor = gr.Slider(label="Camera Trace Scale Factor", minimum=0, maximum=5, step=0.1, value=1.0)

            with gr.Column():
                enable_camera_condition = gr.Checkbox(label='Enable Camera Condition', value=True)
                camera_cfg = gr.Slider(label="Camera CFG", minimum=1.0, maximum=4.0, step=0.1, value=1.0, visible=False)
                cfg_scale = gr.Slider(label='CFG Scale', minimum=1.0, maximum=15.0, step=0.5, value=5.5)
                frame_stride = gr.Slider(label='Frame Stride', minimum=1, maximum=10, step=1, value=2)
                steps = gr.Slider(label="Sampling Steps (DDPM)", minimum=1, maximum=250, step=1, value=25)
                seed = gr.Slider(label="Random Seed", minimum=0, maximum=2**31, step=1, value=12333)

        with gr.Row():
            generate_btn = gr.Button("Generate")
            generate_btn.click(fn=lambda x: x, inputs=[output_video1], outputs=[output_video2])
            generate_btn.click(
                fn=generate,
                inputs=[model_name, input_image, input_text, negative_prompt, camera_pose_type, trace_extract_ratio, frame_stride, steps, trace_scale_factor, camera_cfg, cfg_scale, seed, enable_camera_condition],
                outputs=[output_video1, output_3d],
            )

            reload_btn = gr.Button("Reload")
            reload_btn.click(fn=lambda: gr.Dropdown(choices=load_model_name(*resolution)), outputs=[model_name])
            reload_btn.click(fn=lambda: gr.Dropdown(choices=load_camera_pose_type()), outputs=[camera_pose_type])

        gr_examples = gr.Examples(
            examples=load_example(),
            inputs=[input_image, input_text],
            examples_per_page=-1,
        )

        reload_btn.click(fn=lambda: gr.Dataset(samples=load_example()), outputs=[gr_examples.dataset])

    return tab


def get_demo(args):
    image2video = Image2Video(args.result_dir, args.model_meta_path, args.camera_pose_meta_path, device=args.device)

    if args.use_qwen2vl_captioner:
        captioner = Qwen2VL_Captioner(model_path="pretrained_models/Qwen2-VL-7B-Instruct-AWQ", device=torch.device(args.device))
    else:
        captioner = None

    with gr.Blocks(css="""
        .block {height: 100% !important; width: 100% !important;}
        #output_vid video {width: auto !important; margin: auto !important;}
    """) as demo:
        gr.Markdown("""
            <div align='center'>
                <h1> CamI2V: Camera-Controlled Image-to-Video Diffusion Model </h1>
            </div>
        """)
        get_tab(image2video, resolution=(512, 320), captioner=captioner)
        get_tab(image2video, resolution=(256, 256), captioner=captioner)

    return demo


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--result_dir", type=str, default="./demo/results")
    parser.add_argument("--model_meta_path", type=str, default="./demo/models.json")
    parser.add_argument("--example_meta_path", type=str, default="./demo/examples.json")
    parser.add_argument("--camera_pose_meta_path", type=str, default="./demo/camera_poses.json")
    parser.add_argument("--use_qwen2vl_captioner", action="store_true")
    parser.add_argument("--use_host_ip", action="store_true")

    return parser


def get_ip_addr():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 53))
        return s.getsockname()[0]
    except:
        return None


if __name__ == "__main__":
    parser = get_parser()
    args, _ = parser.parse_known_args()

    get_demo(args).launch(server_name=get_ip_addr() if args.use_host_ip else None, allowed_paths=["demo", "internal/prompts"])
