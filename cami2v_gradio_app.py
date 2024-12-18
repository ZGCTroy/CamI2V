import argparse
import json
import socket

import gradio as gr
import pandas as pd
import torch

from scripts.gradio.cami2v_test import Image2Video, bezier_curve


def load_example():
    key_order = ["input_image", "input_text", "camera_pose_type", "trace_extract_ratio", "steps", "trace_scale_factor", "camera_cfg", "cfg_scale", "seed", "enable_camera_condition"]
    with open(args.example_meta_path, "r") as f:
        data = json.load(f)

    return [[example.get(x, None) for x in key_order] for example in data if example["enable"]]


def load_model_name():
    with open(args.model_meta_path, "r") as f:
        data = json.load(f)

    return list(filter(lambda x: "interp" not in x, data.keys()))


def load_camera_pose_type():
    with open(args.camera_pose_meta_path, "r") as f:
        data = json.load(f)

    return list(data.keys())


def plot_bezier_curve(a: float, b: float):
    results = bezier_curve(torch.linspace(0, 1, 16), a, b).numpy()
    return pd.DataFrame({"x": results[0], "y": results[1]})


def dynamicrafter_demo(result_dir, model_meta_path, camera_pose_meta_path, device="cuda"):
    image2video = Image2Video(result_dir, model_meta_path, camera_pose_meta_path, device=device)

    with gr.Blocks(analytics_enabled=False, css=r"""
        #input_img img {height: 320px !important;}
        #output_vid video {height: 256px !important; margin: auto !important;}
    """) as dynamicrafter_iface:
        gr.Markdown("""
            <div align='center'>
                <h1> CamI2V: Camera-Controlled Image-to-Video Diffusion Model </h1>
            </div>
        """)

        with gr.Row():
            input_image = gr.Image(label="Input Image", elem_id="input_img")

        with gr.Row():
            output_3d = gr.Model3D(label="Camera Trajectory", elem_id="cam_traj", clear_color=[1.0, 1.0, 1.0, 1.0])
            output_video1 = gr.Video(label="New Generated Video", elem_id="output_vid", interactive=False, autoplay=True, loop=True)
            output_video2 = gr.Video(label="Previous Generated Video", elem_id="output_vid", interactive=False, autoplay=True, loop=True)
            state_last_video = gr.State(value=None)

        with gr.Row():
            end_btn = gr.Button("Generate")
            reload_btn = gr.Button("Reload", elem_id="reload_button")

        with gr.Row():
            input_text = gr.Text(label='Prompts')

        with gr.Row():
            with gr.Column():
                model_name = gr.Dropdown(label='Model Name', elem_id="model_name", choices=load_model_name())
                camera_pose_type = gr.Dropdown(label='Camera Pose Type', elem_id="camera_pose_type", choices=load_camera_pose_type())
                trace_extract_ratio = gr.Slider(minimum=0, maximum=1.0, step=0.1, elem_id="trace_extract_ratio", label="Trace Extract Ratio", value=0.5)
                trace_scale_factor = gr.Slider(minimum=0, maximum=20, step=0.1, elem_id="trace_scale_factor", label="Camera Trace Scale Factor", value=1.0)

            with gr.Column():
                enable_camera_condition = gr.Checkbox(label='Enable Camera Condition', elem_id="enable_camera_condition", value=True)
                camera_cfg = gr.Slider(minimum=1.0, maximum=4.0, step=0.1, elem_id="Camera CFG", label="Camera CFG", value=1.0)
                cfg_scale = gr.Slider(minimum=1.0, maximum=15.0, step=0.5, label='CFG Scale', value=6, elem_id="cfg_scale")
                steps = gr.Slider(minimum=1, maximum=60, step=1, elem_id="steps", label="Sampling Steps (DDPM)", value=25)
                seed = gr.Slider(label="Random Seed", minimum=0, maximum=2**31, step=1, value=12333)

            with gr.Column(visible=False):
                use_bezier_curve = gr.Checkbox(label='Use Bézier Curve', elem_id="use_bezier_curve", value=False)
                with gr.Row():
                    bezier_coef_a = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, elem_id="bezier_coef_a", label="Coefficient A", value=0.5)
                    bezier_coef_b = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, elem_id="bezier_coef_a", label="Coefficient B", value=0.5)
                bezier_curve = gr.LinePlot(label="Bézier Curve", elem_id="bezier_curve", value=plot_bezier_curve(0.5, 0.5), x="x", y="y")

        gr_examples = gr.Examples(
            examples=load_example(),
            inputs=[input_image, input_text, camera_pose_type, trace_extract_ratio, steps, trace_scale_factor, camera_cfg, cfg_scale, seed, enable_camera_condition],
            outputs=[output_video1, output_3d],
            fn=image2video.get_image,
            examples_per_page=-1,
        )
        end_btn.click(
            inputs=[model_name, input_image, input_text, camera_pose_type, trace_extract_ratio, steps, trace_scale_factor, camera_cfg, cfg_scale, seed, enable_camera_condition],
            outputs=[output_video1, output_3d],
            fn=image2video.get_image
        )
        reload_btn.click(
            fn=lambda: (gr.Dropdown(choices=load_model_name()), gr.Dropdown(choices=load_camera_pose_type()), gr.Dataset(samples=load_example())),
            outputs=[model_name, camera_pose_type, gr_examples.dataset]
        )

        output_video1.change(fn=lambda s, v: (gr.State(value=v), s), inputs=[state_last_video, output_video1], outputs=[state_last_video, output_video2])
        bezier_coef_a.change(fn=lambda a, b: gr.LinePlot(plot_bezier_curve(a, b)), inputs=[bezier_coef_a, bezier_coef_b], outputs=[bezier_curve])
        bezier_coef_b.change(fn=lambda a, b: gr.LinePlot(plot_bezier_curve(a, b)), inputs=[bezier_coef_a, bezier_coef_b], outputs=[bezier_curve])

    return dynamicrafter_iface


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--result_dir", type=str, default="./gradio_results")
    parser.add_argument("--model_meta_path", type=str, default="./configs/models.json")
    parser.add_argument("--example_meta_path", type=str, default="./configs/examples.json")
    parser.add_argument("--camera_pose_meta_path", type=str, default="./configs/camera_poses.json")
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
    args = parser.parse_args()

    dynamicrafter_iface = dynamicrafter_demo(args.result_dir, args.model_meta_path, args.camera_pose_meta_path, args.device)
    dynamicrafter_iface.queue(max_size=12)
    dynamicrafter_iface.launch(max_threads=10, server_name=get_ip_addr() if args.use_host_ip else None, allowed_paths=["internal/prompts"])
