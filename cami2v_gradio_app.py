import argparse
import json
import socket

import gradio as gr
import pandas as pd
import torch

from scripts.gradio.cami2v_test import Image2Video, bezier_curve


def load_example():
    key_order = ["input_image", "input_text", "camera_pose_type", "trace_extract_ratio", "frame_stride", "steps", "trace_scale_factor", "camera_cfg", "cfg_scale", "eta", "seed",
                 "enable_camera_condition", "loop", "use_bezier_curve", "bezier_coef_a", "bezier_coef_b"]
    with open("examples.json", "r") as f:
        data = json.load(f)

    return [[example.get(x, None) for x in key_order] for example in data if example["enable"]]


def load_model_name():
    with open("models.json", "r") as f:
        data = json.load(f)

    return list(filter(lambda x: "interp" not in x, data.keys()))


def load_camera_pose_type():
    with open("prompts/camera_pose_files/meta_data.json", "r") as f:
        data = json.load(f)

    return list(data.keys())


def plot_bezier_curve(a: float, b: float):
    results = bezier_curve(torch.linspace(0, 1, 16), a, b).numpy()
    return pd.DataFrame({"x": results[0], "y": results[1]})


max_seed = 2 ** 31


def dynamicrafter_demo(result_dir='./tmp', device="cuda"):
    image2video = Image2Video(result_dir, device=device)

    with gr.Blocks(analytics_enabled=False, css="""
        #input_img, #cam_traj {max-width: 512px !important; max-height: 320px !important;}
        #output_vid {width: auto !important; height: auto !important;}
    """) as dynamicrafter_iface:
        gr.Markdown("""
            <div align='center'>
                <h1> CamI2V: Camera-Controlled Image-to-Video Diffusion Model </h1>
            </div>
        """)

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image", elem_id="input_img")
            with gr.Column():
                output_3d = gr.Model3D(label="Camera Trajectory", elem_id="cam_traj", clear_color=[1.0, 1.0, 1.0, 1.0])

        with gr.Row():
            with gr.Column():
                output_video1 = gr.Video(label="New Generated Video", elem_id="output_vid", interactive=False, autoplay=True, loop=True)
            with gr.Column():
                output_video2 = gr.Video(label="Previous Generated Video", elem_id="output_vid", interactive=False, autoplay=True, loop=True)
                state_last_video = gr.State(value=None)

        with gr.Row():
            end_btn = gr.Button("Generate")
            reload_btn = gr.Button("Reload Examples & Camera Pose Types", elem_id="reload_button")

        with gr.Row():
            input_text = gr.Text(label='Prompts')

        with gr.Row():
            with gr.Column():
                model_name = gr.Dropdown(label='Model Name', elem_id="model_name", choices=load_model_name(), value="512_cami2v", allow_custom_value=True)
                camera_pose_type = gr.Dropdown(label='Camera Pose Type', elem_id="camera_pose_type", choices=load_camera_pose_type(), value="zoom in", allow_custom_value=True)
                cond_frame_index = gr.Slider(minimum=0, maximum=15, step=1, elem_id="cond_frame_index", label="Condition Frame Index", value=0)
                trace_extract_ratio = gr.Slider(minimum=0, maximum=1.0, step=0.1, elem_id="trace_extract_ratio", label="Trace Extract Ratio", value=1.0)
                trace_scale_factor = gr.Slider(minimum=0, maximum=20, step=0.1, elem_id="trace_scale_factor", label="Camera Trace Scale Factor", value=1.0)

            with gr.Column():
                enable_camera_condition = gr.Checkbox(label='Enable Camera Condition', elem_id="enable_camera_condition", value=True)
                camera_cfg = gr.Slider(minimum=1.0, maximum=4.0, step=0.1, elem_id="Camera CFG", label="Camera CFG", value=1.0)
                cfg_scale = gr.Slider(minimum=1.0, maximum=15.0, step=0.5, label='CFG Scale', value=6, elem_id="cfg_scale")
                eta = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label='ETA', value=1.0, elem_id="eta")
                frame_stride = gr.Slider(minimum=1, maximum=10, step=1, label='Frame Stride', value=5, elem_id="frame_stride")
                steps = gr.Slider(minimum=1, maximum=60, step=1, elem_id="steps", label="Sampling Steps (DDPM)", value=10)

            with gr.Column():
                with gr.Row():
                    use_bezier_curve = gr.Checkbox(label='Use Bézier Curve', elem_id="use_bezier_curve", value=False)
                    loop = gr.Checkbox(label='Loop', elem_id="loop", value=False)
                with gr.Row():
                    bezier_coef_a = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, elem_id="bezier_coef_a", label="Coefficient A", value=0.5)
                    bezier_coef_b = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, elem_id="bezier_coef_a", label="Coefficient B", value=0.5)
                bezier_curve = gr.LinePlot(label="Bézier Curve", elem_id="bezier_curve", value=plot_bezier_curve(0.5, 0.5), x="x", y="y")
                seed = gr.Slider(label='Random Seed', minimum=0, maximum=max_seed, step=1, value=12333)

        gr_examples = gr.Examples(
            examples=load_example(),
            inputs=[input_image, input_text, camera_pose_type, trace_extract_ratio, frame_stride, steps, trace_scale_factor, camera_cfg, cfg_scale, eta, seed, enable_camera_condition, loop,
                    use_bezier_curve, bezier_coef_a, bezier_coef_b],
            # outputs=[output_3d, output_video],
            fn=image2video.get_image,
            examples_per_page=-1,
        )
        end_btn.click(
            inputs=[model_name, input_image, input_text, camera_pose_type, trace_extract_ratio, frame_stride, steps, trace_scale_factor, camera_cfg, cfg_scale, eta, seed, enable_camera_condition,
                    loop, use_bezier_curve, bezier_coef_a, bezier_coef_b, cond_frame_index],
            outputs=[output_3d, output_video1],
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

    dynamicrafter_iface = dynamicrafter_demo('./gradio_results', args.device)
    dynamicrafter_iface.queue(max_size=12)
    dynamicrafter_iface.launch(max_threads=10, server_name=get_ip_addr())
    # dynamicrafter_iface.launch(server_name='0.0.0.0', server_port=80, max_threads=1)
