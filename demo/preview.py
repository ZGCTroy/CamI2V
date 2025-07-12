import numpy as np
import open3d as o3d
from pyvirtualdisplay import Display
from tqdm import tqdm


def render(
    size: tuple[int, int], intrinsics: np.ndarray, w2cs: np.ndarray, points: np.ndarray, colors: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    
    with Display(visible=False, size=(512, 320)):

        renderer = o3d.visualization.rendering.OffscreenRenderer(*size)

        K = o3d.camera.PinholeCameraIntrinsic(*size, intrinsics)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 2

        scene: o3d.visualization.rendering.Open3DScene = renderer.scene
        scene.set_background(np.array([0.0, 0.0, 0.0, 1.0]))
        scene.view.set_post_processing(False)
        scene.clear_geometry()
        scene.add_geometry("point cloud", pcd, mat)

        rgb_frames = []
        for w2c in tqdm(w2cs, desc="Rendering"):
            renderer.setup_camera(K, w2c)
            rgb_frames.append(renderer.render_to_image())

    return np.stack(rgb_frames)


if __name__ == "__main__":
    with Display(visible=False, size=(512, 320)):
        renderer = o3d.visualization.rendering.OffscreenRenderer(512, 320)

        mesh = o3d.geometry.TriangleMesh.create_box()
        mesh.paint_uniform_color([1, 0, 0])

        renderer.scene.add_geometry("cube", mesh)
        img = renderer.render_to_image()
        o3d.io.write_image("test.png", img)
