from argparse import ArgumentParser

import imageio
import numpy as np
import open3d as o3d
from tqdm import tqdm

def select_visible_points(point_cloud, viewpoint):
    """
    从给定视点选择点云中可见的点
    :param point_cloud: 输入的点云 (o3d.geometry.PointCloud)
    :param viewpoint: 视点位置 (numpy array, shape=(3,))
    :return: 可见点云 (o3d.geometry.PointCloud)
    """
    # Step 1: 将点云移动到以视点为原点的坐标系
    points = np.asarray(point_cloud.points)
    transformed_points = points - viewpoint  # 平移点云到视点坐标系

    # Step 2: 创建变换后的点云
    transformed_pcd = o3d.geometry.PointCloud()
    transformed_pcd.points = o3d.utility.Vector3dVector(transformed_points)

    # Step 3: 计算变换后点云的凸包
    hull, hull_indices = transformed_pcd.compute_convex_hull()

    # Step 4: 从原始点云中选择可见点（凸包上的点）
    visible_indices = np.unique(hull_indices)
    visible_points = points[visible_indices]

    # Step 5: 构建可见点云
    visible_pcd = o3d.geometry.PointCloud()
    visible_pcd.points = o3d.utility.Vector3dVector(visible_points)

    return visible_pcd


def render(
    size: tuple[int, int], intrinsics: np.ndarray, w2cs: np.ndarray, points: np.ndarray, colors: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:

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
    for w2c in tqdm(w2cs, desc="Rendering previews"):
        renderer.setup_camera(K, w2c)


        # camera = np.array([0.0, 0.0, 0.0])  # 视点位置
        radius = 300.0  # 视点到点云的最大距离
        # _, visible_indices = pcd.hidden_point_removal(np.linalg.inv(w2c)[:3,3].reshape(-1), radius)
        # visible_pcd = pcd.select_by_index(visible_indices)
        visible_pcd = pcd
        scene.clear_geometry()
        scene.add_geometry("point cloud", visible_pcd, mat)

        rgb_frames.append(renderer.render_to_image())

    return np.stack(rgb_frames)


# def render_with_depth(
#     size: tuple[int, int], intrinsics: np.ndarray, w2cs: np.ndarray, points: np.ndarray, colors: np.ndarray
# ) -> tuple[np.ndarray, np.ndarray]:
#
#     renderer = o3d.visualization.rendering.OffscreenRenderer(*size)
#
#     K = o3d.camera.PinholeCameraIntrinsic(*size, intrinsics)
#
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd.colors = o3d.utility.Vector3dVector(colors)
#
#     mat = o3d.visualization.rendering.MaterialRecord()
#     mat.shader = "defaultUnlit"
#     # mat.point_size = 3
#
#     scene: o3d.visualization.rendering.Open3DScene = renderer.scene
#     scene.set_background(np.array([0.0, 0.0, 0.0, 1.0]))
#     scene.view.set_post_processing(False)
#     scene.clear_geometry()
#     scene.add_geometry("point cloud", pcd, mat)
#
#     rgb_frames, depth_frames = [], []
#     for w2c in tqdm(w2cs, desc="Rendering previews"):
#         renderer.setup_camera(K, w2c)
#         rgb_frames.append(renderer.render_to_image())
#         depth_frames.append(renderer.render_to_depth_image(z_in_view_space=True))
#
#     return np.stack(rgb_frames), np.where(np.stack(depth_frames) == np.inf, 0.0, 1.0)
#
#
# def render_from_npz(data_path: str, output_path: str):
#     data = np.load(data_path, allow_pickle=True)["arr_0"].item()
#
#     renderer = o3d.visualization.rendering.OffscreenRenderer(*data["size"])
#
#     K = o3d.camera.PinholeCameraIntrinsic(*data["size"], data["K"])
#
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(data["points"])
#     pcd.colors = o3d.utility.Vector3dVector(data["colors"])
#
#     mat = o3d.visualization.rendering.MaterialRecord()
#     mat.shader = "defaultUnlit"
#     # mat.point_size = 3
#
#     scene: o3d.visualization.rendering.Open3DScene = renderer.scene
#     scene.set_background(np.array([0.0, 0.0, 0.0, 1.0]))
#     scene.view.set_post_processing(False)
#     scene.clear_geometry()
#     scene.add_geometry("point cloud", pcd, mat)
#
#     rgb_frames, depth_frames = [], []
#     for w2c in tqdm(data["w2cs"], desc="Rendering previews"):
#         renderer.setup_camera(K, w2c)
#         rgb_frames.append(renderer.render_to_image())
#         depth_frames.append(renderer.render_to_depth_image(z_in_view_space=True))
#
#     imageio.mimsave(output_path, np.stack(rgb_frames), fps=8)
#     imageio.mimsave("test.mp4", np.where(np.stack(depth_frames) == np.inf, 0, 255).astype(np.uint8), fps=8)


if __name__ == "__main__":
    o3d.visualization.rendering.OffscreenRenderer(512, 320) # test
