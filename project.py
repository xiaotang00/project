import numpy as np
import cv2
import matplotlib.pyplot as plt
# from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from pathlib import Path
# from functools import reduce
from nuscenes.nuscenes import NuScenes
import os
from matplotlib import cm  # 用于颜色映射

dataroot = "/data/nuScenes/"
nusc = NuScenes(version="v1.0-mini", dataroot=dataroot, verbose=True)

def get_matrix(calibrated_data, inverse=False):
    # 返回的结果是坐标系的变换矩阵
    output = np.eye(4)
    output[:3, :3] = Quaternion(calibrated_data["rotation"]).rotation_matrix
    output[:3,  3] = calibrated_data["translation"]
    if inverse:
        output = np.linalg.inv(output)
    return output

def project_lidar_to_image(lidar_points, lidar_to_global, global_to_image):
    """
    投影单帧点云到图像
    """
    hom_points = np.concatenate([lidar_points[:, :3], np.ones((len(lidar_points), 1))], axis=1)
    global_points = hom_points @ lidar_to_global.T
    image_points = global_points @ global_to_image.T
    image_points[:, :2] /= image_points[:, [2]]  # 归一化
    return image_points

def filter_lidar_points_in_fov(lidar_points, fov_deg=120):
    """
    过滤点云，满足前向 FOV 和特定条件（x > 1 或 z > 1）。
    """
    # 提取相机坐标系下的点云信息
    x, y, z = lidar_points[:, 0], lidar_points[:, 1], lidar_points[:, 2]

    depth_condition = z > 6
    fov_rad = np.deg2rad(fov_deg)  # 转换为弧度
    angle_condition = np.abs(np.arctan2(y, z)) <= fov_rad / 2

    # 结合所有条件
    valid_mask = depth_condition & angle_condition

    # 筛选点云
    filtered_lidar_points = lidar_points[valid_mask]

    return filtered_lidar_points

def points_lidar_to_camera(lidar_points, lidar_to_global, global_to_camera):
    hom_points = np.concatenate([lidar_points[:, :3], np.ones((len(lidar_points), 1))], axis=1)

    lidar_to_camera = global_to_camera @ lidar_to_global # 将当前点云帧lidar坐标系转换到camera坐标系的矩阵
    camera_points = hom_points @ lidar_to_camera.T

    # global_points = hom_points @ lidar_to_global.T
    # camera_points = global_points @ global_to_camera.T

    return camera_points

def prev_multi_sweeps(lidar_sample_data_rec, max_sweeps, sweeps):
    ref_time = 1e-6 * lidar_sample_data_rec["timestamp"]
    lidar_calibrated_data_rec = nusc.get("calibrated_sensor", lidar_sample_data_rec["calibrated_sensor_token"])
    lidar_to_ego = get_matrix(lidar_calibrated_data_rec)
    ego_pose = nusc.get("ego_pose", lidar_sample_data_rec["ego_pose_token"])
    ego_to_global = get_matrix(ego_pose)
    lidar_to_global = ego_to_global @ lidar_to_ego
    curr_sd_rec = lidar_sample_data_rec  # 从当前帧开始

    while len(sweeps) < max_sweeps:
        if curr_sd_rec["prev"] == "":
            if len(sweeps) == 0:
                lidar_path = os.path.join(dataroot, nusc.get_sample_data_path(curr_sd_rec["token"]))
                sweep = {
                    "lidar_path": Path(lidar_path).relative_to(dataroot).__str__(),
                    "sample_data_token": curr_sd_rec["token"],
                    # "transform_matrix": None,
                    "current_to_global": lidar_to_global,
                    "time_lag": curr_sd_rec["timestamp"] * 0,
                }
                sweeps.append(sweep)
            else:
                break
                sweeps.append(sweeps[-1])
        else:
            print("curr_sd_rec[ego_pose_token]", curr_sd_rec["ego_pose_token"])
            # 获取当前帧的pose和transform矩阵
            current_pose_rec = nusc.get("ego_pose", curr_sd_rec["ego_pose_token"])
            ego_to_global = get_matrix(current_pose_rec)

            current_cs_rec = nusc.get("calibrated_sensor", curr_sd_rec["calibrated_sensor_token"])
            current_to_ego = get_matrix(current_cs_rec)

            current_to_global = ego_to_global @ current_to_ego
            # tm = reduce(
            #     np.dot,
            #     [ref_from_car, car_from_global, global_from_car, car_from_current],
            # )

            # lidar_path = nusc.get_sample_data_path(curr_sd_rec["token"])
            time_lag = ref_time - 1e-6 * curr_sd_rec["timestamp"]

            lidar_path = os.path.join(dataroot, nusc.get_sample_data_path(curr_sd_rec["token"]))
            sweep = {
                "lidar_path": Path(lidar_path).relative_to(dataroot).__str__(),
                "sample_data_token": curr_sd_rec["token"],
                # "transform_matrix": tm,
                # "global_from_car": global_from_car,
                # "car_from_current": car_from_current,
                "current_to_global": current_to_global,
                "time_lag": time_lag,
            }
            sweeps.append(sweep)

            curr_sd_rec = nusc.get("sample_data", curr_sd_rec["prev"])

    sweeps.reverse()
    return sweeps

def next_multi_sweeps(lidar_sample_data_rec, max_sweeps, sweeps):
    ref_time = 1e-6 * lidar_sample_data_rec["timestamp"]
    lidar_calibrated_data_rec = nusc.get("calibrated_sensor", lidar_sample_data_rec["calibrated_sensor_token"])
    lidar_to_ego = get_matrix(lidar_calibrated_data_rec)
    ego_pose = nusc.get("ego_pose", lidar_sample_data_rec["ego_pose_token"])
    ego_to_global = get_matrix(ego_pose)
    lidar_to_global = ego_to_global @ lidar_to_ego
    prev_sweeps = len(sweeps)  # 已有点云帧数
    curr_sd_rec = lidar_sample_data_rec  # 从当前帧开始

    while len(sweeps) < max_sweeps + prev_sweeps :
        if curr_sd_rec["next"] == "":
            if len(sweeps) == 0:
                lidar_path = os.path.join(dataroot, nusc.get_sample_data_path(curr_sd_rec["token"]))
                sweep = {
                    "lidar_path": Path(lidar_path).relative_to(dataroot).__str__(),
                    "sample_data_token": curr_sd_rec["token"],
                    # "transform_matrix": None,
                    "current_to_global": lidar_to_global,
                    "time_lag": curr_sd_rec["timestamp"] * 0,
                }
                sweeps.append(sweep)
            else:
                break
                sweeps.append(sweeps[-1])
        else:
            print("curr_sd_rec[ego_pose_token]", curr_sd_rec["ego_pose_token"])
            # 获取当前帧的pose和transform矩阵
            current_pose_rec = nusc.get("ego_pose", curr_sd_rec["ego_pose_token"])
            ego_to_global = get_matrix(current_pose_rec)

            current_cs_rec = nusc.get("calibrated_sensor", curr_sd_rec["calibrated_sensor_token"])
            current_to_ego = get_matrix(current_cs_rec)

            current_to_global = ego_to_global @ current_to_ego
            # tm = reduce(
            #     np.dot,
            #     [ref_from_car, car_from_global, global_from_car, car_from_current],
            # )

            # lidar_path = nusc.get_sample_data_path(curr_sd_rec["token"])
            time_lag = ref_time + 1e-6 * curr_sd_rec["timestamp"]

            lidar_path = os.path.join(dataroot, nusc.get_sample_data_path(curr_sd_rec["token"]))
            sweep = {
                "lidar_path": Path(lidar_path).relative_to(dataroot).__str__(),
                "sample_data_token": curr_sd_rec["token"],
                # "transform_matrix": tm,
                # "global_from_car": global_from_car,
                # "car_from_current": car_from_current,
                "current_to_global": current_to_global,
                "time_lag": time_lag,
            }
            sweeps.append(sweep)
            curr_sd_rec = nusc.get("sample_data", curr_sd_rec["next"])

    # sweeps.reverse()
    return sweeps

def multi_project(ref_chan, cam_chan):
    sample = nusc.sample[3]  # 使用第一帧样本

    # 获取相机参数
    cameras = cam_chan
    camera_token = sample["data"][cameras]
    camera_data = nusc.get("sample_data", camera_token)
    camera_ego_pose = nusc.get("ego_pose", camera_data["ego_pose_token"])
    global_to_ego = get_matrix(camera_ego_pose, True)
    camera_calibrated_data = nusc.get("calibrated_sensor", camera_data["calibrated_sensor_token"])
    ego_to_camera = get_matrix(camera_calibrated_data, True)
    camera_intrinsic = np.eye(4)
    camera_intrinsic[:3, :3] = camera_calibrated_data["camera_intrinsic"] # 相机内参
    global_to_camera = ego_to_camera @ global_to_ego
    # global_to_image = camera_intrinsic @ ego_to_camera @ global_to_ego
    # print("完成global_to_image", global_to_image)

    lidar_token = sample["data"][ref_chan]
    lidar_sample_data_rec = nusc.get('sample_data', lidar_token)

    # 获取多帧点云
    sweeps=[]
    sweeps = prev_multi_sweeps(lidar_sample_data_rec, max_sweeps=15, sweeps=sweeps) # 向前取帧
    print("len(prev_sweeps)", len(sweeps))
    sweeps = next_multi_sweeps(lidar_sample_data_rec, max_sweeps=15, sweeps=sweeps) # 向后取帧
    print("len(prev_sweeps + next_sweeps)", len(sweeps))
    # sweeps.reverse()
    print("finish sweeps !")

    # 初始化合成的图像
    front_image_points = []
    for sweep in sweeps:
        lidar_file = os.path.join(dataroot, sweep["lidar_path"])
        lidar_points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 5)

        # 投影每帧点云
        all_camera_points = points_lidar_to_camera(lidar_points, sweep["current_to_global"], global_to_camera)
        # image_points = project_lidar_to_image(lidar_points, sweep["current_to_global"], global_to_image)
        camera_points = filter_lidar_points_in_fov(all_camera_points)

        image_points = camera_points @ camera_intrinsic.T
        image_points[:, :2] /= image_points[:, [2]]  # 归一化

        front_image_points.append(image_points)


    # 合并所有帧的投影点
    all_image_points_combine = np.vstack(front_image_points)

    # 加载相机图像
    image_file = os.path.join(dataroot, camera_data["filename"])
    image = cv2.imread(image_file)

    print("all_image_points_combine:", all_image_points_combine.shape)
    print("all_image_points: ", len(front_image_points), ",", len(front_image_points[0]))
    print("all_image_points_combine[0]:", all_image_points_combine[0])

    
    # 获取所有点的深度值
    depths = all_image_points_combine[all_image_points_combine[:, 2] > 0, 2]
    # 使用matplotlib的颜色映射
    cmap = cm.get_cmap('plasma')
    norm = plt.Normalize(vmin=np.min(depths), vmax=np.max(depths))  # 归一化深度值

    # 对每个投影点根据深度值设置颜色
    for x, y, z in all_image_points_combine[all_image_points_combine[:, 2] > 0, :3].astype(int):
        if not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
            continue  # 跳过无效点
        # 获取对应的颜色
        color = cmap(norm(z))  
        color = (color[2] * 255, color[1] * 255, color[0] * 255)  
        # 绘制点
        cv2.circle(image, (int(x), int(y)), 3, color, -1, 16)

    # 保存最终的图像
    cv2.imwrite("/home/neu-wang/tang/project/vis-k3d/combined_projection_depth_color.jpg", image)
    print("finish")

def main():

    ref_chan = "LIDAR_TOP"
    cam_chan = "CAM_FRONT"

    multi_project(ref_chan, cam_chan)

if __name__ == "__main__":
    main()








