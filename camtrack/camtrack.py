#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import cv2

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    TriangulationParameters,
    build_correspondences,
    triangulate_correspondences,
    rodrigues_and_translation_to_view_mat3x4
)

def triangulate(ind_1, ind_2, pose_1, pose_2, corner_storage, intrinsic_mat, ids_to_remove, parameters = TriangulationParameters(8.0, 0, 2)):
    frame_corners_1, frame_corners_2 = corner_storage[ind_1], corner_storage[ind_2]
    correspondences = build_correspondences(frame_corners_1, frame_corners_2, ids_to_remove)
    view_1, view_2 = pose_to_view_mat3x4(pose_1), pose_to_view_mat3x4(pose_2)
    return triangulate_correspondences(correspondences, view_1, view_2, intrinsic_mat, parameters)


def camera_pose(id, corner_storage, point_cloud_builder, intrinsic_mat, dist_coef = None):
    frame_corners = corner_storage[id]
    points = frame_corners.points
    frame_ids = frame_corners.ids
    cloud_ids = point_cloud_builder.ids
    ids, frame_indexes, cloud_indexes = np.intersect1d(frame_ids, cloud_ids, return_indices=True)
    if ids.shape[0] < 4:
        return None 
    
    cloud_points = point_cloud_builder.points[cloud_indexes]
    frame_points = points[frame_indexes]
    retval, rvec, tvec, inliers = cv2.solvePnPRansac(cloud_points, frame_points, intrinsic_mat, dist_coef)
    if retval is None:
        return None
    retval, rvec, tvec = cv2.solvePnP(cloud_points[inliers], frame_points[inliers], intrinsic_mat, dist_coef)
    if retval is None:
        return None 
    return view_mat3x4_to_pose(rodrigues_and_translation_to_view_mat3x4(rvec, tvec)), np.setdiff1d(ids, inliers)


def recalculate_poses(used, point_cloud_builder, corner_storage, intrinsic_mat):
    new_poses = []
    for id in used:
        new_pose, _ = camera_pose(id, corner_storage, point_cloud_builder, intrinsic_mat)
        new_poses.append(new_pose)
    return new_poses


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    frames_cnt = len(rgb_sequence)
    unused = np.array([i for i in range(0, frames_cnt) if i not in [known_view_1[0], known_view_2[0]]])
    used = np.array([known_view_1[0], known_view_2[0]])
    used_pose = [known_view_1[1], known_view_2[1]]

    points, ids, cos = triangulate(known_view_1[0], known_view_2[0], known_view_1[1], known_view_2[1], corner_storage, intrinsic_mat, None) 
    point_cloud_builder = PointCloudBuilder(ids, points)
    # print("Add frames", *known_ids)

    while unused.shape[0] > 0:
        added = []
        for i in range(len(unused)):
            pose_i, ids_to_remove = camera_pose(unused[i], corner_storage, point_cloud_builder, intrinsic_mat)
            if pose_i is None:
                continue

            for j in range(len(used)):
                points, ids, cos = triangulate(unused[i], used[j], pose_i, used_pose[j], corner_storage, intrinsic_mat, ids_to_remove)
                point_cloud_builder.add_points(ids, points)

            used = np.append(used, [unused[i]])
            used_pose.append(pose_i)
            added.append(unused[i])
            # print("Frame", unused[i], "done!!")

        if len(added) == 0:
            break
        unused = np.setdiff1d(unused, added)

        used_pose = recalculate_poses(used, point_cloud_builder, corner_storage, intrinsic_mat)

    view_mats = [None for i in range(frames_cnt)]
    for i in range(len(used)):
        view_mats[used[i]] = pose_to_view_mat3x4(used_pose[i])

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
