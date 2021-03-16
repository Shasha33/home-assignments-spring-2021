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
    rodrigues_and_translation_to_view_mat3x4,
    FrameCorners,
    Correspondences
)

def triangulate(ind_1, ind_2, pose_1, pose_2, corner_storage, intrinsic_mat, ids_to_remove, parameters = TriangulationParameters(0.1, 10, 5)):
    frame_corners_1, frame_corners_2 = corner_storage[ind_1], corner_storage[ind_2]
    # print(frame_corners_1.ids, frame_corners_2.ids, ids_to_remove)
    correspondences = build_correspondences(frame_corners_1, frame_corners_2, ids_to_remove)
    view_1, view_2 = pose_to_view_mat3x4(pose_1), pose_to_view_mat3x4(pose_2)
    return triangulate_correspondences(correspondences, view_1, view_2, intrinsic_mat, parameters)


def camera_pose(id, corner_storage, point_cloud_builder, intrinsic_mat, dist_coef = None):
    frame_corners = corner_storage[id]
    points = frame_corners.points
    frame_ids = frame_corners.ids
    cloud_ids = point_cloud_builder.ids
    ids, frame_indexes, cloud_indexes = np.intersect1d(frame_ids, cloud_ids, return_indices=True)
    if ids.shape[0] < 6:
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


def choose_frames(corner_storage, intrinsic_mat, frames_cnt, params = (0.1, 0.8, 0.5, 0.5)):
    ransacReprojThreshold, prob, threshold, inliers_ratio = params

    id_1, id_2 = 0, 1
    view_1, view_2 = np.eye(3, 4), None
    max_points = 0

    num_pairs = frames_cnt / 2 * (frames_cnt - 1) / 2 / 2
    pair_cnt = 0
    for i in range(0, frames_cnt, 2):
        for j in range(i + 1, frames_cnt, 2):
            # print("Testing ", i, j, " current max points: ", max_points)
            pair_cnt += 1
            print("progress: ", int(pair_cnt / num_pairs * 100), "%", sep="")
            correspondences = build_correspondences(corner_storage[i], corner_storage[j])
            if correspondences.ids.shape[0] < 5:
                continue

            points_1, points_2 = correspondences.points_1, correspondences.points_2
            _, homography_mask = cv2.findHomography(points_1, points_2, ransacReprojThreshold = ransacReprojThreshold, method = cv2.RANSAC)
            essential_matrix, essentia_mask = cv2.findEssentialMat(points_1, points_2, prob = prob, threshold = threshold, method = cv2.RANSAC)
            if np.sum(homography_mask == essentia_mask) / essentia_mask.shape[0] < inliers_ratio:
                # print("toо few inliers")
                continue

            R1, R2, t = cv2.decomposeEssentialMat(essential_matrix)
            potential_views = [np.hstack((R1, t)), np.hstack((R2, t)), np.hstack((R1, -t)), np.hstack((R2, -t))]

            inlier_indexes = (essentia_mask == 1).reshape(-1,)
            correspondences = Correspondences(correspondences.ids[inlier_indexes], correspondences.points_1[inlier_indexes], correspondences.points_2[inlier_indexes])
            
            for new_view in potential_views:
                points, ids, cos = triangulate_correspondences(correspondences, view_1, new_view, intrinsic_mat, TriangulationParameters(0.1, 5, 5))
                # print(i, j, points.shape[0])
                if points.shape[0] > max_points :
                    max_points = points.shape[0]
                    view_2 = new_view
                    id_1, id_2 = i, j
            
    if view_2 is None:
        return None, None
    return (id_1, view_mat3x4_to_pose(view_1)), (id_2, view_mat3x4_to_pose(view_2))

            



def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )
    frames_cnt = len(rgb_sequence)
    
    if known_view_1 is None or known_view_2 is None:
        known_view_1, known_view_2 = choose_frames(corner_storage, intrinsic_mat, frames_cnt)
        if known_view_1 is None:
            known_view_1, known_view_2 = choose_frames(corner_storage, intrinsic_mat, frames_cnt, (3, 0.99, 1, 0))

    print("Known frames", known_view_1[0], known_view_2[0])
    unused = np.array([i for i in range(0, frames_cnt) if i not in [known_view_1[0], known_view_2[0]]])
    used = np.array([known_view_1[0], known_view_2[0]])
    used_pose = [known_view_1[1], known_view_2[1]]

    points, ids, cos = triangulate(known_view_1[0], known_view_2[0], known_view_1[1], known_view_2[1], corner_storage, intrinsic_mat, None, TriangulationParameters(10, 0, 0.1)) 
    point_cloud_builder = PointCloudBuilder(ids, points)
    # print("Add frames", known_view_1[0], known_view_2[0])

    while unused.shape[0] > 0:
        added = []
        for i in range(len(unused)):
            result = camera_pose(unused[i], corner_storage, point_cloud_builder, intrinsic_mat)
            if result is None:
                continue
            (pose_i, ids_to_remove) = result
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

        # used_pose = recalculate_poses(used, point_cloud_builder, corner_storage, intrinsic_mat)

    view_mats = [None for i in range(frames_cnt)]
    for i in range(len(used)):
        print(i)
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
