# Taken from recorder_console.py from Johannes L. Schoenberger (jsch at inf.ethz.ch),
# https://github.com/microsoft/HoloLensForCV/blob/master/Samples/py/recorder_console.py

import os
import sys
import glob
import tarfile
import argparse
import sqlite3
import shutil
import json
import subprocess
import urllib.request
import numpy as np


def rotmat2qvec(rotmat):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = rotmat.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def read_sensor_poses(path, identity_camera_to_image=False):
    poses = {}
    with open(path, "r") as fid:
        header = fid.readline()
        for line in fid:
            line = line.strip()
            if not line:
                continue
            elems = line.split(",")
            assert len(elems) == 34

            filename = elems[0]
            timestamp = filename.split(".")[0].split("/")[1]
            timestamp = "".join(timestamp.split("_"))
            # Compose the absolute camera pose from the two relative
            # camera poses provided by the recorder application.
            # The absolute camera pose defines the transformation from
            # the world to the camera coordinate system.
            frame_to_origin = np.array(list(map(float, elems[1:17])))
            frame_to_origin = frame_to_origin.reshape(4, 4).T
            camera_to_frame = np.array(list(map(float, elems[17:33])))
            camera_to_frame = camera_to_frame.reshape(4, 4).T
            if abs(np.linalg.det(frame_to_origin[:3, :3]) - 1) < 0.01:
                if identity_camera_to_image:
                    camera_to_image = np.eye(4)
                else:
                    camera_to_image = np.array(
                        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
                poses[int(timestamp)] = np.dot(
                    camera_to_image,
                    np.dot(camera_to_frame, np.linalg.inv(frame_to_origin)))
    return poses


def read_sensor_images(recording_path, camera_name):
    print("Reading images from " + camera_name )
    image_poses = read_sensor_poses(os.path.join(
        recording_path, camera_name + ".csv"))

    image_paths = sorted(glob.glob(
        os.path.join(recording_path,"images", camera_name, "*.jpg")))

    paths = []
    names = []
    time_stamps = []
    poses = []

    for image_path in image_paths:
        basename = os.path.basename(image_path)
        name = os.path.join(camera_name, basename).replace("\\", "/")
        time_stamp = int(os.path.splitext(basename)[0])
        if time_stamp in image_poses:
            paths.append(image_path)
            names.append(name)
            time_stamps.append(time_stamp)
            poses.append(image_poses[time_stamp])
    return paths, names, np.array(time_stamps), poses


def synchronize_sensor_frames(recording_path, output_path):
    # Collect all sensor frames.

    images = {}
    for i in range(0,5):
        camera_name = "camera_" + str(i)
        images[camera_name] = read_sensor_images(recording_path, camera_name)

    numOfImgs = len(images["camera_0"][0])
    sync_frames = []
    sync_poses = []

    for i in range(0,numOfImgs):
        frames = []
        poses = []
        for j in range(0, 5):
            camera_name = "camera_" + str(j)
            frames.append(images[camera_name][1][i])
            poses.append(images[camera_name][3][i])
        sync_frames.append(frames)
        sync_poses.append(poses)


    return sync_frames, sync_poses


def run_reconstruction(workpath):
    reconstruction_path = workpath + "/reconstruction"
    database_path = reconstruction_path + "/database.db"
    image_path = workpath + "/images"
    image_list_path = reconstruction_path + "/image_list.txt"
    sparse_colmap_path = reconstruction_path + "/sparse_colmap"
    sparse_hololens_path = reconstruction_path + "/sparse_hololens"
    dense_path = reconstruction_path + "/dense"
    rig_config_path = reconstruction_path + "/rig_config.json"

    os.makedirs(reconstruction_path)
    os.makedirs(sparse_colmap_path)
    os.makedirs(sparse_hololens_path)
    os.makedirs(dense_path)

    frames, poses = synchronize_sensor_frames(workpath, image_path)

    with open(image_list_path, "w") as fid:
        for frame in frames:
            for image_name in frame:
                fid.write("{}\n".format(image_name))


    subprocess.call([
        "COLMAP.bat", "feature_extractor",
        "--image_path", image_path,
        "--database_path", database_path,
        "--image_list_path", image_list_path,
    ])
    print(3)


    #GET CAMERA CALIBRATION

    # These OpenCV camera model parameters were determined for a specific
    # HoloLens using the self-calibration capabilities of COLMAP.
    # The parameters should be sufficiently accurate as an initialization and
    # the parameters will be refined during the COLMAP reconstruction process.

    #   camera_0 = front camera (pv)
    #   camera_1 = left-left camera (vlc_ll)
    #   camera_2 = left-front camera (vlc_lf)
    #   camera_3 = right-front camera (vlc_rf)
    #   camera_4 = right-right camera (vlc_rr)

    camera_model_id = 4
    camera_model_name = "OPENCV"
    camera_width = [1280,640,640,640,640]
    camera_height = [720,480,480,480,480]
    camera_params = { #TODO get calibration
        "camera_0": "1545.412763409073 1551.2172618412867 597.2250365319205 322.12956014588366",
        "camera_1": "450.072070 450.274345 320 240 "
                  "-0.013211 0.012778 -0.002714 -0.003603",
        "camera_2": "448.189452 452.478090 320 240 "
                  "-0.009463 0.003013 -0.006169 -0.008975",
        "camera_3": "449.435779 453.332057 320 240 "
                  "-0.000305 -0.013207 0.003258 0.001051",
        "camera_4": "450.301002 450.244147 320 240 "
                  "-0.010926 0.008377 -0.003105 -0.004976",
    }


    cameras_file = open(os.path.join(sparse_hololens_path, "cameras.txt"), "w")
    images_file = open(os.path.join(sparse_hololens_path, "images.txt"), "w")
    points_file = open(os.path.join(sparse_hololens_path, "points3D.txt"), "w")

    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    camera_ids = {}
    for i in range(0, 5):
        camera_name = "camera_" + str(i)
        camera_params_list = \
            list(map(float, camera_params[camera_name].split()))
        camera_params_float = np.array(camera_params_list, dtype=np.double)

        cursor.execute("INSERT INTO cameras"
                       "(model, width, height, params, prior_focal_length) "
                       "VALUES(?, ?, ?, ?, ?);",
                       (camera_model_id, camera_width[i],
                        camera_height[i], camera_params_float, 1))

        camera_id = cursor.lastrowid
        camera_ids[camera_name] = camera_id

        cursor.execute("UPDATE images SET camera_id=? "
                       "WHERE name LIKE '{}%';".format(camera_name),
                       (camera_id,))
        connection.commit()

        cameras_file.write("{} {} {} {} {}\n".format(
            camera_id, camera_model_name,
            camera_width, camera_height,
            camera_params[camera_name]))

    for image_names, image_poses in zip(frames, poses):
        for image_name, image_pose in zip(image_names, image_poses):
            camera_name = os.path.dirname(image_name)
            camera_id = camera_ids[camera_name]
            cursor.execute(
                "SELECT image_id FROM images WHERE name=?;", (image_name,))
            image_id = cursor.fetchone()[0]
            qvec = rotmat2qvec(image_pose[:3, :3])
            tvec = image_pose[:, 3]
            images_file.write("{} {} {} {} {} {} {} {} {} {}\n\n".format(
                image_id, qvec[0], qvec[1], qvec[2], qvec[3],
                tvec[0], tvec[1], tvec[2], camera_id, image_name
            ))

    connection.close()

    cameras_file.close()
    images_file.close()
    points_file.close()

    subprocess.call([
        args.colmap_path, "exhaustive_matcher",
        "--database_path", database_path,
        "--SiftMatching.guided_matching", "true",
    ])
    print(4)

    with open(rig_config_path, "w") as fid:
        fid.write("""[
      {{
        "ref_camera_id": {},
        "cameras":
        [
          {{
              "camera_id": {},
              "image_prefix": "camera_0"
          }},
          {{
              "camera_id": {},
              "image_prefix": "camera_1"
          }},
          {{
              "camera_id": {},
              "image_prefix": "camera_2"
          }},
          {{
              "camera_id": {},
              "image_prefix": "camera_3"
          }},
          {{
              "camera_id": {},
              "image_prefix": "camera_4"
          }}
        ]
      }}
    ]""".format(camera_ids["camera_0"],
                camera_ids["camera_0"],
                camera_ids["camera_1"],
                camera_ids["camera_2"],
                camera_ids["camera_3"],
                camera_ids["camera_4"]))

    for i in range(args.num_refinements):
        if i == 0:
            sparse_input_path = sparse_hololens_path
        else:
            sparse_input_path = sparse_colmap_path + str(i - 1)

        sparse_output_path = sparse_colmap_path + str(i)

        mkdir_if_not_exists(sparse_output_path)

        subprocess.call([
            args.colmap_path, "point_triangulator",
            "--database_path", database_path,
            "--image_path", image_path,
            "--import_path", sparse_input_path,
            "--export_path", sparse_output_path,
        ])
        print(5)

        subprocess.call([
            args.colmap_path, "rig_bundle_adjuster",
            "--input_path", sparse_output_path,
            "--output_path", sparse_output_path,
            "--rig_config_path", rig_config_path,
            "--BundleAdjustment.max_num_iterations", str(25),
            "--BundleAdjustment.max_linear_solver_iterations", str(100),
        ])
        print(6)

    if not dense:
        return

    subprocess.call([
        args.colmap_path, "image_undistorter",
        "--image_path", image_path,
        "--input_path", sparse_output_path,
        "--output_path", dense_path,
    ])
    print(7)

    subprocess.call([
        args.colmap_path, "patch_match_stereo",
        "--workspace_path", dense_path,
        "--PatchMatchStereo.geom_consistency", "0",
        "--PatchMatchStereo.min_triangulation_angle", "2",
    ])
    print(8)
    subprocess.call([
        args.colmap_path, "stereo_fusion",
        "--workspace_path", dense_path,
        "--StereoFusion.min_num_pixels", "15",
        "--input_type", "photometric",
        "--output_path", os.path.join(dense_path, "fused.ply"),
    ])


argv = sys.argv
workpath = argv[1]
if not os.path.exists(workpath):
    print("Folder does not exist")
    exit()

run_reconstruction(workpath)

print("Done")

