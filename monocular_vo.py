#!/usr/bin/env python3

import os
import cv2
import time
import numpy as np
import math

import pykitti

HEADLESS = True
IMAGE_DIR='image_3'
KITTI_DIR = 'dataset'
KITTI_SEQ = '02'
KITTI_FRAMES = 500
TRAJ_WIN_SIZE = 1200
TRAJ_INIT_PT = TRAJ_WIN_SIZE/2
POSES_FILE = 'poses.txt'

# Experimentally found.
SCALE=1.18

# Only to convert to grayscale. Not needed for grayscale ds.
def imggen(ds):
    for img in ds.cam2:
        yield np.asarray(img.convert('L'), dtype=np.uint8)

def save_poses(poses):
    # Write results to file.
    with open(POSES_FILE, "w+") as f:
        for p in poses:
            s = ' '.join(['%.6e' % i for i in p.flatten()])
            f.write(s + '\n')

def compute_l1_l2(ds, poses_xy):
    L1 = 0
    L2 = 0
    for i in range(0, KITTI_FRAMES):
        t = ds.poses[i][:3,3]
        x_gt = int(t[0])
        y_gt = -int(t[2])
        x = poses_xy[i][0]
        y = poses_xy[i][1]

        L1 += abs(x_gt - x) + abs(y_gt - y)
        L2 += (x_gt - x)**2 + (y_gt - y)**2
    return L1, math.sqrt(L2)


# 1. Capture images I_t and I_t+1.
# 2. Find keypoints, compute descritors.
# 3. Find good matches with bruteforce matcher.
# 4. Nister's 5-point + RANSAC to compute essential matrix.
# 5. Estimate rotation (R) and translation (t) from essential matrix.
# 6. Draw point on map (x, y) calculated from R and t.
def run(scale=SCALE, max_distance=32, headless=HEADLESS):
    start_time = time.time()
    # Prepare initial position rotation (R) and translation (t) matrices.
    R_final = np.array([[ 1.0, 0.0, 0.0],
                        [ 0.0, 1.0, 0.0],
                        [ 0.0, 0.0, 1.0]])
    t_final = np.zeros((3,1))

    # Initialize KITTI dataset.
    ds = pykitti.odometry(KITTI_DIR, KITTI_SEQ, frames=range(0, KITTI_FRAMES))

    # Get camera matrix from calibration data.
    camera_matrix = ds.calib.K_cam3

    # Get initial image.
    imgs = imggen(ds)

    # Draw ground truth.
    if not headless:
        # Window for trajectory drawing.
        traj = np.zeros((TRAJ_WIN_SIZE, TRAJ_WIN_SIZE, 3), np.uint8)

        for i in range(0, KITTI_FRAMES):
            # Get translation matrix t.
            t = ds.poses[i][:3,3]
            x = int(t[0])
            # y seems inverted in ground truth.
            y = -int(t[2])
            traj = cv2.circle(traj,
                              (int(x + TRAJ_INIT_PT),int(y + TRAJ_INIT_PT)),
                              2,
                              (0,255,0),
                              -1)

    # Initialize ORB. TODO: Tune params.
    orb = cv2.ORB_create()

    # Detect and compute descriptors for first image.
    img0 = imgs.__next__()
    kps0, des0 = orb.detectAndCompute(img0,None)
    kp0 = np.array([(kp.pt[0], kp.pt[1]) for kp in kps0])

    # poses in KITTI format written to file.
    poses = []
    # poses for runtime evaluation.
    poses_xy = []

    # Write poses for the first frame.
    poses.append(np.concatenate((R_final, t_final), axis=1))
    x = int(t_final[0][0])
    y = int(t_final[2][0])
    poses_xy.append((x,y))
    for img1 in imgs:
        # Detect and compute descriptors.
        kps1, des1 = orb.detectAndCompute(img1,None)
        kp1 = np.array([(kp.pt[0], kp.pt[1]) for kp in kps1])

        # Match features.
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.match(des0,des1)

        good_kp0 = []
        good_kp1 = []
        for m in matches:
            if m.distance < max_distance:
                good_kp0.append(kp0[m.queryIdx])
                good_kp1.append(kp1[m.trainIdx])
        good_kp0 = np.array(good_kp0)
        good_kp1 = np.array(good_kp1)

        E, _ = cv2.findEssentialMat(good_kp0,
                                    good_kp1,
                                    camera_matrix,
                                    cv2.RANSAC,
                                    0.999, 1.0)

        # Estimate rotation and translation.
        _, R, t, _ = cv2.recoverPose(E,
                                     good_kp0,
                                     good_kp1,
                                     camera_matrix)

        t = t * scale

        # Accumulate rotation and translation.
        t_final += np.dot(R_final, t)
        R_final = np.dot(R, R_final)

        # Final coordinates.
        x = int(t_final[0][0])
        y = int(t_final[2][0])

        poses.append(np.concatenate((R_final, t_final), axis=1))
        poses_xy.append((x,y))

        if not headless:
           frame = img1.copy()
           traj = cv2.circle(traj,
                             (int(x+TRAJ_INIT_PT),int(y+TRAJ_INIT_PT)),
                             2,
                             (255,0,0),
                             -1)

           cv2.imshow('frame', frame)
           cv2.imshow('traj', traj)

           k = cv2.waitKey(30) & 0xff
           if k == 27:
               break

        # Use new as old one in next iteration.
        kp0 = kp1.copy()
        des0 = des1.copy()

    stop_time = time.time()

    save_poses(poses)
    L1, L2 = compute_l1_l2(ds, poses_xy)

    print("Final L2:", L2)
    print("Final duration (s):", stop_time - start_time)

    if not headless:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
