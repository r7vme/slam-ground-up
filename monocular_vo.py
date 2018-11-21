#!/usr/bin/env python3

import os
import cv2
import time
import numpy as np

import pykitti

IMAGE_DIR='image_3'
KITTI_DIR = 'dataset'
KITTI_SEQ = '02'
KITTI_FRAMES = 1000
TRAJ_WIN_SIZE = 1200
TRAJ_INIT_PT = TRAJ_WIN_SIZE/2
POSES_FILE = 'poses.txt'

# Only to convert to grayscale. Not needed for grayscale ds.
def imggen(ds):
    for img in ds.cam2:
        yield np.asarray(img.convert('L'), dtype=np.uint8)

# 1. Capture images I_t and I_t+1.
# 2. Detect features on I_t and compute optical flow with I_t+1.
# 3. Nister's 5-point + RANSAC to compute essential matrix.
# 4. Estimate rotation (R) and translation (t) from essential matrix.
# 5. Draw point on map (x, y) calculated from R and t.
def run():
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

    # Window for trajectory drawing.
    traj = np.zeros((TRAJ_WIN_SIZE, TRAJ_WIN_SIZE, 3), np.uint8)

    # Draw ground truth.
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

    poses = []
    for img1 in imgs:
        # Detect and compute descriptors.
        kps1, des1 = orb.detectAndCompute(img1,None)
        kp1 = np.array([(kp.pt[0], kp.pt[1]) for kp in kps1])

        # Match features.
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.match(des0,des1)
        matches = [m for m in matches if m.distance < 32]
        print(len(matches))
        # TODO: It's slow and does not work!!!

        E, _ = cv2.findEssentialMat(kp0,
                                    kp1,
                                    camera_matrix,
                                    cv2.RANSAC,
                                    0.999, 1.0)

        # Estimate rotation and translation.
        _, R, t, _ = cv2.recoverPose(E,
                                     kp0,
                                     kp1,
                                     camera_matrix)

        # TODO: We also need to scale our coordinates.
        # For now let's say we use relative coordinates.

        # Accumulate rotation and translation.
        t_final += np.dot(R_final, t)
        R_final = np.dot(R, R_final)

        poses.append(np.concatenate((R_final, t_final), axis=1))

        # Final coordinates.
        x = t_final[0]
        y = t_final[2]

        frame = img1.copy()
        # Draw the tracks.
        #for i,(new,old) in p1_good,:
        #    a,b = new.ravel()
        #    frame = cv2.circle(frame, (a,b), 5, (255), -1)

        traj = cv2.circle(traj,
                          (x+TRAJ_INIT_PT,y+TRAJ_INIT_PT),
                          2,
                          (255,0,0),
                          -1)

        cv2.imshow('frame', frame)
        cv2.imshow('traj', traj)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Use new as old one in next iteration.
        img0 = img1.copy()
        kp0 = kp1.copy()
        des0 = des1.copy()

    # Write results to file.
    with open(POSES_FILE, "w+") as f:
        for p in poses:
            s = ' '.join(['%.6e' % i for i in p.flatten()])
            f.write(s + '\n')
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
