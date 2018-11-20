#!/usr/bin/env python3

import os
import cv2
import time
import numpy as np

import pykitti

IMAGE_DIR='image_3'
KITTI_DIR = 'dataset'
KITTI_SEQ = '02'
KITTI_FRAMES = 50
MIN_FEATURES = 100
TRAJ_WIN_SIZE = 1200
TRAJ_INIT_PT = TRAJ_WIN_SIZE/2
POSES_FILE = 'poses.txt'

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Settings for the LK tracker and the corner detector
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

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

    p0 = None

    # Initialize KITTI dataset.
    ds = pykitti.odometry(KITTI_DIR, KITTI_SEQ, frames=range(0, KITTI_FRAMES))

    # Get camera matrix from calibration data.
    camera_matrix = ds.calib.K_cam3

    # Get initial image.
    imgs = imggen(ds)
    img0 = imgs.__next__()

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

    poses = []
    for img1 in imgs:
        # Check for new features only if we have less that min_features.
        if (p0 is None) or (p0.size < MIN_FEATURES):
            p0 = cv2.goodFeaturesToTrack(img0,
                                         mask = None,
                                         **feature_params)
            mask = np.zeros_like(img0)

        p1, st, err = cv2.calcOpticalFlowPyrLK(img0,
                                               img1,
                                               p0,
                                               None,
                                               **lk_params)

        # Select good points.
        p1_good = p1[st==1]
        p0_good = p0[st==1]

        # Find essential matrix.
        E, _ = cv2.findEssentialMat(p0_good,
                                    p1_good,
                                    camera_matrix,
                                    cv2.RANSAC,
                                    0.999, 1.0)

        # Estimate rotation and translation.
        _, R, t, _ = cv2.recoverPose(E,
                                     p0_good,
                                     p1_good,
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
        for i,(new,old) in enumerate(zip(p1_good,p0_good)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b), (c,d), (255), 2)
            frame = cv2.circle(frame, (a,b), 5, (255), -1)
        img = cv2.add(frame, mask)

        traj = cv2.circle(traj,
                          (x+TRAJ_INIT_PT,y+TRAJ_INIT_PT),
                          2,
                          (255,0,0),
                          -1)

        cv2.imshow('frame', img)
        cv2.imshow('traj', traj)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Use new as old one in next iteration.
        img0 = img1.copy()
        p0 = p1_good.reshape(-1,1,2)

    # Write results to file.
    with open(POSES_FILE, "w+") as f:
        for p in poses:
            s = ' '.join(['%.6e' % i for i in p.flatten()])
            f.write(s + '\n')
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
