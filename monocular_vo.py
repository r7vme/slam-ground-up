#!/usr/bin/env python3

import os
import cv2
import time
import numpy as np

IMAGE_DIR="image_3"
MIN_FEATURES = 50
CAMERA_FOCAL = 707
CAMERA_PP = (602, 183)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Settings for the LK tracker and the corner detector
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


# imggen gererator that reads images from
# directory and convert then to gray.
# TODO: Filter non PNG files.
def imggen():
    for f in sorted(os.listdir(IMAGE_DIR)):
        img_path = os.path.join(IMAGE_DIR, f)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(cv2.COLOR_BGR2GRAY)
        yield img

# 1. Capture images I_t and I_t+1.
# 2. Undostort images I_t and I_t+1.
# 3. Detect features (ORB) I_t and track features I_t+1.
# 4. Nister's 5-point + RANSAC to compute essential matrix.
# 5. Estimate rotation (R) and translation (t) from essential matrix.
# 6. Scale R and t with external source.
def run():
    # Prepare initial position rotation (R) and translation (t) matrices.
    R_final = np.array([[ 1.0, 0.0, 0.0],
                        [ 0.0, 1.0, 0.0],
                        [ 0.0, 0.0, 1.0]])
    t_final = np.zeros((3,1))

    p0 = None

    imgs = imggen()

    # Get initial image.
    img0 = imgs.__next__()

    # Mask for drawing feature trajectories.
    mask = np.zeros_like(img0)

    # Window for trajectory drawing.
    win_size = 512
    traj = np.zeros((win_size,win_size,3), np.uint8)

    for img1 in imgs:
        # Check for new features only if we have less that min_features.
        if (p0 is None) or (p0.size < MIN_FEATURES):
            p0 = cv2.goodFeaturesToTrack(img0,
                                         mask = None,
                                         **feature_params)

        p1, st, err = cv2.calcOpticalFlowPyrLK(img0,
                                               frame,
                                               p0,
                                               None,
                                               **lk_params)

        # Select good points
        p1_good = p1[st==1]
        p0_good = p0[st==1]

        # Find essential matrix.
        E, _ = cv2.findEssentialMat(p0_good,
                                    p1_good,
                                    CAMERA_FOCAL,
                                    CAMERA_PP,
                                    cv2.RANSAC,
                                    0.999, 1.0)

        # Estimate rotation and translation.
        _, R, t, _ = cv2.recoverPose(E,
                                     p0_good,
                                     p1_good,
                                     focal=CAMERA_FOCAL,
                                     pp=CAMERA_PP)

        # TODO: We also need to scale our coordinates.
        # For now let's say we use relative coordinates.

        # Accumulate rotation and translation.
        R_final = np.dot(R, R_final)
        t_final += np.dot(R_final, t)

        # Final coordinates.
        x = t_final[0]
        y = t_final[2]


        # draw the tracks
        for i,(new,old) in enumerate(zip(p1_good,p0_good)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b), (c,d), (0,255,0), 2)
            frame = cv2.circle(frame, (a,b), 5, (0,128,0), -1)
        img = cv2.add(frame, mask)

        traj = cv2.circle(traj, (x,y), 5, (0,128,0), -1)

        cv2.imshow('frame', img)
        cv2.imshow('traj', traj)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Use new as old one in next iteration.
        img0 = img1.copy()
        p0 = p1_good.reshape(-1,1,2)

def main():
    run()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
