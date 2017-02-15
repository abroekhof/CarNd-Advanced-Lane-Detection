import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

class Undistortor:
    def __init__(self):
        self.mtx = None
        self.dist = None
        self.image_shape = None

    def create_calibration_points(self):
        numx, numy = 9, 6
        objp = np.zeros((numx*numy, 3), np.float32)
        objp[:, :2] = np.mgrid[0:numx, 0:numy].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob('./camera_cal/calibration*.jpg')

        # Step through the list and search for chessboard corners
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (numx, numy), None)

            # If found, add object points, image points
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

        self.image_shape = gray.shape[::-1]

        return objpoints, imgpoints

    def calibrate_camera(self):
        if not self.image_shape:
            return
        objpoints, imgpoints = self.create_calibration_points()
        ret, self.mtx, self.dist, _, _ = cv2.calibrateCamera(
            objpoints, imgpoints, self.image_shape, None, None
        )
        return ret

    def undistort(self, img):
        if not (self.dist or self.mtx):
            self.calibrate_camera()
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

