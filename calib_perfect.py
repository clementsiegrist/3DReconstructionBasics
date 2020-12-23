'''
References :

https://www.pcigeomatics.com/geomatica-help/concepts/orthoengine_c/Chapter_45.html
https://stackoverflow.com/questions/29317262/opencv-video-saving-in-python
https://www.learnopencv.com/camera-calibration-using-opencv/ methods described
https://stackoverflow.com/questions/27115862/taking-multiple-pictures-in-opencv c++
https://stackoverflow.com/questions/42119899/opencv-cv2-python-findchessboardcorners-failing-on-seemingly-simple-chessboard
Good features to track
https://stackoverflow.com/questions/31249037/calibrating-webcam-using-python-and-opencv-error
https://stackoverflow.com/questions/29628445/meaning-of-the-retval-return-value-in-cv2-calibratecamera
https://docs.opencv.org/master/d9/d0c/group__calib3d.html
https://www.researchgate.net/post/How_to_find_aperture_width_and_aperture_height_of_a_camera
https://support.apple.com/kb/SP747?locale=en_US


'''

import argparse
import os
import numpy as np
import cv2
import glob
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def video_to_frames(path_output_dir):
    # extract frames from a video and save to directory as 'x.png' where
    # x is the frame index
    import time
    fpsLimit = 1  # throttle limit
    startTime = time.time()
    vidcap = cv2.VideoCapture(0)
    count = 0
    while vidcap.isOpened():
        nowTime = time.time()
        if (int(nowTime - startTime)) > fpsLimit:
            success, image = vidcap.read()
            if success:
                cv2.imwrite(os.path.join(path_output_dir, '%d.png') % count, image)
                count += 1
                if count / 2 == 4:
                    break
                else:
                    pass
                startTime = time.time()

    cv2.destroyAllWindows()

def videoclickcapture():
    # import the opencv library
    import cv2

    # define a video capture object
    vid = cv2.VideoCapture(0)

    while (True):

        # Capture the video frame
        # by frame
        ret, frame = vid.read()

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('c'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


def video_to_frames_(path_output_dir):
    # extract frames from a video and save to directory as 'x.png' where
    # x is the frame index
    import time
    fpsLimit = 1  # throttle limit
    startTime = time.time()
    vidcap = cv2.VideoCapture(0)
    count = 0
    while vidcap.isOpened():
        nowTime = time.time()
        if (int(nowTime - startTime)) > fpsLimit:
            success, image = vidcap.read()
            if success:
                cv2.imwrite(os.path.join(path_output_dir, '%d.png') % count, image)
                count += 1
                if count / 2 == 10:
                    break
                else:
                    pass
                startTime = time.time()

    cv2.destroyAllWindows()
    vidcap.release()

def calibrate(dirpath):  # prefix, image_format, square_size, width=9, height=6):

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((5 * 5, 3), np.float32)
    objp[:, :2] = np.mgrid[0:5, 0:5].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob(dirpath)
    # '/Users/clementsiegrist/untitled7/img/*.png'

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        # Other methods could be used for point matching as RANSAC, SIFT, LMEDS
        corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)

        for i in corners:
            x, y = i.ravel()
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('Corners', img)
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)  # Draw and display the corners
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        return ret, mtx, dist, rvecs, tvecs

    # ret = Root Mean Square Error
    # The RMSE calculation is done by projecting
    # the 3D chessboard points (objectPoints)
    # into the image plane using the final set of
    # calibration parameters (cameraMatrix, distCoeffs, rvecs and tvecs)
    # and comparing the known position of the corners (imagePoints).
    # mtx = camera Matrix
    # dist = distortion coefficient : radial distortion coefficient are k1, k2, k3 / tangential distortion coeffcients are p1 and p2
    # rvecs = rotation vectors
    # tvecs = translation vectors

def calibratextended_findchess(dirpath):
  '''
    dirpath : /content/img_ where img obtained with video_to_frames() are stored.
    path_params : /content/interest_points where to store the homologeous points.
  '''
  # termination criteria
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

  # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
  objp = np.zeros((6*7,3), np.float32)
  objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

  # Arrays to store object points and image points from all the images.
  objpoints = [] # 3d point in real world space
  imgpoints = [] # 2d points in image plane.

  images = glob.glob(os.path.join(dirpath, '*.png'))

  for fname in images:
      img = cv2.imread(fname)
      gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

      # Find the chess board corners
      ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
      #params = corners
      #np.savetxt(path_params, params)

      print('ret is {}'.format(ret))
      # If found, add object points, image points (after refining them)
      if ret == True:

          objpoints.append(objp)
          corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
          imgpoints.append(corners2)

          # Draw and display the corners
          img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
          cv2.imshow('img',img)
          retval, cameraMatrix, distCoeffs, rvecs, \
          tvecs, stdDeviationsIntrinsics, \
          stdDeviationsExtrinsics, perViewErrors = cv2.calibrateCameraExtended(objpoints, imgpoints, gray.shape[::-1],
                                                                               None, None, flags=cv2.CALIB_FIX_FOCAL_LENGTH)

          return retval, cameraMatrix, distCoeffs, rvecs, \
          tvecs, stdDeviationsIntrinsics, \
          stdDeviationsExtrinsics, perViewErrors

    # ret = Root Mean Square Error
    # The RMSE calculation is done by projecting
    # the 3D chessboard points (objectPoints)
    # into the image plane using the final set of
    # calibration parameters (cameraMatrix, distCoeffs, rvecs and tvecs)
    # and comparing the known position of the corners (imagePoints).
    # mtx = camera Matrix
    # dist = distortion coefficient : radial distortion coefficient are k1, k2, k3 / tangential distortion coeffcients are p1 and p2
    # rvecs = rotation vectors
    # tvecs = translation vectors

def undis_repro(img_test, repro_name, mtx, dist):

    # Load new image
    img = cv2.imread(img_test)
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv2.imwrite(repro_name, dst)

def undis_repro_(img_test, repro_name, mtx, dist):

    # Load new image
    img = cv2.imread(img_test)
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    #x, y, w, h = roi
    #dst = dst[y:y + h, x:x + w]
    cv2.imwrite(repro_name+'.jpg', dst)
    return newcameramtx, dst

def undis_repro_multiple(img_path, repro_name, mtx, dist):

    # Load new image
    images = glob.glob(os.path.join(img_path, '*.png'))

    for fname in images:
        #for i in range(2):
            img = cv2.imread(fname)
            h, w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
            # undistort
            dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
            # crop the image
            cv2.imwrite(repro_name, dst)
            return newcameramtx

def save_coefficients(ret, mtx, dist, rvecs, tvecs, path):
    """ Save the camera matrix and the distortion coefficients to given path/file.

    ret = Root Mean Square Error thhe RMSE calculation is done by projecting the 3D chessboard
         points (objectPoints) into the image plane using the final set of calibration parameters
         (cameraMatrix, distCoeffs, rvecs and tvecs) and comparing the known position of the corners (imagePoints).

    mtx = camera Matrix
    dist = distortion coefficient : radial distortion coefficient are k1, k2, k3 / tangential distortion coeffcients are p1 and p2
    rvecs = rotation vectors
    tvecs = translation vectors

    """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    rvecs = np.array(rvecs)
    tvecs = np.array(tvecs)
    cv_file.write("RMSE", ret)
    cv_file.write("K", mtx)
    cv_file.write("D", dist)
    cv_file.write("rvecs", rvecs)
    cv_file.write("tvecs", tvecs)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()


def load_coefficients(path):
    """ Loads camera matrix and distortion coefficients. """
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()

    cv_file.release()
    return [camera_matrix, dist_matrix]

def matches_img(img_1, img_2, store_path):
    '''
    https://stackoverflow.com/questions/54203873/how-to-get-the-positions-of-the-matched-points-with-brute-force-matching-sift/54220651#54220651
    :param img_1:
    :param img_2:
    :param store_path:
    :return:
    '''

    import matplotlib.pyplot as plt

    img1 = cv2.imread(img_1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img_2, cv2.IMREAD_COLOR)
    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    print(matches)
    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3), \
    plt.show()
    cv_file = cv2.FileStorage(store_path, cv2.FILE_STORAGE_WRITE)
    kp1, kp2 = np.array(kp1), np.array(kp2)
    point_1 = []
    point_2 = []
    for match_ in matches:
        p1 = kp1[match_.queryIdx].pt
        p2 = kp2[match_.trainIdx].pt
        point_1.append(p1)
        point_2.append(p2)
        stack = np.hstack((np.array(p1), np.array(p2)))

    return point_1, point_2
        #np.savetxt(store_path, stack)

        #cv_file.write('matches_points', stack)

    #matches = np.array(match, dtype=np.int8)
    #matches_ = cv2.convertScaleAbs(match)
    #print('Matches shape', matches_.shape)
    #cv_file.write('matches_points', matches)
    #cv_file.write('Int1', kp1)
    #cv_file.write('Int2', kp2)
    #points = np.hstack((kp1, kp2))
    #np.savetxt(store_path, points)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Camera calibration')
    parser.add_argument('--image_dir', type=str, required=True, help='image directory path')
    parser.add_argument('--save_file', type=str, required=True, help='YML file to save calibration matrices')
    args = parser.parse_args()

    img_path = '/Users/clementsiegrist/untitled7/img_'
    calib_path = '/Users/clementsiegrist/untitled7/img_/*.img'
    test_img = '/Users/clementsiegrist/untitled7/img_/2.png'
    store_path = '/Users/clementsiegrist/untitled7/img_/repro1.png'
    cam_params = '/Users/clementsiegrist/untitled7/o_gl_3D/3D_rec_params.txt'
    path_img1 = '/Users/clementsiegrist/untitled7/img_/0.png'
    path_img2 = '/Users/clementsiegrist/untitled7/img_/2.png'
    points_path = '/Users/clementsiegrist/untitled7/o_gl_3D/interest_points.txt'
    capture_dir = '/Users/clementsiegrist/untitled7/o_capture'
    o_l = '/Users/clementsiegrist/untitled7/o_load'

    img_size = 1280*720
    a_heigth, a_width = 3888, 2430

    retval, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, \
    stdDeviationsExtrinsics, perViewErrors = calibratextended_findchess(img_path)
    video_to_frames(capture_dir)
    ovx, fovy, Focale, PrincipalPoint, aspectRatio = cv2.calibrationMatrixValues(cameraMatrix, imageSize=(a_heigth, a_width),
                                                                                 apertureHeight=a_heigth, apertureWidth=a_width)
    #save_coefficients(retval, cameraMatrix, distCoeffs, rvecs, tvecs, cam_params)
    p1, p2 = matches_img(path_img1, path_img2, points_path)