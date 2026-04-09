import cv2
import numpy as np
import math


from ament_index_python.packages import get_package_share_directory
import os

def load_calibration():
    pkg_path = get_package_share_directory('navigation_pkg')

    mtx = np.loadtxt(os.path.join(pkg_path, 'config', 'camera_matrix_real.txt'), delimiter=",")
    dist = np.loadtxt(os.path.join(pkg_path, 'config', 'distortion_coefficients real.txt'), delimiter=",")

    return mtx, dist


# # Load camera calibration
# def load_calibration():
#     mtx = np.loadtxt("camera_matrix.txt", delimiter=",")
#     dist = np.loadtxt("distortion_coefficients.txt", delimiter=",")
#     return mtx, dist

# HSV colour ranges
COLOR_RANGES = {
    "red": [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (180, 255, 255))],
    "green": [((40, 50, 50), (80, 255, 255))],
    "blue": [((100, 150, 50), (140, 255, 255))]
}

def detect_balls_morphology(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detections = []

    for color, ranges in COLOR_RANGES.items():
        mask = None
        for lower, upper in ranges:
            m = cv2.inRange(hsv, np.array(lower), np.array(upper))
            mask = m if mask is None else cv2.bitwise_or(mask, m)

        # Morphology
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            continue

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        u = (x_min + x_max) // 2
        v = (y_min + y_max) // 2
        area = cv2.countNonZero(mask)
        r = int(math.sqrt(area / np.pi))  # approximate radius

        detections.append((color, u, v, r))

    return detections

def compute_3D(u, v, r, diameter_mm, mtx):
    fx, fy = mtx[0,0], mtx[1,1]
    cx, cy = mtx[0,2], mtx[1,2]
    # Distance formula using known diameter
    Z = (684.25316 * diameter_mm) / (2*r)
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return X, Y, Z

def capture_and_compute(ball_diameter_mm, frame):
    """
    Captures one frame from camera, detects balls, computes 3D coordinates.
    
    Returns:
        List of dicts: [{'color': str, 'X': float, 'Y': float, 'Z': float}, ...]
    """
    mtx, dist = load_calibration()

    frame = cv2.undistort(frame, mtx, dist)
    detections = detect_balls_morphology(frame)

    min_z = float('inf')
    best_detection = None

    for color, u, v, r in detections:
        X, Y, Z = compute_3D(u, v, r, ball_diameter_mm, mtx)
        
        # Check if the current Z is the smallest we've seen
        if Z < min_z:
            min_z = Z
            best_detection = (color, X, Y, Z)

    return best_detection

# Example usage:
if __name__ == "__main__":
    ball_diameter_mm = 79  # Set your real ball diameter
    ball_info = capture_and_compute(ball_diameter_mm)
    print(ball_info)
