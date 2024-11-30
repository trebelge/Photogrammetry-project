import numpy as np
import pandas as pd

# 1->3 + 3->0
# [[x, y, z, yaw, pitch, roll], ...]
transfos = [[2.6731477715726246, -0.038883774318427954, -0.043027931685839786, 79.1701, -0.0721, 0.4585], [2.8177869926306376, 0.5911232677815503, 0.009485646122286338, 78.8048, -0.134, -0.0699]]
points =    [
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1.66335791, 1.27067388, 0.56844778],
            [1.70336184, 1.12470115, 1.23555726],
            [1.43013292, 1.33773561, 0.27150001],
            [1.40362294, 1.11811516, 1.26746603],
            [1.20876295, 1.3628459, 0.23688013],
            [1.15571453, 1.22289369, 0.90174045],
            [1.21127237, 1.4590278, -0.0516343],
            [1.157833, 1.31543636, 0.62042801],
            [1.08108681, 1.11069278, 1.53865813],
            [1.1911451, 1.72755891, 0.93920955],
            [1.32824526, 1.63298147, 0.58223023],
            [1.25681447, 1.82323647, 1.26440744],
            [1.55319327, 1.57135622, 0.27112049],
            [1.60042428, 1.7243393, 0.93341153],
            [1.58875749, 1.44450395, -0.15705723]
            ]


def yaw_pitch_roll_matrix(angle):
    """
    Generates a 3x3 rotation matrix from Yaw, Pitch, and Roll angles.
    - Yaw: rotation around the Z-axis
    - Pitch: rotation around the Y-axis
    - Roll: rotation around the X-axis
    :param angle: (yaw, pitch, roll)
    :return: Rotation matrix
    """
    yaw, pitch, roll = angle
    c_yaw, s_yaw = np.cos(yaw), np.sin(yaw)
    c_pitch, s_pitch = np.cos(pitch), np.sin(pitch)
    c_roll, s_roll = np.cos(roll), np.sin(roll)

    R_z = np.array([
        [c_yaw, s_yaw, 0],
        [-s_yaw, c_yaw, 0],
        [0, 0, 1]
    ])

    R_y = np.array([
        [c_pitch, 0, -s_pitch],
        [0, 1, 0],
        [s_pitch, 0, c_pitch]
    ])

    R_x = np.array([
        [1, 0, 0],
        [0, c_roll, s_roll],
        [0, -s_roll, c_roll]
    ])

    R = R_z @ R_y @ R_x
    return R


def model_station_b(params, points_station_a):
    """
    Map points from frame a to frame b
    :param params: Optimization parameters
    :param points_station_a: Points from original frame
    :return: Predicted points in frame b
    """

    x, y, z = params[:3]
    angles = np.radians(params[3:])
    R = yaw_pitch_roll_matrix(angles)

    transformed_points = (R @ (points_station_a  - np.array([x, y, z])).T).T  # From frame 1 to frame 2
    return transformed_points


result = points
# Compose transformations
for i in range(len(transfos)):
    result = model_station_b(transfos[i], result)

df = pd.DataFrame(result)
df.to_excel("output.xlsx", index=False)
print("Export successful in 'output.xlsx'")

