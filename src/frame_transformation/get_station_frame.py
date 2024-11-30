import numpy as np
import argparse
from scipy.optimize import least_squares
import matplotlib.pyplot as plt


# Romain Englebert
# Photogrammetry
# 22/11/2024


# -------- DATA --------

# Index in dimension 0 of points_station1 and points_station2 corresponds to the same object point
# We assume frame 1 is (x, y, z, alpha, beta, gamma) = (0, 0, 0, 0, 0, 0)

stations = {
    0: [
    [0.77152035, 1.78727468, -0.12964014],
    [0.8412981, 1.61458396, 0.59375389],
    [0.93879107, 1.47697033, 1.2540493],
    [0.62201624, 1.59671564, 0.29616878],
    [0.70895871, 1.37744698, 1.28537401],
    [0.45469047, 1.55511604, 0.2602915],
    [0.46151274, 1.39093742, 0.92310513],
    [0.42963401, 1.48471895, 0.63239565],
    [0.44655382, 1.2576307, 1.56928471],
    [0.37115951, 1.65771616, -0.02861393],
    [0.21643835, 1.86816283, 1.11432404],
    [0.3487525, 1.80907213, 0.61725542],
    [0.26871337, 1.95117192, 1.29284851],
    [0.69480918, 1.87876116, 0.30359957],
    [0.68849436, 2.0318306, 0.96412857],
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
    [0, 0, 0]
    ],

    1 : [
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
    ],

    2 : [
    [1.5680693, 1.1370177, -0.12612186],
    [1.38541693, 1.13094786, 0.59880943],
    [1.22503779, 1.08294339, 1.25796861],
    [1.44719233, 1.33820028, 0.30148029],
    [1.21920277, 1.33820404, 1.29359168],
    [1.46825034, 1.51017184, 0.26599055],
    [1.31944366, 1.56370199, 0.93134768],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [1.74998866, 1.50904855, 0.62946684],
    [0, 0, 0],
    [1.67505118, 1.16910164, 0.30818664],
    [1.81608447, 1.12550819, 0.97132771],
    [1.75669032, 1.56600973, 0.60140972],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [1.58603498, 1.11003806, -0.02011382],
    [1.72152366, 1.05765083, 0.65181286],
    [1.91954088, 0.98759256, 1.57297958],
    [1.32817064, 1.08241489, 0.97223492],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
    ],

    3 : [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [1.43932324, 1.29959069, 0.63469665],
    [1.65938185, 1.27395343, 1.57849355],
    [1.26188233, 1.27753335, -0.02480618],
    [1.03813537, 1.17033099, 1.11998943],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [1.10374085, 1.23785672, 0.60096827],
    [0.97631707, 1.17169663, 1.27205656],
    [1.12400404, 1.48116356, 0.30368363],
    [0.91433231, 1.46515624, 1.30391705],
    [1.10636222, 1.70117367, 0.2686293],
    [0.96273247, 1.72787482, 0.93340009],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [1.38940278, 1.63466107, 0.61584666],
    [0, 0, 0],
    [1.36863262, 1.4026668, 0.30275335],
    [1.52641071, 1.38314042, 0.96806515],
    [1.25388704, 1.34478607, -0.12635949],
    ]
}

# ----------- Functions -----------


def plot(points_station_a):
    """
     Plot only the common measured points between a-b stations.
     For better visualization, place the z-axis downwards
    """

    fig_3d = plt.figure(figsize=(10, 7))
    ax3d = fig_3d.add_subplot(111, projection='3d')

    X, Y, Z = np.array(points_station_a).T[0], np.array(points_station_a).T[1], np.array(points_station_a).T[2]

    ax3d.scatter(X, Y, Z, label=f'Common measured points', s=50)
    ax3d.scatter(0, 0, 0, label=f'o.o', s=50)

    ax3d.set_title('3D Plot')
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ax3d.legend()
    plt.show()


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

    x, y, z = params[:3]  # Station 2 position
    angles = params[3:]   # Station 2 Orientation
    R = yaw_pitch_roll_matrix(angles)

    transformed_points = (R @ (points_station_a  - np.array([x, y, z])).T).T  # From frame 1 to frame 2
    return transformed_points


def residuals(params, points_station_a, points_station_b):
    """
    Loss function used in the least squares.
    :param params: Optimization parameters [x, y, z, yaw, pitch, roll]
    :param points_station_a: stations[1-4]
    :param points_station_b: stations[1-4]
    :return: The difference between our transformation a-b model based on optimization parameters and the actual b-points.
    """

    # Removing measurement not in common
    not_common = [i for i in range(len(points_station_a)) if points_station_a[i][0] == 0. or points_station_b[i][0] == 0.]

    for i, point_index in enumerate(not_common):
        del points_station_a[point_index-i]
        del points_station_b[point_index-i]

    if len(points_station_a) <= 2:
        raise Exception("You need at least two common measurements between each station")

    points_station_a = np.array(points_station_a)
    points_station_b = np.array(points_station_b)

    # Objective function for optimization
    loss = model_station_b(params, points_station_a) - points_station_b  # Difference between the model based on optimization parameters and actual points.
    return loss.ravel()


def calculate_transformation(points_station_a, points_station_b, initial_guess):
    """
    Calculate the transformation (rotation + translation) that maps points
        from Station a to Station b.
    :param points_station_a: (np.ndarray) Points in the first station's coordinate frame.
    :param points_station_b: (np.ndarray) Corresponding points in the second station's coordinate frame.
    :param initial_guess: (np.ndarray) Initial guess in the form [x, y, z, roll(x), pitch(y), yaw(z)] in m and rad.
    :return: Optimization parameters. It is concatenation of both following :
             - translation (np.ndarray): Translation vector [x, y, z].
             - rotation_angles (np.ndarray): Rotation angles [roll(x), pitch(y), yaw(z)] in radians.
    """
    return least_squares(residuals, initial_guess, args=(points_station_a, points_station_b))


def transform(translation, rotation, station):
    """
    Get the points measured in b, represented in the frame a. Thanks to the computed transformation between each frame.
    :param translation: [x, y, z]
    :param rotation: [yaw, pitch, roll]
    :param station: (int) Number of the station to transform in the new frame
    :return: Point measured in frame a, represented in frame b. (a in b)
    """
    params = [translation[0], translation[1], translation[2], rotation[0], rotation[1], rotation[2]]
    points_ainb = model_station_b(params, stations[station].copy())
    for i in range(len(points_ainb)):
        if stations[station][i] == [0, 0, 0]:
            points_ainb[i] = [0, 0, 0]
    return points_ainb


def std(station, points_ainb):
    """
    Compute distance-based empirical std of points resulting from the calculated transformation regarding the measurements in frame b.
    :param station: (int) 1-4
    :param points_ainb: Points measured in a represented in b resulting from the calculated transformation.
    :return: Standard deviation of common points between both stations. It will be higher for measurements from only 1 single station.
    """
    diff = [points_ainb[i] - stations[station][i] for i in range(len(points_ainb)) if points_ainb[i][0] != 0 and stations[station][i][0] != 0]
    distances = np.linalg.norm(diff, axis=1)
    return np.std(distances)

# -------- Main --------


def main():
    """
    Main function to parse user input and calculate the transformation.
    Use --help if you don't understand
    Execute python3 ./get_station_frame.py --fro 1 --to 0 --write 1 --show 1 in the terminal in the parents directory of this file.
    """
    parser = argparse.ArgumentParser(description="Calculate the transformation between two stations.")
    parser.add_argument("--fro", type=int, help="Index of the first station.", default=3)
    parser.add_argument("--to", type=int, help="Index of the second station.", default=0)
    parser.add_argument("--show", type=int, help="Plot ? (1 or 0)", default=0)
    parser.add_argument("--write", type=int, help="Write points measured from station a expressed in station b. (1 or 0)", default=1)

    args = parser.parse_args()

    # Retrieve the points for the specified stations
    station_a = args.fro
    station_b = args.to
    show = args.show
    write = args.write

    if station_a not in stations or station_b not in stations:
        print(f"Error: Station {station_a} or Station {station_b} does not exist.")
        return

    points_station_a = stations[station_a].copy()
    points_station_b = stations[station_b].copy()

    # Calculate the transformation
    initial_guess = [0, 0, 0, 0, 0, 0]  # TO CHOOSE
    result = calculate_transformation(points_station_a, points_station_b, initial_guess)
    x, y, z, yaw, pitch, roll = result.x
    translation = list((float(x), float(y), float(z)))
    rotation = np.degrees((yaw, pitch, roll))
    rotation = [float(round(angle, 4)) for angle in rotation]

    # Print the results
    print(f"Transformation from Station {station_a} to Station {station_b}:")
    print(f"Translation (x, y, z): {translation}")
    print(f"Rotation angles (yaw(z), pitch(y), roll(x)) in degrees: {rotation}")
    print("Residual error :", result.cost)

    if show == 1:
        plot(points_station_a)

    if write == 1:
        points_ainb = transform(translation, [yaw, pitch, roll], station_a)
        print(f"\n Points measured in {station_a} represented in {station_b}\n {points_ainb}")
        print()
        print(f"Std : {std(station_b, points_ainb)}m")


if __name__ == "__main__":
    main()