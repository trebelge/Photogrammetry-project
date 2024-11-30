import numpy as np
from scipy.constants import point
from scipy.optimize import root
from scipy.optimize import least_squares
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from mpl_toolkits.mplot3d import Axes3D
from read_xml import get_image_points, get_object_points


path = 'markers.xml'

# ---------- Camera parameters ------------
# Nikon D3100

a = 23.1
b = 15.4
px_a = 4608
px_b = 3072

# ---------- Data initialization ----------


points = get_image_points(path)

CP = np.array([1, 2, 6, 9, 10, 13, 16, 20, 22, 24, 27, 28, 29, 30]) - 1
GCP = np.array([3, 4, 5, 7, 8, 11, 12, 14, 15, 17, 18, 19, 21, 23, 25, 26]) - 1

tp_img = points[:, CP, :]
gcp_img = points[:, GCP, :]


for i in range(len (tp_img)):
    for j in range(len(tp_img[0])):
        if tp_img[i][j][0] is not None:
            tp_img[i][j][0] *= a/px_a
            tp_img[i][j][1] *= a/px_a
    for j in range(len(gcp_img[0])):
        if gcp_img[i][j][0] is not None:
            gcp_img[i][j][0] *= a/px_a
            gcp_img[i][j][1] *= a/px_a

gcp_obj = get_object_points(path)[GCP, :] * 1000


def check(tp_img, gcp_img):
    # Check if the system is solvable and remove useless elements
    # If a tie point is in less than 2 images
    # If an image contains less than 1 GCP
    # If Redundancy is negative

    N = 0  # #elts diff. from () in tp_img
    u = len(tp_img)  # #images
    t = len(tp_img[0])  # #tp
    gcp = len(gcp_obj)

    #  Compute number of equations and check no tie point is in less than 2 images
    bad_tp = []
    for i in range(t):  # For each tp
        a = 0
        for j in range(u):  # For each image

            if tp_img[j][i][0] is not None:
                a += 1

        if a < 2:
            print("Tie point unsolvable")
            bad_tp.append(i)
        else:
            N += a

    #  Remove unsolvable tie points
    for i in bad_tp:
        t -= 1
        tp_img = [[val for j, val in enumerate(ln) if j != i] for ln in tp_img]

    # Check each img EO are solvable
    bad_img = []
    for i in range(u):  # For each image
        a = 0
        for j in range(len(gcp_img[0])):  # For each gcp

            if gcp_img[i][j][0] is not None:
                a += 1
        if a < 1:
            print("EO unsolvable")
            bad_img.append(i)
        else:
            N += a

    # Remove unsolvable images
    for i in bad_img:
        u -= 1
        tp_img = tp_img[:i] + tp_img[i + 1:]
        gcp_img = gcp_img[:i] + gcp_img[i + 1:]


    # Check if system solvable
    r = 2*N - 6*u - 3*t
    print("N = ", N, ", u = ", u, ", t = ", t)
    print("Redundancy = ", r)

    if r < 0:
        print("Not redundant...")
        return -1, -1, -1, -1, -1, -1

    return N, u, t, gcp, tp_img, gcp_img



# ---------- Calibration ----------


def calibration():
    f = 24  # focal length
    x0, y0 = a/2, b/2  # PP offset
    return x0, y0, f

def dx(x):
    return 0

def dy(y):
    return 0


# ---------- System construction and solving ----------


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


def collinearity(yaw, pitch, roll, XL, YL, ZL, XA, YA, ZA, xy):
    # Return partial collinearity equation
    # xy = 0 : x-equation
    # xy = 1 : y-equation

    R = yaw_pitch_roll_matrix([yaw, pitch, roll])
    return (R[xy, 0] * (XA-XL) + R[xy, 1] * (YA-YL) + R[xy, 2] * (ZA-ZL)) / (R[2, 0] * (XA-XL) + R[2, 1] * (YA-YL) + R[2, 2] * (ZA-ZL))


def system(params, tp_img, gcp_img, u, t, gcp, IO):
    # Return non-linear system in the form F(x) = 0
    # params contains adjustment parameters : each image angles, each image PC position, each tie point object coordinates

    # unzip parameters
    M_u = params[:u * 3].reshape(u, 3)  # Rotation angles for each image
    XL_u = params[u * 3:u * 6].reshape(u, 3)  # Position for each image
    XA_t = params[u * 6:].reshape(t, 3)  # Tie points coordinates
    x0, y0, f = IO

    F = []

    # Add 2 equations for each tie point
    for i in range(t):  # For each tie point
        for j in range(u):  # For each image
            if tp_img[j][i][0] is not None:
                xa, ya = tp_img[j][i]

                F.append(xa - x0 + dx(xa) + f * collinearity(M_u[j][0], M_u[j][1], M_u[j][2], XL_u[j][0], XL_u[j][1], XL_u[j][2], XA_t[i][0], XA_t[i][1], XA_t[i][2], 0))
                F.append(ya - y0 + dy(ya) + f * collinearity(M_u[j][0], M_u[j][1], M_u[j][2], XL_u[j][0], XL_u[j][1], XL_u[j][2], XA_t[i][0], XA_t[i][1], XA_t[i][2], 1))

    # Add 2 equations for each GCP with known GCP object corrdinate
    for i in range(gcp):  # Pour chaque tie point
        for j in range(u):  # Pour chaque image

            if gcp_img[j][i][0] is not None:
                xa, ya = gcp_img[j][i]

                F.append(xa - x0 + dx(xa) + f * collinearity(M_u[j][0], M_u[j][1], M_u[j][2], XL_u[j][0], XL_u[j][1],
                                                             XL_u[j][2], gcp_obj[i][0], gcp_obj[i][1], gcp_obj[i][2], 0))
                F.append(ya - y0 + dy(ya) + f * collinearity(M_u[j][0], M_u[j][1], M_u[j][2], XL_u[j][0], XL_u[j][1],
                                                             XL_u[j][2], gcp_obj[i][0], gcp_obj[i][1], gcp_obj[i][2], 1))

    return F


def bundle_block_adjustment(tp_img, gcp_img, u, t, gcp):
    # Compute EO and tie point object coordinates

    IO = calibration()

    # Initial guesses
    M_u = np.zeros((u, 3))  # Orientation each image
    XL_u = np.ones((u, 3))  # Object coordinates of PC of each image
    XA_t = np.ones((t, 3))  # Object coordinates of tie points

    XL_u[:, 0] = 1000
    XL_u[:, 1] = 1000
    XL_u[:, 2] = 500
    initial_guess = np.hstack((M_u.flatten(), XL_u.flatten(), XA_t.flatten()))  # Format adjustment parameters
    # Solve the system
    solution = least_squares(lambda param: system(param, tp_img, gcp_img, u, t, gcp, IO), initial_guess, gtol=1e-6, ftol=1e-6)

    return solution.x


# ---------- Plot ----------

def plot(solution, show):
    # Plot each image tie points
    # Plot computed tie point 3D object coordinates

    # Format adjustment parameters
    M_u = solution[:u * 3].reshape(u, 3)        # Rotation angles for each image
    XL_u = solution[u * 3:u * 6].reshape(u, 3)  # Position for each image
    XA_t = solution[u * 6:].reshape(t, 3)       # Tie points coordinates

    for i in range(u):
        print("Camera ", i, " - Orientation : ", np.degrees(M_u[i]), " - Position : ", XL_u[i]/1000)

    if show == 1:
        image_paths = ['/home/trebelge/Cours NTU/Photogrammetry/Project/Code/img/img_1.JPG', '/home/trebelge/Cours NTU/Photogrammetry/Project/Code/img/img_2.JPG', '/home/trebelge/Cours NTU/Photogrammetry/Project/Code/img/img_3.JPG', '/home/trebelge/Cours NTU/Photogrammetry/Project/Code/img/img_4.JPG']

        # Plot tie points on each image
        fig, axes = plt.subplots(1, u, figsize=(15, 10))

        # Iterate over each image and plot it with the tie points
        for i in range(u):
            ax = axes[i]
            img = mpimg.imread(image_paths[i])
            ax.imshow(img)

            # Plot the tie points for the current image
            for tp in tp_img[i]:
                if tp[0] is not None:
                    x, y = tp
                    ax.scatter(x*px_a/a, y*px_a/a, color='red', s=40, marker='x', label='Tie point')  # Plot the tie point (in red)

            # Plot the GCP's points for the current image
            for gcp in gcp_img[i]:
                if gcp[0] is not None:
                    x, y = gcp
                    ax.scatter(x * px_a / a, y * px_a / a, color='blue', s=40,
                               marker='x', label='GCP')  # Plot the tie point (in red)

            ax.set_title(f'Image {i + 1}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')

        plt.legend()
        plt.tight_layout()
        plt.show()


    # Plot point in 3D space
    fig_3d = plt.figure(figsize=(10, 7))
    ax3d = fig_3d.add_subplot(111, projection='3d')

    X, Y, Z = XA_t[:, 0], XA_t[:, 1], XA_t[:, 2]
    ax3d.scatter(X, Y, Z, label=f'Tie point', s=50)

    X, Y, Z = gcp_obj[:, 0], gcp_obj[:, 1], gcp_obj[:, 2]
    ax3d.scatter(X, Y, Z, label=f'GCPs', s=50)

    ax3d.set_title('3D Plot')
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ax3d.legend()
    plt.show()


# ---------- main -----------

N, u, t, gcp, tp_img, gcp_img = check(tp_img, gcp_img)
if N != -1:
    sol = bundle_block_adjustment(tp_img, gcp_img, u, t, gcp)
    plot(sol, 0)