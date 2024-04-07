import numpy as np
from tkinter import filedialog
import tkinter as tk
import matplotlib.pyplot as plt
from pyransac3d import Plane


def read_point_cloud(file_path):
    with open(file_path, 'r') as file:
        point_cloud = []
        for line in file:
            point = line.strip().split(',')
            point_cloud.append([float(point[0]), float(point[1]), float(point[2])])
    return np.array(point_cloud)


def is_horizontal(normal_vector):
    # Sprawdzenie czy współrzędna z normalnego wektora płaszczyzny(normal_vector[2]) ma większą wartość bezwzględną
    # niż współrzędne x (normal_vector[0] i y (normal_vector[1]) wektora.
    # Jeśli warunek jest spełniony to oznacza że płaszczyzna jest pozioma.
    return np.abs(normal_vector[2]) > np.abs(normal_vector[0]) and np.abs(normal_vector[2]) > np.abs(normal_vector[1])


def is_vertical(normal_vector):
    # Sprawdzenie czy x (normal_vector[0]) ma większą wartość bezwzględną niż
    # y (normal_vector[1]) i z (normal_vector[2]) wektora.
    # Jeśli warunek jest spełniony to oznacza że płaszczyzna jest pionowa.
    return np.abs(normal_vector[0]) > np.abs(normal_vector[1]) and np.abs(normal_vector[0]) > np.abs(normal_vector[2])


def main():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    point_cloud = read_point_cloud(file_path)

    plane = Plane()
    best_plane, inliers = plane.fit(point_cloud, thresh=0.01, minPoints=3, maxIteration=1000)

    # Sprawdzenie czy stosunek liczby punktów inliers do całkowitej liczby punktów jest większy niż 80%
    # Jeśli jest większy to chmura reprezentuje płaszczyzne
    if len(inliers) / len(point_cloud) > 0.8:
        normal_vector = best_plane[:3]
        print(normal_vector)
        if is_horizontal(normal_vector):
            print("Płaszczyzna jest pozioma.")
        else:
            print("Płaszczyzna jest pionowa")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # x = point_cloud[:, 0]
        # y = point_cloud[:, 1]
        # z = point_cloud[:, 2]
        inlier_cloud = point_cloud[inliers]
        ax.scatter(inlier_cloud[:, 0], inlier_cloud[:, 1], inlier_cloud[:, 2], c='r', marker='o', label='Inliers')

        if np.all(point_cloud[:, 1] == 0):
            xx, zz = np.meshgrid(np.linspace(np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0]), 10),
                                 np.linspace(np.min(point_cloud[:, 2]), np.max(point_cloud[:, 2]), 10))
            yy = (-best_plane[3] - best_plane[0] * xx - best_plane[2] * zz) / best_plane[1]
        else:
            xx, yy = np.meshgrid(np.linspace(np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0]), 10),
                                 np.linspace(np.min(point_cloud[:, 1]), np.max(point_cloud[:, 1]), 10))
            zz = (-best_plane[3] - best_plane[0] * xx - best_plane[1] * yy) / best_plane[2]

        ax.plot_surface(xx, yy, zz, color='g', alpha=0.5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()

    else:
        print("Chmura nie jest płaszczyzną")


if __name__ == "__main__":
    main()
