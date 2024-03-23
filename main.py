import csv
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog
from sklearn.cluster import KMeans


def ransac_flat():
    root = tk.Tk()
    root.withdraw()

    points = []
    file = filedialog.askopenfilename()

    with open(file, newline='') as csv_file:
        csvreader = csv.reader(csv_file, delimiter=',')
        for row in csvreader:
            points.append([float(row[0]), float(row[1]), float(row[2])])

    points = np.array(points)

    # Algorytm k-means
    k = 3  # Liczba klastrów
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(points)
    klaster_center = kmeans.cluster_centers_  # Pobranie centrum każdego klastra
    klaster = kmeans.labels_  # Przypisanie klastra dla każdego punktu

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = ['r', 'g', 'b']
    for i in range(k):
        klaster_ = points[klaster == i]
        ax.scatter(klaster_[:, 0], klaster_[:, 1], klaster_[:, 2], c=colors[i], label=f'Klaster {i + 1}')
        ax.scatter(klaster_center[i, 0], klaster_center[i, 1], klaster_center[i, 2], c='black', marker='x',
                   label=f'Centrum klastra {i + 1}')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()


ransac_flat()
