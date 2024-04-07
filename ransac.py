import csv
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
import tkinter as tk

# Algorytm RANSAC


def conf_reader(file_path):
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for x, y, z in reader:
            yield float(x), float(y), float(z)


# Przeliczenie odległości punktu od płaszczyzny
def distance_to_plane(point, plane_params):
    A, B, C, D = plane_params
    x, y, z = point
    return abs(A * x + B * y + C * z + D) / np.sqrt(A ** 2 + B ** 2 + C ** 2)


# Wybranie trzech losowych punktów z chmury punktów, obliczenie wektora A i B na ich podstawie
def compute_vectors(points):
    # Wylosuj trzy różne indeksy punktów
    random_in_points = np.random.choice(len(points), 3, replace=False)

    # Wybierz współrzędne dla punktów A, B i C
    coordinates_A = points[random_in_points[0]]
    coordinates_B = points[random_in_points[1]]
    coordinates_C = points[random_in_points[2]]

    # Utwórz punkty A, B i C
    point_A = np.array(coordinates_A)
    point_B = np.array(coordinates_B)
    point_C = np.array(coordinates_C)

    vectorA = point_A - point_C
    vectorB = point_B - point_C

    # Sprawdzenie, czy wektory nie są zerowe
    if np.all(vectorA == 0):
        vectorA = np.array([0, 0, 1])  # Jeśli vectorA jest zerowy, ustawia go na [0, 0, 1]

    if np.all(vectorB == 0):
        vectorB = np.array([0, 0, 1])  # Jeśli vectorB jest zerowy, ustawia go na [0, 0, 1]

    return point_A, point_B, point_C, vectorA, vectorB


def open_file_dialog():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path

# Próg odległości od płaszczyzny, powyżej którego punkty są uznawane za odstające
threshold = 1.0

# Wybierz plik CSV
file_path = open_file_dialog()

# Wybierz wszystkie punkty z pliku CSV
points = list(conf_reader(file_path))

# Utwórz punkty A, B, C oraz wektory A, B
point_A, point_B, point_C, vectorA, vectorB = compute_vectors(points)

vectorUA = (vectorA) / (np.linalg.norm(vectorA))
vectorUB = (vectorB) / (np.linalg.norm(vectorB))
vectorUC = np.cross(vectorUA, vectorUB)

D = -(np.sum(np.multiply(vectorUC, point_C)))

# Obliczanie odległości każdego punktu od płaszczyzny
distances = np.array([distance_to_plane(point, (vectorUC[0], vectorUC[1], vectorUC[2], D)) for point in points])
# Wybranie dopasowanych punktów do płaszczyzny na podstawie ustalonego progu odległości
inliers = np.where(np.abs(distances) <= threshold)[0]
# Wybranie odstających punktów
outliers = np.where(np.abs(distances) > threshold)[0]

print("Liczba inliers:", len(inliers))
print("Liczba outliers:", len(outliers))

# Wizualizacja chmury punktów
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Pobierz współrzędne x, y, z punktów inliers i outliers
# Listy punktów dospasowanych
inliers_x = [points[i][0] for i in inliers]
inliers_y = [points[i][1] for i in inliers]
inliers_z = [points[i][2] for i in inliers]

# Listy punktów niedopasowanych
outliers_x = [points[i][0] for i in outliers]
outliers_y = [points[i][1] for i in outliers]
outliers_z = [points[i][2] for i in outliers]

# Rysowanie punktów
ax.scatter(inliers_x, inliers_y, inliers_z, c='b', marker='o', label='Inliers')
ax.scatter(outliers_x, outliers_y, outliers_z, c='r', marker='o', label='Outliers')

# Rysowanie płaszczyzny obliczonej
x_min, x_max = np.min(inliers_x), np.max(inliers_x)
y_min, y_max = np.min(inliers_y), np.max(inliers_y)
z_min, z_max = np.min(inliers_z), np.max(inliers_z)
# Generowanie siatki punktów w przestrzeni 2D
# np.linspace(z_min, x_max, 10) Generuje 10 równo rozłożonych wartości w przedziale od x_min do z_max
# np.meshgrid tworzy 2 macierze, jedną dla X, drugą dla Y
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))

A = vectorUC[0]
B = vectorUC[1]
C = vectorUC[2]

# Zwrot pierwszego wiersza
yyy = yy[0]
yyyy = yyy[0]

# Rysowanie płaszczyzny
if yyyy != 0:
    # Siatka jest niepionowa
    zz = (-A * xx - B * yy + D) / C
    ax.plot_surface(xx, yy, zz, alpha=0.5, color='green')
else:
    # Siatka jest pionowa, obliczanie yy na podstawie xx i zz
    xx, zz = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(z_min, z_max, 10))
    yy = (D - A * xx - C * zz) / B
    ax.plot_surface(xx, yy, zz, alpha=0.5, color='green')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.legend()
plt.show()
