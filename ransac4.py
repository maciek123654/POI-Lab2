import csv
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
import tkinter as tk


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


# Dopasowanie płaszczyzny do chmury punktów przy użyciu algorytmu RANSAC
# points = lista punktów (z,y,z)
# iterations = liczba iteracji algorytmu
# threshold = próg odległości punktów od dopasowanej płaszczyzny
# min_inliers = liczba punktów, które muszą być uznane za inliers, aby uznać dopasowanie płaszczyzny
def ransac_plane_fitting(points, iterations=1000, threshold=1.0, min_inliers=50):
    best_plane = None
    best_inliers = []

    for _ in range(iterations):
        random_indices = np.random.choice(len(points), 3, replace=False)
        # Lista losowo wybranych punktów, bez powtórzeń
        sample_points = [points[i] for i in random_indices]

        _, _, _, vectorA, vectorB = compute_vectors(sample_points)  # Obliczanie vectorA i vectorB, pominięcie A, B, C (_)
        vectorUA = (vectorA) / (np.linalg.norm(vectorA))  # Wyznaczenie wektorów jednostkowych
        vectorUB = (vectorB) / (np.linalg.norm(vectorB))  # Wyznaczenie wektorów jednostkowych
        vectorUC = np.cross(vectorUA, vectorUB)  # Wyznaczenie wektora normalnego
        D = -(np.sum(np.multiply(vectorUC, sample_points[2])))  # Obliczanie równania płaszczyzny

        distances = np.array([distance_to_plane(point, (vectorUC[0], vectorUC[1], vectorUC[2], D)) for point in points])
        inliers = np.where(np.abs(distances) <= threshold)[0]

# Sprawdzenie czy grupa inlierów spełnia warunki uznania za najlepszą
# Jeśli liczba punktów w aktualnej iteracji jest większa od minimalnej oraz jest większa od największej poprzedniej
# to aktualna grupa staje się nową najlepszą grupą
        if len(inliers) > min_inliers and len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = (vectorUC[0], vectorUC[1], vectorUC[2], D)

    if best_plane is None:
        return None

    # Obliczamy współczynniki wektora normalnego do znalezionej płaszczyzny
    normal_vector = np.array([best_plane[0], best_plane[1], best_plane[2]])

    return normal_vector


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
print("Obliczanie odległości")

# Wizualizacja chmury punktów
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Pobierz współrzędne x, y, z punktów inliers i outliers
inliers_x = [points[i][0] for i in inliers]
inliers_y = [points[i][1] for i in inliers]
inliers_z = [points[i][2] for i in inliers]

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
    xx, zz = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(z_min, z_max, 10))
    yy = (D - A * xx - C * zz) / B
    ax.plot_surface(xx, yy, zz, alpha=0.5, color='green')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.legend()
plt.show()


def ransac_plane_fitting(points, iterations=1000, threshold=1.0, min_inliers=50):
    best_plane = None
    best_inliers = []

    for _ in range(iterations):
        random_indices = np.random.choice(len(points), 3, replace=False)
        sample_points = [points[i] for i in random_indices]

        _, _, _, vectorA, vectorB = compute_vectors(sample_points)
        vectorUA = (vectorA) / (np.linalg.norm(vectorA))
        vectorUB = (vectorB) / (np.linalg.norm(vectorB))
        vectorUC = np.cross(vectorUA, vectorUB)
        D = -(np.sum(np.multiply(vectorUC, sample_points[2])))

        distances = np.array([distance_to_plane(point, (vectorUC[0], vectorUC[1], vectorUC[2], D)) for point in points])
        inliers = np.where(np.abs(distances) <= threshold)[0]

        if len(inliers) > min_inliers and len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = (vectorUC[0], vectorUC[1], vectorUC[2], D)

    if best_plane is None:
        return None

    return best_plane


# Dopasowanie płaszczyzny za pomocą algorytmu RANSAC
normal_vector = ransac_plane_fitting(points)
# Średnia odległość punktów do płaszczyzny
distances = np.array([distance_to_plane(point, normal_vector) for point in points])
mean_distance = np.mean(distances)
print("Średnia odległość punktów do płaszczyzny:", mean_distance)
print("A: ", normal_vector[0], " B: ", normal_vector[1], " C: ", normal_vector[2])
if mean_distance == 0:
    # Określenie charakteru płaszczyzny
    if abs(normal_vector[0]) < 1e-3 and abs(normal_vector[1]) < 1e-3:
        print("Płaszczyzna jest pozioma.")
    elif abs(normal_vector[2]) < 1e-3:
        print("Płaszczyzna jest pionowa.")
    else:
        print("Chmura nie jest ani pionowa ani pozioma")
elif mean_distance != 0:
    print("Chmura nie jest płaszczyzną")