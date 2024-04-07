import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from tkinter import Tk
from tkinter.filedialog import askopenfilename


def load_xyz_file(file_path):
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            coordinates = line.strip().split(',')
            points.append([float(coord) for coord in coordinates])
    return np.array(points)


def choose_file():
    Tk().withdraw()  # Ukrycie głównego okna
    file_path = askopenfilename(title="Wybierz plik XYZ", filetypes=[("Pliki XYZ", "*.xyz")])
    return file_path


# Normalizacja danych
# Skalowanie danych aby miały wartość średnią = 0 i wariancję = 1
def normalize_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)


# Ustawienie parametrów algorytmu DBSCAN
def set_dbscan_parameters():
    eps = 0.3  # Promień sąsiedztwa punktów
    min_samples = 10  # Minimalna liczba punktów w sąsiedztwie
    return eps, min_samples


def main():
    file_path = choose_file()
    if not file_path:
        print("Nie wybrano pliku.")
        return

    # Wczytanie danych
    data = load_xyz_file(file_path)

    # Normalizacja danych
    data_normalized = normalize_data(data)

    # Ustawienie parametrów algorytmu DBSCAN
    eps, min_samples = set_dbscan_parameters()

    # Utworzenie instancji algorytmu DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    # Uczenie modelu
    dbscan.fit(data_normalized)

    # Etykiety klastrów (-1 oznacza punkty odstające)
    labels = dbscan.labels_

    # Liczba klastrów włączając w to punkty odstające
    # set(label) - tworzy zbiór unikalnych etykiet klastrów
    # Sprawdzenie czy etykieta odstająca (-1) istnieje w zbiorze etykiet.
    # Jeśli tak, to do ogólnej liczby klastrów jest dodawane 1 aby uzględnić punkty odstające
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Liczba znalezionych klastrów: {n_clusters}")

    # Tworzenie wykresu 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Wyświetlenie punktów na wykresie
    for label in set(labels):
        if label == -1:
            color = 'k'  # Czarny dla punktów odstających
        else:
            color = plt.cm.jet(label / n_clusters)  # Kolorowanie klastrów

        # Wyodrębnienie punktów należących do danego klastra
        cluster_points = data[labels == label]

        # Wyświetlenie punktów na wykresie
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], color=color, s=20)

    # Ustawienie etykiet osi
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


if __name__ == "__main__":
    main()
