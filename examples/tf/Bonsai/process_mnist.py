# -*- coding: utf-8 -*-
"""
Beispiel zur Verarbeitung des MNIST-Datensatzes im ubyte-Format.
Die Funktion processMinst liest die Bild- und Labeldateien ein,
kombiniert die Labels (erste Spalte) und die Bilddaten (Pixelwerte) zu einer Matrix
und speichert das Ergebnis als .npy und .txt Dateien.
"""

import os
import numpy as np

def load_mnist_images(file_path):
    """Liest die Bilddatei im IDX-Format ein und gibt ein numpy-Array zurück.
    
    Das Format der Datei:
      - 4-Byte-Magic-Number
      - 4-Byte: Anzahl der Bilder
      - 4-Byte: Anzahl der Zeilen
      - 4-Byte: Anzahl der Spalten
      - Anschließend folgen die Pixelwerte (unsigned byte).
    """
    with open(file_path, 'rb') as f:
        # Lese den Header (16 Byte)
        header = np.frombuffer(f.read(16), dtype='>i4')
        magic, num_images, rows, cols = header
        # Lese alle Pixelwerte
        images = np.frombuffer(f.read(), dtype=np.uint8)
        # Forme die Bilder um; jedes Bild als Vektor (Zeilen * Spalten)
        images = images.reshape(num_images, rows * cols)
    return images

def load_mnist_labels(file_path):
    """Liest die Label-Datei im IDX-Format ein und gibt ein numpy-Array zurück.
    
    Das Format der Datei:
      - 4-Byte-Magic-Number
      - 4-Byte: Anzahl der Labels
      - Anschließend folgen die Labels (unsigned byte).
    """
    with open(file_path, 'rb') as f:
        # Lese den Header (8 Byte)
        header = np.frombuffer(f.read(8), dtype='>i4')
        magic, num_labels = header
        # Lese die Labels
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def processMinst(workingDir, downloadDir):
    """
    Verarbeitet den MNIST-Datensatz (im ubyte/IDX-Format).
    
    Erwartet im downloadDir folgende Dateien:
      - train-images-idx3-ubyte
      - train-labels-idx1-ubyte
      - t10k-images-idx3-ubyte
      - t10k-labels-idx1-ubyte
      
    Es werden zwei Dateien erzeugt:
      - train.npy und test.npy (als NumPy-Arrays, erste Spalte = Label, Rest = Pixelwerte)
      - train.txt und test.txt (Textformat, Spalten durch Leerzeichen getrennt)
    """
    # Erzeuge den absoluten Pfad
    path = os.path.abspath(os.path.join(workingDir, downloadDir))
    
    # Definiere die Dateipfade
    train_images_file = os.path.join(path, 'train-images-idx3-ubyte')
    train_labels_file = os.path.join(path, 'train-labels-idx1-ubyte')
    test_images_file  = os.path.join(path, 't10k-images-idx3-ubyte')
    test_labels_file  = os.path.join(path, 't10k-labels-idx1-ubyte')
    
    # Überprüfe, ob die Dateien vorhanden sind
    for file in [train_images_file, train_labels_file, test_images_file, test_labels_file]:
        assert os.path.isfile(file), f"Datei nicht gefunden: {file}"
    
    # Lade die Trainingsdaten
    train_images = load_mnist_images(train_images_file)
    train_labels = load_mnist_labels(train_labels_file)
    # Kombiniere die Labels und Bilder: erste Spalte = Label, Rest = Bilddaten
    train_data = np.column_stack((train_labels, train_images))
    
    # Lade die Testdaten
    test_images = load_mnist_images(test_images_file)
    test_labels = load_mnist_labels(test_labels_file)
    test_data = np.column_stack((test_labels, test_images))
    
    # Optional: Falls sich die Labels nicht bei 0 beginnen, können wir sie anpassen.
    # (Für MNIST sind die Labels bereits 0 bis 9.)
    # z.B.: 
    # train_data[:, 0] = train_data[:, 0] - train_data[0, 0]
    # test_data[:, 0]  = test_data[:, 0] - test_data[0, 0]
    
    # Speichere die Daten als .npy
    np.save(os.path.join(workingDir, 'train.npy'), train_data)
    np.save(os.path.join(workingDir, 'test.npy'), test_data)
    
    # Speichere die Daten zusätzlich als .txt
    np.savetxt(os.path.join(workingDir, 'train.txt'), train_data, fmt='%d')
    np.savetxt(os.path.join(workingDir, 'test.txt'), test_data, fmt='%d')
    
    print("MNIST-Daten wurden erfolgreich verarbeitet und gespeichert.")

if __name__ == '__main__':
    # Konfiguration: Arbeits- und Download-Verzeichnis
    workingDir = '/home/jschosto/EdgeML/examples/tf/Bonsai/MNIST-10'
    downloadDir = '/home/jschosto/EdgeML/MINST'  # Stelle sicher, dass hier ein Ordner mit den MNIST-Dateien existiert.
    
    print("Verarbeite MNIST-Daten...")
    processMinst(workingDir, downloadDir)
    print("Fertig.")