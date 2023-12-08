import cv2
import numpy as np


file_path = 'resim.jpg'
img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

def median_filter(img, size):
    result = np.zeros_like(img, dtype=np.uint8)
    img = np.pad(img, (size // 2, size // 2), mode='constant')

    rows, cols = img.shape

    for i in range(size // 2, rows - size // 2):
        for j in range(size // 2, cols - size // 2):
            neighbors = img[i - size // 2:i + size // 2 + 1, j - size // 2:j + size // 2 + 1].flatten()
            result[i - size // 2, j - size // 2] = np.median(neighbors)

    return result


def mean_filter(img, size):
    result = np.zeros_like(img, dtype=np.uint8)
    img = np.pad(img, (size // 2, size // 2), mode='constant')

    rows, cols = img.shape

    for i in range(size // 2, rows - size // 2):
        for j in range(size // 2, cols - size // 2):
            neighbors = img[i - size // 2:i + size // 2 + 1, j - size // 2:j + size // 2 + 1].flatten()
            result[i - size // 2, j - size // 2] = int(np.mean(neighbors))

    return result

if img is None:
    print(f"Resim yüklenemedi. Lütfen dosya yolunu kontrol edin: {file_path}")
    exit()

median_filtered_img = median_filter(img, size=3)
mean_filtered_img = mean_filter(img, size=3)

cv2.imshow('Tuz Biberli Resim', img)
cv2.imshow('Medyan Filtreli ', median_filtered_img)
cv2.imshow('Ortalama Filtreli ', mean_filtered_img)
cv2.waitKey(0)
cv2.destroyAllWindows()