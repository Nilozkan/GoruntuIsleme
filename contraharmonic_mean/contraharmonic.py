import cv2
import numpy as np
import matplotlib.pyplot as plt

def contraharmonic_mean_filter(image, kernel_size, Q):
    result = cv2.filter2D(image, -1, np.power(image, Q+1) / np.power(image, Q), borderType=cv2.BORDER_CONSTANT)
    return result

# Resmi oku (örnek resmi kendi dosya yolunuzla değiştirin)
resim_path = "resim.jpg"
resim = cv2.imread(resim_path, cv2.IMREAD_GRAYSCALE)

# Contraharmonic Mean filtresi uygula
kernel_size = 3  # Filtre boyutu
Q = 1  # Q parametresi

contraharmonic_filtered = contraharmonic_mean_filter(resim, kernel_size, Q)

# Görselleştirme
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1), plt.imshow(resim), plt.title('Orjinal Resim')
plt.subplot(1, 3, 2), plt.imshow(contraharmonic_filtered ), plt.title('Contraharmonic Mean Filtreleme')

plt.show()
