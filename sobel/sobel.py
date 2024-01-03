import cv2
import matplotlib.pyplot as plt
import numpy as np

# Resmi oku (örnek resmi kendi dosya yolunuzla değiştirin)
resim_path = "resim.jpg"
resim = cv2.imread(resim_path, cv2.IMREAD_GRAYSCALE)

# Sobel filtresi uygula
sobel_x = cv2.Sobel(resim, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(resim, cv2.CV_64F, 0, 1, ksize=3)

# Kenar büyüklüğünü hesapla
kenar_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

# Kenar yöneğini hesapla
kenar_yonu = np.arctan2(sobel_y, sobel_x)

# Görselleştirme
plt.figure(figsize=(12, 6))

plt.subplot(2, 3, 1), plt.imshow(resim, cmap='gray'), plt.title('Orjinal')
plt.subplot(2, 3, 2), plt.imshow(sobel_x, cmap='gray'), plt.title('Sobel X')
plt.subplot(2, 3, 3), plt.imshow(sobel_y, cmap='gray'), plt.title('Sobel Y')
plt.subplot(2, 3, 4), plt.imshow(kenar_magnitude, cmap='gray'), plt.title('Kenar Büyüklüğü')
plt.subplot(2, 3, 5), plt.imshow(kenar_yonu, cmap='gray'), plt.title('Kenar Yönü')

plt.show()
