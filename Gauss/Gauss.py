import cv2
import numpy as np
from matplotlib import pyplot as plt

# Resmi yükle
img = cv2.imread('resim.jpg')  # 'ornek_resim.jpg' yerine kendi resminizin adını kullanın
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Resmi RGB formatına dönüştür

# 5x5 Gauss filtresi oluştur
gaussian_kernel = cv2.getGaussianKernel(5, 1)  # Standart sapması 1 olan 5x5 Gauss filtresi

# 2 boyutlu Gauss filtresini oluştur
gaussian_kernel_2d = np.outer(gaussian_kernel, gaussian_kernel.T)

# Görüntüye Gauss filtresini uygula
filtered_img = cv2.filter2D(img, -1, gaussian_kernel_2d)

# Sonuçları gösterme
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Orijinal Resim')

plt.subplot(1, 2, 2)
plt.imshow(filtered_img)
plt.title('5x5 Gauss Filtresi (σ=1)')

plt.tight_layout()
plt.show()