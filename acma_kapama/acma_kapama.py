import cv2
import numpy as np
import matplotlib.pyplot as plt

# Resmi oku (örnek resmi kendi dosya yolunuzla değiştirin)
resim_path = "resim.jpg"
resim = cv2.imread(resim_path, cv2.IMREAD_GRAYSCALE)

# Morfolojik işlem için kernel oluştur
kernel = np.ones((5, 5), np.uint8)

# Açma işlemi (erosion followed by dilation)
opening = cv2.morphologyEx(resim, cv2.MORPH_OPEN, kernel)

# Kapama işlemi (dilation followed by erosion)
closing = cv2.morphologyEx(resim, cv2.MORPH_CLOSE, kernel)

# Görselleştirme
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1), plt.imshow(resim), plt.title('Orjinal Resim')
plt.subplot(1, 3, 2), plt.imshow(opening), plt.title('Açma İşlemi')
plt.subplot(1, 3, 3), plt.imshow(closing), plt.title('Kapama İşlemi')

plt.show()
