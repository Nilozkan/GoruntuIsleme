import cv2
import numpy as np
import matplotlib.pyplot as plt

# Resmi oku (örnek resmi kendi dosya yolunuzla değiştirin)
resim_path = "resim.jpg"
resim = cv2.imread(resim_path)

# Gama düzeltme uygula
gamma = 1.5  # Gama değeri

gama_duzeltme = np.power(resim / 255.0, gamma) * 255.0
gama_duzeltme = gama_duzeltme.astype(np.uint8)

# Görselleştirme
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(resim, cv2.COLOR_BGR2RGB)), plt.title('Orijinal Resim')
plt.subplot(1, 2, 2), plt.imshow(cv2.cvtColor(gama_duzeltme, cv2.COLOR_BGR2RGB)), plt.title('Gama Düzeltme')

plt.show()
