import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from PIL import Image
import matplotlib.pyplot as plt

# Resmi aç
resim_path = "resim.jpg"
resim = Image.open(resim_path)

# Resmi düz bir diziye çevir
duz_veri = np.array(resim).reshape((-1, 3))

# Normalizasyon
normalizer = MinMaxScaler()
normalized_data = normalizer.fit_transform(duz_veri)

# Normalizasyon sonucunu yeniden resim boyutuna çevir
normalized_image_data = normalized_data.reshape(np.array(resim).shape)

# Standardizasyon
scaler = StandardScaler()
standardized_data = scaler.fit_transform(duz_veri)

# Standardizasyon sonucunu yeniden resim boyutuna çevir
standardized_image_data = standardized_data.reshape(np.array(resim).shape)

# Görsel çıktı için resimleri yan yana göster
plt.figure(figsize=(15, 5))

# Orjinal resim
plt.subplot(1, 3, 1)
plt.imshow(resim)
plt.title("Orjinal Resim")

# Normalizasyon sonucu
plt.subplot(1, 3, 2)
plt.imshow(normalized_image_data.astype(np.uint8))
plt.title("Normalizasyon Sonucu")

# Standardizasyon sonucu
plt.subplot(1, 3, 3)
plt.imshow(standardized_image_data.astype(np.uint8))
plt.title("Standardizasyon Sonucu")

plt.show()

