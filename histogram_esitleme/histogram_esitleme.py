import cv2
import matplotlib.pyplot as plt

# Resmi oku (örnek resmi kendi dosya yolunuzla değiştirin)
resim_path = "resim.jpg"
resim = cv2.imread(resim_path, cv2.IMREAD_GRAYSCALE)

# Histogram eşitleme işlemi
eşitlenmiş_resim = cv2.equalizeHist(resim)

# Orjinal ve eşitlenmiş resmi göster
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(resim, cmap='gray')
plt.title("Orjinal Histogram")

plt.subplot(1, 2, 2)
plt.imshow(eşitlenmiş_resim, cmap='gray')
plt.title("Eşitlenmiş Histogram")

plt.show()
