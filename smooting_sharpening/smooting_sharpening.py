import cv2
import matplotlib.pyplot as plt

# Resmi oku (örnek resmi kendi dosya yolunuzla değiştirin)
resim_path = "resim.jpg"
resim = cv2.imread(resim_path, cv2.IMREAD_GRAYSCALE)

# Smoothing (Gaussian Blur)
smoothed_image = cv2.GaussianBlur(resim, (5, 5), 0)

# Sharpening (Laplacian)
laplacian = cv2.Laplacian(resim, cv2.CV_64F)
sharpened_image = resim - laplacian

# Görselleştirme
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1), plt.imshow(resim, cmap='gray'), plt.title('Orjinal')
plt.subplot(1, 3, 2), plt.imshow(smoothed_image, cmap='gray'), plt.title('Smoothing (Gaussian Blur)')
plt.subplot(1, 3, 3), plt.imshow(sharpened_image, cmap='gray'), plt.title('Sharpening (Laplacian)')

plt.show()
