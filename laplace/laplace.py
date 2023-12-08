import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('resim.jpg', 0)

laplacian = cv2.Laplacian(img, cv2.CV_64F)

plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray')
plt.title('İlk Resim'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])

plt.show()

gaussian = cv2.GaussianBlur(img, (5, 5), 0)

plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray')
plt.title('Orjinal Resim'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(gaussian, cmap='gray')
plt.title('Gaussian'), plt.xticks([]), plt.yticks([])

plt.show()