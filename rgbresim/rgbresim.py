import numpy as np
import cv2


def initialize_centers(points, k):
    centers = points.copy()
    np.random.shuffle(centers)
    return centers[:k]


def closest_center(point, centers):
    return np.argmin(np.linalg.norm(centers - point, axis=1))


def kmeans(data, k, max_iters=100):
    centers = initialize_centers(data, k)
    converged = False
    iterations = 0

    while (not converged) and (iterations < max_iters):
        new_centers = np.zeros_like(centers)
        counts = np.zeros(k)

        for point in data:
            closest = closest_center(point, centers)
            new_centers[closest] += point
            counts[closest] += 1

        new_centers = new_centers / counts[:, None]

        if np.allclose(new_centers, centers):
            converged = True
        else:
            centers = new_centers

        iterations += 1

    return centers


img = cv2.imread('resim.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

pixels = img.reshape((-1, 3))

k = 4
centers = kmeans(pixels.astype(np.float64), k)


for i, point in enumerate(pixels):
    closest = closest_center(point, centers)
    pixels[i] = centers[closest]


segmented_img = pixels.reshape(img.shape)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Orijinal Resim')

plt.subplot(1, 2, 2)
plt.imshow(segmented_img)
plt.title('Bölütleme Sonucu')

plt.tight_layout()
plt.show()