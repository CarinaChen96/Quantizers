from skimage import io
import numpy as np


centers = np.load('codebook_yourname.npy')

c_image = io.imread('compressed_yourname.png')

# using the value of the central point to stand for the value of all the points with the same label
image = np.zeros((c_image.shape[0], c_image.shape[1], 3), dtype=np.uint8)
for i in range(c_image.shape[0]):
    for j in range(c_image.shape[1]):
        image[i, j, :] = centers[c_image[i, j], :]

io.imsave('reconstructed_yourname.jpg', image)
io.imshow(image)
io.show()