from skimage import io
from sklearn.cluster import KMeans
import numpy as np

image = io.imread('yourname.png')
io.imshow(image)
io.show()

rows = image.shape[0]
cols = image.shape[1]

image = image.reshape(image.shape[0] * image.shape[1], -1)
# using the sklearn lib to implement the Kmeans and the n_clusters can be changed
# to achieve different clustering results
kmeans = KMeans(n_clusters=32, n_init=10, max_iter=200)
kmeans.fit(image)

clusters = np.asarray(kmeans.cluster_centers_, dtype=np.uint8)
labels = np.asarray(kmeans.labels_, dtype=np.uint8)
labels = labels.reshape(rows, cols)

np.save('codebook_yourname.npy', clusters)
io.imsave('compressed_yourname.png', labels)