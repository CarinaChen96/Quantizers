import numpy as np
from PIL import Image


def sort_by_max_difference_dimension(image_arr):
    diff_a = np.amax(image_arr, axis=0) - np.amin(image_arr, axis=0)
    index = np.argmax(diff_a)
    image_arr.sort(key=lambda x: x[index])


def save_image(image, bit_number):
    img = Image.fromarray(image)
    name = 'quantized-image' + str(bit_number) + '.jpg'
    img.save(name)


class MC:

    def __init__(self, img):
        self.img = img
        self.rows, self.cols, color = img.shape
        self.img_a = []
        self.mapping = {1: {}, 2: {}, 4: {}, 8: {}, 16: {}}
        self.q_image = {1: np.copy(img), 2: np.copy(img), 4: np.copy(img), 8: np.copy(img), 16: np.copy(img)}

        for i in range(self.rows):
            for j in range(self.cols):
                self.img_a.append(img[i][j])

    def median_cut(self):
        img_array = self.img_a
        # current_cut, target_cut
        self.helper(img_array, 0, 16)
        self.generate_compressive_image()

    # generate 4 compressive image by given bits number and created mapping table
    def generate_compressive_image(self):
        for n in [1, 2, 4, 8, 16]:
            for i in range(self.rows):
                for j in range(self.cols):
                    pixel_v = self.img[i][j]
                    self.q_image[n][i][j] = self.mapping[n][tuple(pixel_v)]

            save_image(self.q_image[n], n)

    # Recursively partition the pixel group and make mapping tables
    def helper(self, image_arr, current_cut, target):
        if current_cut >= target:
            return
        current_cut += 1

    # It's a sub function of helper, use it to sort the pixel array by maximum difference value of dimensional value
        sort_by_max_difference_dimension(image_arr)
        lens = len(image_arr)
        left = image_arr[0:int(lens / 2)]
        right = image_arr[int(lens / 2):lens]
        if current_cut in [1, 2, 4, 8, 16]:
            self.compute_mean(left, current_cut)
            self.compute_mean(right, current_cut)

        self.helper(left, current_cut, target)
        self.helper(right, current_cut, target)

    # It's a sub function of helper, use it to calculate the mean value of its small group
    def compute_mean(self, image_arr, current_cut):
        mean = np.mean(image_arr, axis=0)
        for pixel in image_arr:
            self.mapping[current_cut][tuple(pixel)] = mean


ori_image = Image.open('yourname.png')
# Some of the input images may not the RGB format while mc have to implement on RGB
rgb_image = ori_image.convert('RGB')
image = np.asarray(rgb_image)
MC(image).median_cut()
