import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm

B = 8
img1 = cv2.imread("lena.png", cv2.IMREAD_COLOR)
h, w = np.array(img1.shape[:2]) // B * B
img1 = img1[:h, :w]

img2 = np.zeros(img1.shape, np.uint8)
img2[:, :, 0] = img1[:, :, 2]
img2[:, :, 1] = img1[:, :, 1]
img2[:, :, 2] = img1[:, :, 0]
plt.imshow(img2)

point = plt.ginput(1)
block = np.floor(np.array(point) / B)
print("Coordinates of selected block: ", block)
scol = int(block[0, 0])
srow = int(block[0, 1])
plt.plot([B * scol, B * scol + B, B * scol + B, B * scol, B * scol],
         [B * srow, B * srow, B * srow + B, B * srow + B, B * srow])
plt.axis([0, w, h, 0])

# transform the picture from RGB to YCrCb
transcol = cv2.cvtColor(img1, cv2.COLOR_BGR2YCR_CB)

SSV = 2
SSH = 2
crf = cv2.boxFilter(transcol[:, :, 1], ddepth=-1, ksize=(2, 2))
cbf = cv2.boxFilter(transcol[:, :, 2], ddepth=-1, ksize=(2, 2))
crsub = crf[::SSV, ::SSH]
cbsub = cbf[::SSV, ::SSH]
imSub = [transcol[:, :, 0], crsub, cbsub]

# the quantization table for Y and CbCr
QY = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
               [12, 12, 14, 19, 26, 48, 60, 55],
               [14, 13, 16, 24, 40, 57, 69, 56],
               [14, 17, 22, 29, 51, 87, 80, 62],
               [18, 22, 37, 56, 68, 109, 103, 77],
               [24, 35, 55, 64, 81, 104, 113, 92],
               [49, 64, 78, 87, 103, 121, 120, 101],
               [72, 92, 95, 98, 112, 100, 103, 99]])

QC = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
               [18, 21, 26, 66, 99, 99, 99, 99],
               [24, 26, 56, 99, 99, 99, 99, 99],
               [47, 66, 99, 99, 99, 99, 99, 99],
               [99, 99, 99, 99, 99, 99, 99, 99],
               [99, 99, 99, 99, 99, 99, 99, 99],
               [99, 99, 99, 99, 99, 99, 99, 99],
               [99, 99, 99, 99, 99, 99, 99, 99]])

QF = 99.0
if 50 > QF > 1:
    scale = np.floor(5000 / QF)
elif QF < 100:
    scale = 200 - 2 * QF
else:
    print("Quality Factor must be in the range [1..99]")
scale = scale / 100.0
Q = [QY * scale, QC * scale, QC * scale]

TransAll = []
TransAllQuant = []
ch = ['Y', 'Cr', 'Cb']
plt.figure()
for idx, channel in enumerate(imSub):
    plt.subplot(1, 3, idx + 1)
    channelrows = channel.shape[0]
    channelcols = channel.shape[1]
    Trans = np.zeros((channelrows, channelcols), np.float32)
    TransQuant = np.zeros((channelrows, channelcols), np.float32)
    blocksV = channelrows // B
    blocksH = channelcols // B
    vis0 = np.zeros((channelrows, channelcols), np.float32)
    vis0[:channelrows, :channelcols] = channel
    vis0 = vis0 - 128
    for row in range(blocksV):
        for col in range(blocksH):
            currentblock = cv2.dct(vis0[row * B:(row + 1) * B, col * B:(col + 1) * B])
            Trans[row * B:(row + 1) * B, col * B:(col + 1) * B] = currentblock
            TransQuant[row * B:(row + 1) * B, col * B:(col + 1) * B] = np.round(currentblock // Q[idx])
    TransAll.append(Trans)
    TransAllQuant.append(TransQuant)
    if idx == 0:
        selectedTrans = Trans[srow * B:(srow + 1) * B, scol * B:(scol + 1) * B]
    else:
        sr = int(np.floor(srow / SSV))
        sc = int(np.floor(scol / SSV))
        selectedTrans = Trans[sr * B:(sr + 1) * B, sc * B:(sc + 1) * B]

    # this part is to show the dct result of the 8x8 block in jpeg implementation
    plt.imshow(selectedTrans, cmap=cm.jet, interpolation='nearest')
    plt.colorbar(shrink=0.5)
    plt.title('DCT of ' + ch[idx])


