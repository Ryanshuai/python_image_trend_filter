import cv2
import numpy as np

im_h, im_w = 64, 64

# for k=1
im = np.ones((im_h, im_w)) * 130

for i in range(im_h):
    for j in range(im_w):
        if i <= im_h // 3 and j >= im_w // 3 * 2:
            im[i][j] = 200
        if i > im_h // 3 * 2 and j <= im_w // 3:
            im[i][j] = 60
        if i + j <= ((im_w + im_h) / 2) // 2:
            im[i][j] = 200
        if i > (im_w + im_h) // 4 and j > (im_w + im_h) // 4 and \
                np.sqrt((i - im_h // 2) * (i - im_h // 2) + (j - im_w // 2) * (j - im_w // 2)) >= ((im_w + im_h) / 4):
            im[i][j] = 60
        if i <= im_h // 8:
            im[i][j] = 100
        if j <= im_w // 8:
            im[i][j] = 160

im = np.clip(im, 0, 255, out=None)
im = im.astype(np.uint8)
cv2.imwrite('k0_im.png', im)

sigma = 10
noisy_im = im + sigma * np.random.randn(im_h, im_w)
im = np.clip(im, 0, 255, out=None)
noisy_im = noisy_im.astype(np.uint8)
cv2.imwrite('noisy_k1_im.png', noisy_im)

# for k=1
im = np.zeros((im_h, im_w))

for i in range(im_h):
    for j in range(im_w - 1, -1, -1):
        if j > im_w // 4 * 3:
            im[i][j] = 50 + 256 * (j - im_w // 4 * 3) // im_w
        else:
            im[i][j] = 50 + 100 * abs(j - im_w // 4 * 3) // im_w
        if i + j == im_w // 2:
            j_record = im[i][j]
        if i + j < im_w // 2:
            im[i][j] = j_record - 3
            j_record -= 1

i_record = 0
for i in range(im_h):
    if i > im_h // 4 * 3:
        i_record += 3
    for j in range(im_w):
        im[i][j] -= i_record

im = np.clip(im, 0, 255, out=None)
im = im.astype(np.uint8)
cv2.imwrite('k1_im.png', im)

sigma = 5
noisy_im = im + sigma * np.random.randn(im_h, im_w)
im = np.clip(im, 0, 255, out=None)
im = im.astype(np.uint8)
cv2.imwrite('noisy_k1_im.png', noisy_im)
