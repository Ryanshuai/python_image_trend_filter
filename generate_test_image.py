import cv2
import numpy as np


im_h, im_w = 64, 64

im = np.ones((im_h, im_w))*100
for i in range(im_h//4):
    for j in range(im_w//4):
        im[i][j] = 200
for i in range(im_h//3):
    for j in range(im_w//2):
        im[im_h-i-1][im_w-j-1] = 50

im = im.astype(np.uint8)
cv2.imwrite('gen_im.png', im)


sigma = 10
noisy_im = im + sigma*np.random.randn(im_h, im_w)
noisy_im = noisy_im.astype(np.uint8)
cv2.imwrite('gen_noisy_im.png', noisy_im)
