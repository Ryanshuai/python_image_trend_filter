import cv2
import numpy as np


im_h, im_w = 64, 64

im = np.ones((im_h, im_w))*100

for i in range(im_h):
    for j in range(im_w):
        if i <= im_h//3 and j >= im_w//3*2:
            im[i][j] = 200
        if i > im_h//3*2 and j <= im_w//3:
            im[i][j] = 50
        if i+j <= ((im_w+im_h)/2)//3:
            im[i][j] = 200
        if i>(im_w+im_h)//4 and j>(im_w+im_h)//4 and \
                np.sqrt((i-im_h//2)*(i-im_h//2)+(j-im_w//2)*(j-im_w//2)) >= ((im_w+im_h)/4):
            im[i][j] = 50

im = im.astype(np.uint8)
cv2.imwrite('k0_im.png', im)


sigma = 10
noisy_im = im + sigma*np.random.randn(im_h, im_w)
noisy_im = noisy_im.astype(np.uint8)
cv2.imwrite('noisy_k0_im.png', noisy_im)
