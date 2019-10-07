
from trend_filter import Trend_Fiter
import cv2
import numpy as np


noisy_im = cv2.imread('gen_noisy_im.png', cv2.IMREAD_GRAYSCALE)
assert noisy_im.shape[0] == noisy_im.shape[1]

tf = Trend_Fiter(noisy_im.shape[0])
res = tf.solve(noisy_im, vlambda=500)

res_im = res.astype(np.uint8)
cv2.imwrite('res_im.png', res_im)

# cv2.imshow('gen_noisy_im.png', noisy_im)
# cv2.imshow('res_im.png', res_im)
# cv2.waitKey()


