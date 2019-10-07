
from trend_filter import Trend_Fiter
import cv2
import numpy as np


noisy_im = cv2.imread('noisy_k0_im.png', cv2.IMREAD_GRAYSCALE)
assert noisy_im.shape[0] == noisy_im.shape[1]

tf = Trend_Fiter(noisy_im.shape[0], k=0)
res = tf.solve(noisy_im, vlambda=1000)

res_im = res.astype(np.uint8)
cv2.imwrite('res_k0_im.png', res_im)


noisy_im = cv2.imread('noisy_k1_im.png', cv2.IMREAD_GRAYSCALE)
assert noisy_im.shape[0] == noisy_im.shape[1]

tf = Trend_Fiter(noisy_im.shape[0], k=0)
res = tf.solve(noisy_im, vlambda=1000)

res_im = res.astype(np.uint8)
cv2.imwrite('res_k1_im.png', res_im)


