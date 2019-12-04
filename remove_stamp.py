# -*- coding: utf-8 -*-
import cv2
import math
import numpy as np
import os
img = cv2.imread('test_test_test.jpg')
red_img = img[:, :, 2]
hue_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#-30 to 0
image_lower_hsv = np.array([160, 45, 100])
image_upper_hsv = np.array([180, 255, 255])
imageMask1 = cv2.inRange(hue_image, image_lower_hsv, image_upper_hsv)
cv2.imwrite('mask1.jpg', imageMask1)
#0 to 30
image_lower_hsv = np.array([0, 45, 100])
image_upper_hsv = np.array([10, 255, 255])
imageMask2 = cv2.inRange(hue_image, image_lower_hsv, image_upper_hsv)
cv2.imwrite('mask2.jpg', imageMask2)
finalMask = cv2.bitwise_or(imageMask1, imageMask2)
cv2.imwrite('finalMask.jpg', finalMask)
#final_img = cv2.add(img, finalMask)
#cv2.imwrite('final_img.jpg', final_img)
#edged = cv2.Canny(finalMask, 100, 255)
#cv2.imwrite('edge.jpg', edged)
_, contours, hierarchy = cv2.findContours(finalMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#areas = [cv2.contourArea(c) for c in contours]
#max_index = np.argmax(areas)
#rect = cv2.minAreaRect(contours[max_index])
for cnt in contours:
    areas = cv2.contourArea(cnt)
    if areas < 100:
        continue
    rect = cv2.minAreaRect(cnt)
    box = np.int0(cv2.boxPoints(rect))
    #img = cv2.drawContours(img.copy(), [box], -1, (255, 0, 0), 3)
    xs = [i[0] for i in box]
    ys = [i[1] for i in box]
    x1 = min(xs)
    x2 = max(xs)
    y1 = min(ys)
    y2 = max(ys)
    h = y2 - y1
    w = x2 - x1
    for i in range(y1, y1+h):
        for j in range(x1, x1+w):
            #print(i, j)
            if red_img[i, j] > 160:
                img[i, j] = [255, 255, 255]
    #roi = img[y1:y1+h, x1:x1+w]
    cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (255, 0,0), 12)
cv2.imwrite('stamp_img.jpg', img)

#index1 = finalMask == 255
#image = np.zeros(img.shape, np.uint8)
#image[:, :] = (255, 255, 255)
#image[index1] = img[index1]
#cv2.imwrite('img.jpg', image)


