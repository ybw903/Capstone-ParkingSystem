import numpy as np
import cv2

img = cv2.imread("t.jpg", cv2.IMREAD_COLOR)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
canny = cv2.Canny(blur,100,200)
contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow("canny", canny)
cv2.waitKey(0)
cv2.destroyAllWindows()
box1 = []
# min_x = 987654321
# max_x = 0
# uy = 0
# dy = 0
for i in range(len(contours)):
    cnt = contours[i]
    area = cv2.contourArea(cnt)
    x,y,w,h = cv2.boundingRect(cnt)
    rect_area = w*h
    aspect_ratio = float(w)/h
    if(aspect_ratio>=0.1)and(aspect_ratio<=0.6)and(rect_area>=1800)and(rect_area<=5000):
        # cv2.rectangle(img,(x-5,y-5),(x+w+5,y+h+5),(0,255,0),1)
        img_roi = img[y-5:y+h+5,x-5:x+w+5]
        box1.append(img_roi)
        # if(x+h>max_x):
        #     max_x=x+h
        # if(x<min_x):
        #     min_x=x
        # dy=y
        # uy=y+h
        # box1.append(cv2.boundingRect(cnt))
# print(min_x,max_x,dy,uy)
# cv2.rectangle(img,(min_x-30,dy-60),(max_x+10,uy+60),(0,255,0),1)
for i in box1:
    cv2.imshow("t", i)
    cv2.waitKey(0)
    cv2.destroyWindow("t")

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

