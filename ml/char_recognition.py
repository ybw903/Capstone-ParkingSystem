import cv2
import numpy as np

FILE_NAME = 'ml/trained.npz'

def load_train_data(file_name):
    with np.load(file_name) as data:
        train = data['train']
        train_labels = data['train_labels']
    return train, train_labels

def check(test, train, train_labels):
    knn = cv2.ml.KNearest_create()
    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
    ret, result, neighbours, dist = knn.findNearest(test, k=5)
    return result

train, train_labels = load_train_data(FILE_NAME)

img = cv2.imread("ml/t.jpg", cv2.IMREAD_COLOR)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

ret,img_gray_binary = cv2.threshold(blur,80,255,cv2.THRESH_BINARY_INV)
cv2.imshow("gray",img_gray_binary)
cv2.waitKey(0)
cv2.destroyAllWindows()

canny = cv2.Canny(img_gray_binary, 100, 200)
contours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow("canny", canny)
cv2.waitKey(0)
cv2.destroyAllWindows()

box1 = []
for i in contours:
    area = cv2.contourArea(i)
    x, y, w, h = cv2.boundingRect(i)
    rect_area = w * h
    aspect_ratio = float(w) / h
    if (aspect_ratio >= 0.1) and (aspect_ratio <= 0.6) and (rect_area >= 1800) and (rect_area <= 5000):
        img_roi = img_gray_binary[y - 5:y + h + 10, x - 5:x + w + 10]
        box1.append(img_roi)

for i in box1:
    cv2.waitKey(0)
    gray_resize = cv2.resize(i, (20, 20))
    ret, gray_f = cv2.threshold(gray_resize, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow("t", gray_f)

    f=gray_resize.reshape(-1, 400).astype(np.float32)
    result = check(f,train,train_labels)
    print(result)
