import numpy as np
import cv2

digits = cv2.imread("digits.png")
digits_gray = cv2.cvtColor(digits, cv2.COLOR_BGR2GRAY)
cells =[np.hsplit(row,100) for row in np.vsplit(digits_gray,50)]

x=np.array(cells)
print(x.shape)

train = x[:,:].reshape(-1,400).astype(np.float32)
print(train.shape)
# test = x[:,50:100].reshape(-1,400).astype(np.float32)
#
label = np.arange(10)
train_labels = np.repeat(label,500)[:,np.newaxis]

print(train_labels.shape)
np.savez("trained.npz", train=train, train_labels=train_labels)


