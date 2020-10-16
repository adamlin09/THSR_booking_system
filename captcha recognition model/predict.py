import numpy as np
from PIL import Image
from keras.models import load_model
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import copy

def predict(img):
    dic19 = ['2', '3', '4', '5', '7', '9', 'A', 'C', 'F', 'H', 'K', 'M', 'N', 'P', 'Q', 'R', 'T', 'Y', 'Z']
    model = load_model('./model/cnn_model.hdf5')
    # print(model.summary())

    x_train = np.stack([img / 255.0])
    prediction = model.predict(x_train)

    text = ''
    for i in prediction:
        text += dic19[np.argmax(i)]

    return text

def captcha_processing(load_path):
    ### size = (w, h)
    ### return RGB image array

    img = cv2.imread(load_path) # numpy array BGR
    h1, w1, c1 = img.shape
    print((h1, w1, c1))

    # Show the original img
    # plt.subplot(5, 1, 1)
    # plt.imshow(img)

    # denoise
    img = cv2.fastNlMeansDenoisingColored(img, h=35, hColor=35, templateWindowSize=7, searchWindowSize=21)

    # 黑白化(字變白)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    # plt.subplot(5, 1, 2)
    # plt.imshow(thresh)
    
    # resize to increase pixels
    scale = 10
    thresh = cv2.resize(thresh, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)  # x10
    imgarr = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)   # x10
    h2, w2 = imgarr.shape

    # erase texts
    imgarr[:, 100 : w2 - 60] = 0
    # plt.subplot(5, 1, 3)
    # plt.imshow(imgarr, 'gray')

    # draw the line
    imgdata = np.where(imgarr == 255)
    X = np.array([imgdata[1]])
    Y = h2 - imgdata[0]
    # j = 0
    # a = []
    # for i in X[0]:
    #     if i == 50:
    #       a.append(Y[j])
    #     j += 1  
    # print(a[np.argmax(a)] - a[np.argmin(a)])

    poly = PolynomialFeatures(degree=2)
    reg = LinearRegression()
    X_ = poly.fit_transform(X.T)
    reg.fit(X_, Y)

    X2 = np.array([[i for i in range(w2)]])
    X2_ = poly.fit_transform(X2.T)
    Y2 = reg.predict(X2_)

    # clear the line
    newarr = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)   # x10
    for i in np.column_stack([Y2.round(0), X2[0]]):
        pos = int(h2 - i[0])
        newarr[pos - 21 : pos + 21, int(i[1])] = 255 - newarr[pos - 21 : pos + 21, int(i[1])]
    newarr[:, :100] = 0
    newarr[:, w2 - 60] = 0

    # plt.subplot(5, 1, 4)
    # plt.imshow(newarr, 'gray')

    # 閉運算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    newarr = cv2.morphologyEx(newarr, cv2.MORPH_OPEN, kernel)
    newarr = cv2.morphologyEx(newarr, cv2.MORPH_CLOSE, kernel)

    # plt.subplot(5, 1, 5)
    # plt.imshow(newarr, 'gray')
    # plt.show()

    # resize to (140, 48)
    newarr = cv2.resize(newarr, (140, 48), interpolation=cv2.INTER_AREA)
    newarr = cv2.cvtColor(newarr, cv2.COLOR_GRAY2RGB)
    
    return newarr
    