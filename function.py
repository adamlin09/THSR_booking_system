from selenium import webdriver
from selenium.webdriver.support.ui import Select
from enum import Enum
import time
import os
import urllib.request
import requests
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import copy
from keras.models import load_model

def set_itenerary(webdriver, start, destination):
    select_start_station = Select(webdriver.find_element_by_name('selectStartStation'))
    select_destination_station = Select(webdriver.find_element_by_name('selectDestinationStation'))
    select_start_station.select_by_visible_text(start)
    select_destination_station.select_by_visible_text(destination)

def set_class(webdriver, train_class = 'standard'):
    #defaul is standard class 
    if train_class == 'business':
        class_button = webdriver.find_element_by_id("trainCon:trainRadioGroup_1")
        class_button.click()

def set_seat_preference(webdriver, seat_preference = 'none'):
    #default is none
    if seat_preference == 'window':
        preference_button = webdriver.find_element_by_id("seatRadio1")
        preference_button.click()
    elif seat_preference == 'aisle':
        preference_button = webdriver.find_element_by_id("seatRadio2")
        preference_button.click()

def search(webdriver, method = 'time', return_back = False, to_date = '', back_date = '', to_time = '', back_time = '', to_id = '', back_id = ''):
    if method == 'No.':     #by No.
        method_button = webdriver.find_element_by_id("bookingMethod_1")
        method_button.click()
    
    if return_back:
        return_back_box = webdriver.find_element_by_name("backTimeCheckBox")
        return_back_box.click()
    
    to_date_field = webdriver.find_element_by_name("toTimeInputField")
    to_date_field.clear()
    to_date_field.send_keys(to_date)
    if return_back:
        back_date_field = webdriver.find_element_by_name("backTimeInputField")
        back_date_field.clear()
        back_date_field.send_keys(back_date)
        back_date_field.clear()
        back_date_field.send_keys(back_date)

    #search by time
    if method == 'time':
        select_to_time = Select(webdriver.find_element_by_name("toTimeTable"))
        select_to_time.select_by_visible_text(to_time)
        if return_back:
            select_back_time = Select(webdriver.find_element_by_name("backTimeTable"))
            select_back_time.select_by_visible_text(back_time)

    #search by No.
    if method == 'No.':
        input_to_id = webdriver.find_element_by_name("toTrainIDInputField")
        input_to_id.send_keys(to_id)
        if return_back:
            input_back_id = webdriver.find_element_by_name("backTrainIDInputField")
            input_back_id.send_keys(back_id)

def set_passenger_num(webdriver, adult = 1, child = 0, disable = 0, elderly = 0, college = 0):
    adult_num = Select(webdriver.find_element_by_name("ticketPanel:rows:0:ticketAmount"))
    adult_num.select_by_index(adult)
    
    child_num = Select(webdriver.find_element_by_name("ticketPanel:rows:1:ticketAmount"))
    child_num.select_by_index(child)
    
    disable_num = Select(webdriver.find_element_by_name("ticketPanel:rows:2:ticketAmount"))
    disable_num.select_by_index(disable)
    
    elderly_num = Select(webdriver.find_element_by_name("ticketPanel:rows:3:ticketAmount"))
    elderly_num.select_by_index(elderly)
    
    college_num = Select(webdriver.find_element_by_name("ticketPanel:rows:4:ticketAmount"))
    college_num.select_by_index(college)

def only_show_discount(webdriver, only_discount = False):
    if only_discount:
        only_discount_box = webdriver.find_element_by_name("offPeakTrainSearchContainer:onlyQueryOffPeak")
        only_discount_box.click()

def get_captcha(webdriver, path):
    img = webdriver.find_element_by_id("BookingS1Form_homeCaptcha_passCode")
    img.screenshot(path)

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

def predict(model_path, img):
    dic19 = ['2', '3', '4', '5', '7', '9', 'A', 'C', 'F', 'H', 'K', 'M', 'N', 'P', 'Q', 'R', 'T', 'Y', 'Z']
    model = load_model(model_path)
    # print(model.summary())

    x_train = np.stack([img / 255.0])
    prediction = model.predict(x_train)

    text = ''
    for i in prediction:
        text += dic19[np.argmax(i)]

    return text