from selenium import webdriver
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import function

# enter to the website of THSR
opt = webdriver.ChromeOptions()
# ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.135 Safari/537.36"
# opt.add_argument("user-agent={}".format(ua))
# opt.add_experimental_option('useAutomationExtension', False)
opt.add_experimental_option("excludeSwitches", ['enable-automation'])

url = "https://irs.thsrc.com.tw/IMINT/"
driver_path = './chromedriver'

browser = webdriver.Chrome(executable_path=driver_path)
browser.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
  "source": """
    Object.defineProperty(navigator, 'webdriver', {
      get: () => undefined
    })
  """
})
browser.get(url)
confirm_button = browser.find_element_by_name("confirm")
confirm_button.click()

# itenerary
start_station = "台北"
destination_station = "嘉義"
function.set_itenerary(browser, start_station, destination_station)

# class, defaul is standard class
train_class = 'standard'
function.set_class(browser, train_class)

#seat preference, default is none
seat_preference = 'none'
function.set_seat_preference(browser, seat_preference)

#search
method = 'time'
return_back = True
to_date, back_date = "2020/09/22", "2020/09/29"
to_time, back_time = '19:00', '15:00'
function.search(browser, method, return_back, to_date, back_date, to_time, back_time)

#passengers
function.set_passenger_num(browser, 1, 0, 0, 0, 0)

#discount train
function.only_show_discount(browser, False)

# get captcha
dir_name = 'captcha'
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)
filename1 = 'original.png'
path1 = os.path.join(dir_name, filename1)
function.get_captcha(browser, path1)
# browser.quit()

# recognize captcha
captcha = function.captcha_processing(path1)

filename2 = 'processed.png'
path2 = os.path.join(dir_name, filename2)
cv2.imwrite(path2, captcha)

model_path = './captcha recognition model/model/cnn_model.hdf5'
text = function.predict(model_path, captcha)

# enter the captcha
code = browser.find_element_by_name('homeCaptcha:securityCode')
code.send_keys(text)

submit = browser.find_element_by_name('SubmitButton')
submit.click()