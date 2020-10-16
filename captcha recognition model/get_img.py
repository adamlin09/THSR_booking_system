import requests
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
import os 
from predict import predict, captcha_processing
import cv2
import matplotlib.pyplot as plt

url = 'https://irs.thsrc.com.tw/IMINT/?locale=tw'
data = {
    'BookingS1Form:hf:0': '',
    #起程站
    'selectStartStation': 2,
    #到達站
    'selectDestinationStation': 10,
    #標準/商務車廂
    'trainCon:trainRadioGroup': 0,
    #座位喜好(無17/窗19/道21)
    'seatCon:seatRadioGroup': 'radio17',
    #時間/車次
    'bookingMethod': 0,
    # 日期時間車次
    'toTimeInputField': '2020/09/10',
    'toTimeTable': '15:00',
    'toTrainIDInputField': '',
    'backTimeInputField': '',
    'backTimeTable': '',
    'backTrainIDInputField': '',
    #張數(全/孩童/愛心/敬老/大學生)
    'ticketPanel:rows:0:ticketAmount': '1F',
    'ticketPanel:rows:1:ticketAmount': '0H',
    'ticketPanel:rows:2:ticketAmount': '0W',
    'ticketPanel:rows:3:ticketAmount': '0E',
    'ticketPanel:rows:4:ticketAmount': '0P',
    # #驗證碼
    # 'homeCaptcha:securityCode': '',
    'portalTag': 'false',
    'SubmitButton': '開始查詢'
}

url = 'https://irs.thsrc.com.tw/IMINT/'
headers = {'user-agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.135 Safari/537.36'}
session = requests.session()
session.headers.update(headers)
r = session.post(url, data=data)
print(r)


# dir_name = 'img'
# if not os.path.isdir(dir_name):
#     os.mkdir(dir_name)

# # print(r.text)
# soup = BeautifulSoup(r.text, 'html.parser')
# img  = soup.find('img')
# img_url ='https://irs.thsrc.com.tw' + img.get('src')
# print(img_url)
# r = session.get(img_url)
# filename = '1' + '.png'
# with open(os.path.join(dir_name, filename), 'wb') as f:
#     f.write(r.content)

# captcha = captcha_processing('./img/1.png')
# cv2.imwrite(os.path.join(dir_name, '2.png'), captcha)
# t = predict(captcha)
# print(t)

# plt.imshow(captcha)
# plt.show()