# THSR_booking_system
It's my self practice program.

The system is a tool help you buy Taiwan High Speed Rail(THSR) tickets easier.  
It can automatically opne your Chrome browser and connect to THSR ticketing website. Then the system will help fill out the information like itinerary, date and so on, and finally use a pre-train CNN model to recognize the Captcha.

# Directory Structure
  * `booking_ticket.py`: source code of booking system.
  * `function.py`: custom functions used in `booking_ticket.py`.
  * `chromedriver.exe`: driver for for selenium to open Chrome.
  * `captcha/`: folder to save original and pre-processed Captcha.
  * `captcha recognition model/`: folder to save CNN model for Captcha recognition.
  * `requirements.txt`: python3 package requirements. Please try to install all the packages listed.
  
# Run
To run the system, follow the below steps.

1. Because this is a prototype, please enter `booking_ticket.py` to edit your information like itinerary, date, time and so on.  
2. run the following command on your commandline.  
Usage:  
```python ./booking_ticket.py```
