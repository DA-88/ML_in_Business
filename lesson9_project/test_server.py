#import json

#from flask import Flask, render_template, redirect, url_for, request
#from flask_wtf import FlaskForm
#from requests.exceptions import ConnectionError
#from wtforms import IntegerField, SelectField, StringField
#from wtforms.validators import DataRequired

import urllib.request

myurl = "http://localhost:8180/predict"
req = urllib.request.Request(myurl)
req.add_header('Content-Type', 'application/json; charset=utf-8')
data = '28.7967,16.0021,2.6449,0.3918,0.1982,27.7004,22.011,-8.2027,40.092,81.8828'.encode('utf-8') # певая строка из датасета для теста
data = '121.5675,29.0624,3.3412,0.3196,0.1543,-139.7976,-69.814,-26.2826,21.5078,331.8356'.encode('utf-8') # строка с другим классом
req.add_header('Content-Length', len(data))
response = urllib.request.urlopen(req, data)
print(response.read())
