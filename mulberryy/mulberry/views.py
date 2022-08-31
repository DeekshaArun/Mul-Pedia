from django.shortcuts import render
import pickle
import numpy as np
import cv2
from django.contrib import messages
from django.http import HttpResponseRedirect, HttpResponse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
# Create your views here.

def home1(request):

    return render(request, 'home.html')

def LeafyieldPred(request):
    return render(request, 'leafyieldForm.html')

def predictionForm(request):

    pH = request.POST['pH']
    EC = request.POST['EC']
    OC = request.POST['OC']
    P = request.POST['P']
    K = request.POST['K']
    Ca = request.POST['Ca']
    Mg = request.POST['Mg']
    S = request.POST['S']
    Zn = request.POST['Zn']
    Fe = request.POST['Fe']
    Mn = request.POST['Mn']
    Cu = request.POST['Cu']
    model = pickle.load(open('mulberry.sav', 'rb'))
    data = np.array([[pH,EC,OC,P,K,Ca,Mg,S,Zn,Fe,Mn,Cu]])
    data = data.astype(np.float64)
    prediction = model.predict(data)
    prediction = prediction[0]
    print(prediction)
    prediction = round(prediction,4)

    P= int(P)
    P =(100* P)/8.8
    P= round(P,2)

    K =int(K)
    K= (100* K)/41
    K= round(K,2)


    #return HttpResponse(prediction)
    return render(request, 'result.html', {'prediction': prediction, 'P': P, 'K':K})

def result(request):
    return render(request, 'result.html')

def leafdisease(request):
    return render(request, 'leafdisease.html')

def upload(request):
    if request.method == 'POST':
        try:

            uploaded_image = request.FILES['cd']



            print('Before loading model')
            # Loading the Model
            from tensorflow.keras.models import load_model
            model = load_model('mulberry.h5')

            # Making prediction using saved model!

            from tensorflow.keras.preprocessing import image
            print('before prepare')

            def prepare(filepath):
                img_size = 300
                img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_size, img_size))
                print('inside prepare')
                return new_array.reshape(-1, img_size, img_size, 1)

            result = model.predict([prepare(uploaded_image)])

            if result[0][0] == 1:
                prediction = 'normal'
                print("Prediction is:", prediction)
                messages.info(request, 'Normal')
            else:
                prediction = 'Diseased'
                print("prediction is:", prediction)
                messages.info(request, 'Diseased')


            return render(request, 'results.html')


        except Exception as e:
            print("Error is:", e)

    else:
            return render(request, 'results.html')

def results(request):
    return render(request, 'results.html')




