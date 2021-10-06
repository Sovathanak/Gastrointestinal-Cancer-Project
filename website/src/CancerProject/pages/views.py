from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import  render
from django.core.files.storage import FileSystemStorage
from keras.models import load_model
from keras.preprocessing import image
from tensorflow import Graph
import tensorflow as tf
import json
import numpy as np
import os
from CancerProject.settings import BASE_DIR

# load in the model
height, width = 224, 224
with open(os.path.join(BASE_DIR, "static/models/model.json"), 'r') as model:
    label = model.read()
label = json.loads(label)

model_graph = Graph()
with model_graph.as_default():
    tf_session = tf.compat.v1.Session()
    with tf_session.as_default():
        model = load_model(os.path.join(BASE_DIR, "static/models/inception_model.h5"))
        
# Create your views here.
def home_view(request, *args, **kwargs):
    # return HttpResponse("<h1>Test</h1>")
    return render(request, "index.html", {})

def info_view(request, *args, **kwargs):
    # return HttpResponse("<h1>Test</h1>")
    return render(request, "info.html", {})

def model_view(request, *args, **kwargs):
    # return HttpResponse("<h1>Test</h1>")
    return render(request, "model.html", {})

def contact_view(request, *args, **kwargs):
    # return HttpResponse("<h1>Test</h1>")
    return render(request, "contact.html", {})

def privacy_view(request, *args, **kwargs):
    # return HttpResponse("<h1>Test</h1>")
    return render(request, "privacy.html", {})

def image_prediction(request):
    if request.method == 'POST' and request.FILES['upload']:
        upload = request.FILES['upload']
        fss = FileSystemStorage()
        file = fss.save(upload.name, upload)
        file_url = fss.url(file)
        test_image = '.'+file_url
        
        # load image
        img = image.load_img(test_image, target_size = (height, width))
        x = image.img_to_array(img)
        x = x.reshape(1, height, width, 3) # reshape to (1, 224, 224, 3), format which the model uses to predict
        
        with model_graph.as_default():
            with tf_session.as_default():
                prediction_res = model.predict(x)
        # placeholder for showing prediction (incomplete/unsure)
        if prediction_res < 0:
            prediction_label = "MSS"
        else:
            prediction_label = "MSIMUT"
        return render(request, 'model.html', {'file_url': file_url, 'prediction_label': prediction_label})
    return render(request, 'model.html')