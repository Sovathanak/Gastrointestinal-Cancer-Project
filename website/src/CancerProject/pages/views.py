from django.shortcuts import render
from django.http import HttpResponse

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