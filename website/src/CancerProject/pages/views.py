from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import  render
from django.core.files.storage import FileSystemStorage

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

def upload(request):
    if request.method == 'POST' and request.FILES['upload']:
        upload = request.FILES['upload']
        fss = FileSystemStorage()
        file = fss.save(upload.name, upload)
        file_url = fss.url(file)
        return render(request, 'model.html', {'file_url': file_url})
    return render(request, 'model.html')