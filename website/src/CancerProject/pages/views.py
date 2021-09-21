from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
def home_view(request, *args, **kwargs):
    # return HttpResponse("<h1>Test</h1>")
    return render(request, "index.html", {})
