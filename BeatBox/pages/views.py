from django.shortcuts import render, HttpResponseRedirect
from django.http import HttpResponse
from django.template import loader
from django.contrib.auth.decorators import login_required

@login_required
def home(request):
    return render(request, 'pages/home.html', {})

def docs(request):
    return render(request, 'pages/docs.html', {})

@login_required
def create_beat(request):
    return render(request, 'pages/create_beat.html', {})

@login_required
def download_beat(request):
    return render(request, 'pages/download_beat.html', {})