from django.shortcuts import render, HttpResponseRedirect
from .models import User
from django.http import HttpResponse
# from .serializers import UserSerializer
# from rest_framework import viewsets
from django.template import loader
from django.contrib.auth import authenticate, login, logout

# Create your views here.

def register(request):
    if request.method == "POST":
        print(request.POST)
        usr = User()
        usr.username = request.POST.get('username', '')
        usr.set_password(request.POST.get('password', ''))
        usr.email = request.POST.get('email', '')
        usr.save()
        login(request, usr)
        return HttpResponseRedirect('/authentication/login/?next=/')

    return render(request, 'authentication/register.html', {})

def kenny_loggins(request):
    if request.method == "POST":
        print(request.POST)
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            # Redirect to a success page.
            return render(request, 'pages/home.html', {})

        else:
            # Return an 'invalid login' error message.
            return render(request, 'authentication/login.html', {})
    return render(request, 'authentication/login.html', {})

def kenny_loggouts(request):
    logout(request)
    # Redirect to a success page.
    return render(request, 'authentication/register.html', {})
