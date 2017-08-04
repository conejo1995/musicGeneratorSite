from django.shortcuts import render, HttpResponseRedirect
from .models import User
from .serializers import UserSerializer
from rest_framework import viewsets

# Create your views here.
def register(request):
    if request.method == "POST":
        print(request.POST)
        usr = User()
        usr.username = request.POST.get('username', '')
        usr.set_password(request.POST.get('password', ''))
        usr.email = request.POST.get('email', '')
        usr.first_name = request.POST.get('first_name', '')
        usr.last_name = request.POST.get('last_name', '')
        usr.save()
        return HttpResponseRedirect('/accounts/login/?next=/')

    return render(request, 'registration/register.html', {})

