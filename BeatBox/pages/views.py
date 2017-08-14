from django.shortcuts import render, HttpResponseRedirect
from django.http import HttpResponse
from django.template import loader
from django.contrib.auth.decorators import login_required
from . import functions
from .models import Song
from django.core.files.storage import FileSystemStorage
from django.contrib.staticfiles.templatetags.staticfiles import static
import os
from wsgiref.util import FileWrapper
import mimetypes

@login_required
def home(request):
    return render(request, 'pages/home.html', {})

def docs(request):
    return render(request, 'pages/docs.html', {})

@login_required
def create_beat(request):
    if request.method == "POST" and request.FILES['sample1'] and request.FILES['sample2']:
        print(request.POST)
        # song = Song()
        # song.name = request.POST.get('songName', 'sample')
        # song.user = request.user
        sample1 = request.FILES['sample1']
        sample2 = request.FILES['sample2']
        fs = FileSystemStorage()
        samp1 = fs.save(sample1.name, sample1)
        samp2 = fs.save(sample2.name, sample2)
        samp1_path = fs.path(samp1)
        samp2_path = fs.path(samp2)
        functions.generate_midi(samp1_path, samp2_path, samp1_path)
        song = Song()
        song.name = request.POST.get('songName', 'sample')
        song.tune_path = samp1_path
        song.user = request.user
        song.save()

        wrapper = FileWrapper(open(samp1_path, "rb"))
        type = mimetypes.guess_type(samp1_path)[0]

        response = HttpResponse(wrapper, content_type=type)
        response['Content-Length'] = os.path.getsize(samp1_path)
        response['Content-Disposition'] = 'attachment; filename="' + request.POST.get("songName") +'.mid"'
        return response

    return render(request, 'pages/create_beat.html', {})

@login_required
def download_beat(request):
    songs = Song.objects.all()
    template = loader.get_template('pages/download_beat.html')

    if request.method == "POST":
        print(request.POST)
    context = {
        'songs': songs,
    }
    return HttpResponse(template.render(context, request))

