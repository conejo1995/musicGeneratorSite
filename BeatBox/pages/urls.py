from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^home/$', views.home, name='home'),
    url(r'^docs/$', views.docs, name='docs'),
    url(r'^home/create_beat/$', views.create_beat, name='create_beat'),
    url(r'^home/download_beat/$', views.download_beat, name='download_beat'),
    # url(r'^$', views.home, name='home'),
]