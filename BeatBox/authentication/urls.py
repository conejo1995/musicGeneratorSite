from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.register, name='register'),
    url(r'^register/$', views.register, name='register'),
    url(r'^login/$', views.kenny_loggins, name='login'),
    url(r'^logout/$', views.kenny_loggouts, name='logout'),
]