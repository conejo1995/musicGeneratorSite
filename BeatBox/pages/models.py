from django.db import models
from . import authentication
from authentication.models import User

# Create your models here.
class Song(models.Model):
    name = models.CharField(max_length=50)
    tune = models.FileField()
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    def __str__(self):
        return self.mainText