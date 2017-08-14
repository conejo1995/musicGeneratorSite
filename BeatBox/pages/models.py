from django.db import models
from authentication.models import User

# Create your models here.
class Song(models.Model):
    name = models.CharField(max_length=50)
    tune_path = models.CharField(max_length=100)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    def __str__(self):
        return self.name
