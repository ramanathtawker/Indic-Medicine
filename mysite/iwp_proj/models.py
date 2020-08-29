from django.db import models

# Create your models here.
class items(models.Model):
    name = models.CharField(max_length=50)
    available=models.IntegerField()
    image = models.ImageField(upload_to='profile_image')
    def __str__(self):
        return self.name