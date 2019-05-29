from django.db import models

# Create your models here.
class movie(models.Model):
	id = models.IntegerField(primary_key=True)
	title = models.CharField(max_length = 100)
	genres = models.CharField(max_length = 100)
	year = models.IntegerField()