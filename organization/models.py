from django.db import models
from django.db.models.signals import post_save
from django.contrib.auth.models import User

# Create your models here.
class Organization(models.Model):
	name = models.CharField(max_length=100)
	slug = models.SlugField(max_length=50, unique=True)
	admin =models.OneToOneField(User,on_delete=models.CASCADE,related_name='org')

	def __str__(self):
		return self.name


class Department(models.Model):
	organization = models.ForeignKey(Organization, 
		                             on_delete=models.CASCADE,
		                             related_name='departments')
	name = models.CharField(max_length=100)

	def __str__(self):
		return self.name

class Faculty(models.Model):
	department = models.ForeignKey(Department, on_delete=models.CASCADE)
	name = models.CharField(max_length=200)
	emp_id = models.CharField(max_length=200 ,unique=True)
	Faculty_load=models.IntegerField(default=1)

	def __str__(self):
		return str(self.name)+"("+str(self.emp_id)+")"
  
