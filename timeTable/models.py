from django.db import models	
from organization.models import Department,Faculty,Organization
from course.models import Course,Subject,Assignment

class time_table(models.Model):

	semester = models.IntegerField(default=0)
	period = models.IntegerField(default=0)
	days= models.IntegerField(default=0)
	lect=models.IntegerField(default=0)
	types=models.IntegerField(default=0)
	sub=models.IntegerField(default=0)
	#days=models.CharField(max_length=3,choices=day)
	

