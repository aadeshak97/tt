from django.db import models
from organization.models import Department,Faculty

# Create your models here.
class Course(models.Model):
	SEM=(
			('I','First'),
			('II','Second'),
			('III','Third'),
			('IV','Fourth'),
			('V','Fifth'),
			('VI','Sixth'),
			('VII','Seventh'),
			('VIII','Eighth')
		)
	PRO=(
			('UG','Under Graduate'),
			('PG','Post Graduate')
		)
	department = models.ForeignKey(Department, on_delete=models.CASCADE)
	semester = models.CharField(max_length=3,choices=SEM)
	program = models.CharField(max_length=3,choices=PRO)

	def __str__(self):
		return str(self.department) + "(" + str(self.semester) + ")"

    
class Subject(models.Model):
	CH=(
			('lab','Lab'),
			('lec','Lecture'),
			('tut','Tutorial'),
		)
	course = models.ForeignKey(Course, on_delete=models.CASCADE)
	code = models.CharField(max_length=20, unique=True)
	subject_type = models.CharField(max_length=3,choices=CH)
	faculty = models.ForeignKey(Faculty, on_delete=models.CASCADE)
	total_duration = models.IntegerField()
	period_per_week = models.IntegerField()
	per_period_duration = models.IntegerField()




