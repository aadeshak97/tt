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
	working_days = models.IntegerField(default=0)
	working_hrs = models.IntegerField(default=0)
	


	def __str__(self):
		return str(self.department) + "(" + str(self.semester) + ")"

    
class Subject(models.Model):
	CH=(
			('1','Lab'),
			('2','Lecture'),
			('3','Tutorial'),
		)
	course = models.ForeignKey(Course, on_delete=models.CASCADE)
	name = models.CharField(max_length=20, unique=False)
	code = models.CharField(max_length=20, unique=True)
	subject_type = models.CharField(max_length=3,choices=CH)
	faculty = models.ManyToManyField(Faculty, through='Assignment')
	total_duration = models.IntegerField()
	period_per_week = models.IntegerField()
	per_period_duration = models.IntegerField()

	def __str__(self):
		return self.name

class Assignment(models.Model):
	#courses = models.ForeignKey(Course, on_delete=models.CASCADE)
	course = models.ForeignKey(Course, on_delete=models.CASCADE,default=0)
	subject = models.ForeignKey(Subject, on_delete=models.CASCADE)
	faculty=models.ForeignKey(Faculty, on_delete=models.CASCADE)
	is_assign=models.BooleanField(default=False)

	def __str__(self):
		return str(self.faculty)+"("+str(self.subject)+")"+"("+str(self.is_assign)+")"+"("+str(self.course)+")"


# class Relation(models.Model):
# 	dept=