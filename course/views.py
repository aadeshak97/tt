from django.shortcuts import render
from .models import Course , Subject
from organization.models import Department,Faculty
from django.http import request




# Create your views here.
def demo(request):
	a= Subject.objects.all()
	return render(request,'course/demo.html',{'sub':a})	



