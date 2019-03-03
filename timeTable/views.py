# from django.shortcuts import render, redirect
# from django.urls import reverse
# from django.http import HttpResponseRedirect
# from .forms import timetable
# from course.models import Organization , Faculty,Assignment
# import GA

# def timetable(request):	
# 	u=Assignment.objects.filter(is_assign=True).values_list('faculty__id','subject__subject_type','subject').count()	
# 	obj = GA.SGA.getInstance(4,u,3)
# 	obj.callGA()


