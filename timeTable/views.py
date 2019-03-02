from django.shortcuts import render
from course.models import Subject
import GA

obj = GA.SGA.getInstance(0,4,7,4,11)
obj.callGA()

# 
