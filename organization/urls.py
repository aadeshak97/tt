from django.urls import path
from . import views

app_name = 'org'

urlpatterns = [
	path('register', views.register_organization, name='org_register'),
    path('org_profile/<slug:slug>',
         views.org_profile,
         name='org_profile'),
	path('org_depart/<slug:slug>',
		views.org_depart,
		name='org_depart'),
	path('department/faculty_list',
	 	views.faculty_list,
	 	name='faculty_list'),
	path('department/course_list',
		views.course_list,
		name='course_list')
]