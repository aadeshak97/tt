from django.urls import path
from . import views

app_name = 'org'

urlpatterns = [
    path('register', views.register_organization, name='org_register'),
    path('org_profile/<slug:slug>',
         views.org_profile,
         name='org_profile')
]