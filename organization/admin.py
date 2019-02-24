from django.contrib import admin
from .models import Organization,Department,Faculty

# Register your models here.
admin.site.register(Organization)
admin.site.register(Department)
admin.site.register(Faculty)
