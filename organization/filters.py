# import django_filters
# from .models import Faculty,Department



# def departments(request):
#     if request is None:
#         return Department.objects.none()

#     organization = request.user.org
#     return organization.departments.all()

# class FacultyFilter(django_filters.FilterSet):
# 	department =django_filters.ModelChoiceFilter(queryset=departments)

# 	class Meta:
# 		model = Faculty
# 		fields = ['department']
    
	
#     