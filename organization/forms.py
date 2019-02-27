from django import forms
from .models import Organization

class OrganizationForm(forms.ModelForm):
	class Meta:
		model = Organization
		exclude = ['admin']

class FacultyFilterForm(forms.Form):

	def __init__(self,org,*args,**kwargs):
		super(FacultyFilterForm,self).__init__(*args,*kwargs)
		choice_list=[(d.id, d.name) for d in org.departments.all()]
		self.fields['department']=forms.ChoiceField(choices=choice_list)

# class FilterForm(forms.Form):

# 	def __init__(self,org,*args,**kwargs):
# 		super(FacultyFilterForm,self).__init__(*args,*kwargs)
# 		choice_list=[(d.id, d.name) for d in org.departments.all()]
# 		self.fields['department']=forms.ChoiceField(choices=choice_list)




