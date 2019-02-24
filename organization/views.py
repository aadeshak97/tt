from django.shortcuts import render, redirect
from django.urls import reverse
from django.http import HttpResponseRedirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate, login
from .forms import OrganizationForm
from .models import Organization

# Create your views here.

def register_organization(request):	
	if request.method == 'POST':
		admin_form = UserCreationForm(request.POST)
		org_form=OrganizationForm(request.POST)
		if admin_form.is_valid() and org_form.is_valid():
			org_form.save(commit=False)
			admin = admin_form.save()
			org_form.instance.admin = admin
			org_form.save()
			user_name = admin_form.cleaned_data.get("username")
			raw_password = admin_form.cleaned_data.get("password1")
			user = authenticate(request,username=user_name, password=raw_password)
			login(request, user)
			return redirect(reverse('org:org_profile', 
				                    args =[org_form.cleaned_data.get("slug")]))

	else:
		admin_form = UserCreationForm()
		org_form=OrganizationForm()
	return render(request,
	          'organization/register.html',
	          {'admin_form':admin_form,'org_form':org_form})


def org_profile(request, slug):
	org=Organization.objects.get(slug=slug)
	return render(request, 'organization/org_home.html',{'org':org})


