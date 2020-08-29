from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from .models import items
# Create your views here.
def index(request):
    name=items.objects.all()
    template=loader.get_template('iwp_proj/index.html')
    context = {
        'name': name,
    }
    return HttpResponse(template.render(context, request))