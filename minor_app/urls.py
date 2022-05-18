from django.contrib import admin
from django.urls import path
from minor_app import views

urlpatterns = [
   path("",views.index,name='minor_app'),
   path("about",views.about,name='about'),
   path("contacts",views.contacts,name='contacts'),
   path("do",views.results,name='results'),
   path("new",views.new,name='new'),
   path("potential",views.potential,name="potential")
]