# uploader/urls.py

from django.urls import path
from .views import *



urlpatterns = [
    path('', FileUploadView.as_view(), name='file-upload'),
    path('register/', register, name='register'),
    path('dashboard/', DashboardView.as_view(), name='dashboard'),
]

