# uploader/urls.py

from django.urls import path
from .views import *



urlpatterns = [
    path('', FileUploadView.as_view(), name='file-upload'),
    path('advanced-plot/', AdvancedPlotView.as_view(), name='advanced_plot'),
    path('register/', register, name='register'),
    path('dashboard/', DashboardView.as_view(), name='dashboard'),
]

