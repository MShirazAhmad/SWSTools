# uploader/urls.py

from django.urls import path
from .views import *



# uploader/urls.py

from django.urls import path
from .views import *
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', FileUploadView.as_view(), name='file-upload'),
    path('select-axes/<int:file_id>/', AxesSelectionView, name='select-axes'),
    path('dashboard/', DashboardView, name='dashboard'),  # Correct for FBV
    # path('register/', RegisterView.as_view(template_name='registration/register.html'), name='register'),
    path('login/', auth_views.LoginView.as_view(template_name='registration/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='file-upload'), name='logout'),
]

