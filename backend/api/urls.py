# backend/api/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('generate/', views.generate_view, name='generate_view'),
    path('sessions/', views.get_session_list, name='get_session_list'),
    path('sessions/<str:session_id>/', views.get_session_history, name='get_session_history'),
]