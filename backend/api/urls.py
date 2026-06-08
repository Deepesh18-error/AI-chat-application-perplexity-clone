from django.urls import path
from . import views

urlpatterns = [
    path('health/', views.health_view, name='health_view'),
    path('auth/register/', views.register_view, name='register_view'),
    path('auth/login/', views.login_view, name='login_view'),
    path('auth/refresh/', views.refresh_view, name='refresh_view'),
    path('auth/logout/', views.logout_view, name='logout_view'),
    path('auth/me/', views.me_view, name='me_view'),
    path('generate/', views.generate_view, name='generate_view'),
    path('sessions/', views.get_session_list, name='get_session_list'),
    path('sessions/<str:session_id>/', views.session_detail_view, name='session_detail'),
]
