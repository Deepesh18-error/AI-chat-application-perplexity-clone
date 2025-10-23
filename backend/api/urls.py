from django.urls import path
from . import views

urlpatterns = [
    path('generate/', views.generate_view, name='generate_view'),
    path('sessions/', views.get_session_list, name='get_session_list'),
    # This single path now handles both GET (to fetch) and DELETE (to remove) a session.
    path('sessions/<str:session_id>/', views.session_detail_view, name='session_detail'),
]