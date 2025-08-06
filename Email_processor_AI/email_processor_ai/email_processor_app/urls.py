from django.urls import path
from .views import ai_response, get_session_history, health_check

urlpatterns = [
    path('ai/', ai_response, name='ai_response'),
    path('session/<str:session_id>/history/', get_session_history, name='session_history'),
    path('health/', health_check, name='health_check'),
]