from django.urls import path
from .views import ai_response

urlpatterns = [
    path('ai/', ai_response, name='ai_response'),
]