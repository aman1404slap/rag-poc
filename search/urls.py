from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/search/', views.api_search, name='api_search'),
    path('api/upload/', views.api_upload, name='api_upload'),
    path('api/jobs/<str:task_id>/', views.api_job_status, name='api_job_status'),
    path('api/videos/', views.api_videos, name='api_videos'),
    path('api/stats/', views.api_stats, name='api_stats'),
]
