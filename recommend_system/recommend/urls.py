from . import views
from django.urls import path
from django.conf.urls import url, include

urlpatterns = [
    path("sim", views.similar_movie, name='similar_movie'),
    path("add_user", views.add_user, name='add_user'),
    path('add_rating', views.add_rating, name="add_rating"),
    path('recommend', views.recommend, name = 'recommend')
    ]