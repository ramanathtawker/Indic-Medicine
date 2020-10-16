from django.urls import path
from . import views
from .views import (
    PostListView ,
    PostDetailView,
    PostCreateView,
    PostUpdateView,
    PostDeleteView,


)
urlpatterns=[
    path('',views.index,name='index'),
    path('shop/',views.shop,name='shop'),
    path('blog/',PostListView.as_view(),name='blog'),
    path('blog/post/<int:pk>/',PostDetailView.as_view(),name='post-detail'),
    path('blog/post/new/', PostCreateView.as_view(), name='post-create'),
    path('blog/post/<int:pk>/update/', PostUpdateView.as_view(), name='post-update'),
    path('blog/post/<int:pk>/delete/', PostDeleteView.as_view(), name='post-delete'),
]