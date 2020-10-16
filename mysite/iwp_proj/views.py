from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from .models import items,Post
from django.views.generic import (
    ListView,
    DetailView,
    CreateView,
    UpdateView,
    DeleteView
)
# Create your views here.
def index(request):
    name=items.objects.all()
    template=loader.get_template('iwp_proj/index.html')
    context = {
        'name': name,
    }
    return HttpResponse(template.render(context, request))

def shop(request):
    name=items.objects.all()
    template=loader.get_template('iwp_proj/shop.html')
    context = {
        'name': name,
    }
    return HttpResponse(template.render(context, request))
def blog(request):
    template=loader.get_template('iwp_proj/blog.html')
    context = {
        'posts': Post.objects.all(),
    }
    return HttpResponse(template.render(context, request))


class PostListView(ListView):
    model = Post
    template_name = 'iwp_proj/blog.html'  # <app>/<model>_<viewtype>.html
    context_object_name = 'posts'
    ordering = ['-date_posted']

class PostDetailView(DetailView):
    model = Post

class PostCreateView(LoginRequiredMixin, CreateView):
    model = Post
    fields = ['title', 'content']

    def form_valid(self, form):
        form.instance.author = self.request.user
        return super().form_valid(form)

class PostUpdateView(LoginRequiredMixin, UserPassesTestMixin, UpdateView):
    model = Post
    fields = ['title', 'content']

    def form_valid(self, form):
        form.instance.author = self.request.user
        return super().form_valid(form)

    def test_func(self):
        post = self.get_object()
        if self.request.user == post.author:
            return True
        return False

class PostDeleteView(LoginRequiredMixin, UserPassesTestMixin, DeleteView):
    model = Post
    success_url = '/blog'

    def test_func(self):
        post = self.get_object()
        if self.request.user == post.author:
            return True
        return False