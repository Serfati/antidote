# -*- encoding: utf-8 -*-


from django.contrib import admin
from django.urls import path, include  # add this

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('processdata.urls')),
    path('', include('app.urls')),  # add this
]
