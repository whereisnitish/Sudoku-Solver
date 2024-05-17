from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .read_image import *
import datetime
import os
from django.conf import settings

# Create your views here.

def index(request):
    return render(request, 'index.html')

def read_img(request):
    if request.method == 'POST':
        current_time = datetime.datetime.now()
        img = request.FILES['image']
        time_string = current_time.strftime("%Y%m%d%H%M%S")
        file_name, file_extension = os.path.splitext(img.name)

        img.name = f"{file_name}_{time_string}{file_extension}"


        fs = FileSystemStorage(location=settings.MEDIA_ROOT)
        fs.save(img.name, img)
        img_path = fs.url(img.name)
        
        result_path = main(img_path[1:])
        url = request.build_absolute_uri('/')
        print(url)
        data = {
            'img_path': url + img_path.replace('/media/', 'media/'),
            'result_path': url + f'{result_path}{file_extension}'
        }
        print(data)
        return render(request, 'result.html', data)