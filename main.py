from zipfile import ZipFile
import os
import urllib
import urllib.request

file = 'fashion_mnist_images.zip'
folder = 'fashion_mnist_images'
url = 'https://nnfs.io/datasets/fashion_mnist_images.zip'

if not os.path.isfile(file):
    print(f'Downloading {url} and saving as {file}')
    urllib.request.urlretrieve(url, file)
print('Unzipping images...')

with ZipFile(file) as zip_images:
    zip_images.extractall(folder)
print("Done!")
print("------------")
