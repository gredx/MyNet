'''
对数据要打包为npy格式
装载数据，reshape，处理成标准结构的数据，使用数据
打包：将一个文件夹中的所有图片打包
    1.获取文件夹中所有图片的列表
    2.读取所有图片，打包
'''
import os
import numpy as np
from skimage import io,data
import matplotlib.pyplot as plt
import cv2
array=[]
for i in range(10):
    array.append(i)
# np.save('1_10',array)
array2 = np.load('1_10.npy')
print(array2)

# print(dir())
'创建目录'
# os.mkdir("newdir")
'改变当前目录'
# os.chdir('D:\\PythonCode\\MyNet\\newdir')

'得到当前工作目录'
# print(os.getcwd())

'得到指定目录下的所有文件(夹)的名字'
dir_list = os.listdir()
# print(dir_list)

# 接下来我要把G:\面码  下 面码的所有图片打包为npy格式
# 相应的文件名也打包
path = 'G:\面码'
os.chdir(path)
list_dir = os.listdir(path)


for filename in list_dir:
    img = io.imread(filename)
    plt.axis('off')
    plt.imshow(img)
    plt.show()
filename = 'D:\\PythonCode\\MyNet\\1.jpg'
image = cv2.imread(filename)
plt.axis('off')
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.show()
