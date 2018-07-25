# 把指定文件夹的所有文件打包成npy格式
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as Image
import skimage.io as io
################################################################################
# 从指定目录下读取所有文件(图片)，打包到npy文件中
# 用matplotlib.pyplot 读取图片
path = 'F:\\liuyang\\Mynet\\data\\trainData'
list = os.listdir(path)

package = []
for filename in list:
    image = plt.imread(os.path.join(path,filename))
    image = image[:800,:1000,:] # 将图片裁剪为统一的大小
    array = np.array(image) # 转成numpy数组
    package.append(array)   # 这一步把所有图片都加载到内存中，不是好的实现方法
    # plt.imshow(image)
    # plt.show()

np.save('package.npy',package) # 保存
###################################################################################
# 把打包的图片还原为图片
# 用 matplotlib.imag.save保存图片
trainData = np.load('package.npy')
trainData = trainData.reshape((15,800,1000,3)) # reshape的参数必须和原来的大小一致
#   trainData:36,000,000 = 15 * 800 * 1000 * 3
newdir = 'F:\liuyang\Mynet\data\\upData'
os.chdir(newdir)
for i,image in enumerate(trainData):
    Image.imsave(str(i)+'.jpg',image)