## 关于图像处理的神经网络和python语言的基本知识点

#### U-net
- U-net是用于医学影像分割很成熟的FCN网络


#### 人脸识别
- 使用了opencv 和 opencv官方github的参数集，对指定图片进行人脸识别
- 测试结果：对于真人脸部的识别比较准确，基本都能找到脸部的位置；但对于动漫、CG中人物的脸部识别完全失败。应用很局限。官方的参数集较弱

#### 零基础入门深度学习
- 来自[零基础入门深度学习](https://www.zybuluo.com/hanbingtao/note/433855)的示例代码（不全）作者讲的都是神经网络的基础概念和底层实现，建议初学者看

#### dirOP
- 实现了两个方法：查找指定目录下所有文件名，查找指定目录下所有指定格式的文件

#### 颜色变换
- 能实现对图片加滤镜的效果，实现原理是对颜色矩阵(旋转)把色调整体都变换

#### 图片打包
- 把目录下的图片裁剪为统一大小，转为numpy.array格式，打包成npy格式，以便深度学习读数据
- 同时包含将打包好的npy图片转为原始图片的方法

```c++
#include <iostream>
int main()
{
  return 0;
}
```
