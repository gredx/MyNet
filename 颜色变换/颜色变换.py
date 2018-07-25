import numpy as np
import matplotlib.pyplot as plt
def plti(image):
    plt.imshow(image)
    plt.show()

# assert( np.log(np.e) == 1.0)
# np.log 即 ln() - 以e为底的对数函数

def do_normalise(im):
    return -np.log(1 / ((1 + im) / 257) - 1)


# 预处理函数
# im中的像素值为 [0, 255] 闭区间， 则 (1+im) 为 [1, 256]
# 先做 (1+im)/257 操作将值归一化到 (0, 1) 开区间内
# 再使用 sigmoid函数 的反函数，效果见sigmod函数图像
# -np.log(1/((1 + 0)/257) - 1) = -5.5451774444795623
# -np.log(1/((1 + 255)/257) - 1) = 5.5451774444795623

def undo_normalise(im):
    return (1 / (np.exp(-im) + 1) * 257 - 1).astype("uint8")


# 预处理函数的反函数
# 即先使用sigmod函数，再将值变换到(0, 257)区间再减1，通过astype保证值位于[0, 255]
# 关于 astype("uint8") ：
# np.array([-1]).astype("uint8") = array([255], dtype=uint8)
# np.array([256]).astype("uint8") = array([0], dtype=uint8)

def rotation_matrix(theta):
    """
    3D 旋转矩阵，围绕X轴旋转theta角
    """
    return np.c_[
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ]


# np.c_[ ] 将列表中的元素在第二维上拼接起来
# np.c_[[1,2],[3,4],[5,6]] =
# array([[1, 3, 5],
#        [2, 4, 6]])

im = plt.imread('F:\\liuyang\\Mynet\data\\trainData\\1.jpg')
print(im.shape)
plti(im)
im_normed = do_normalise(im)
a=[]
for i in range(10):
    a.append(i*0.03)
for degree in a:
    im_rotated = np.einsum("ijk,lk->ijl", im_normed, rotation_matrix(np.pi*degree))
# 利用爱因斯坦求和约定做矩阵乘法，实际上是将每个RGB像素点表示的三维空间点绕X轴（即红色通道轴）旋转180°。
    im2 = undo_normalise(im_rotated)

    plti(im2)
