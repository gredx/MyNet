# 使用的是opencv训练好的参数 haarcascade_frontalface_default.xml
# 数据的识别范围很局限，动漫、CG人脸都识别不出来，
# 参数的设定对识别效果影响很大
import sys
from imp import  reload
import matplotlib.pyplot as plt
reload(sys)
#    __author__ = '郭 璞'

#    __date__ = '2016/9/5'

#    __Desc__ = 人脸检测小例子，以圆圈圈出人脸

import cv2

# 待检测的图片路径

imagepath = 'D:\\PythonCode\\MyNet\\2.jpg'

# 获取训练好的人脸的参数数据，这里直接从GitHub上使用默认值

face_cascade = cv2.CascadeClassifier(r'./haarcascade_frontalface_default.xml')

# 读取图片

image = cv2.imread(imagepath)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 探测图片中的人脸

faces = face_cascade.detectMultiScale(

    gray,

    scaleFactor=1.1,

    minNeighbors=4,

    minSize=(32, 32),

   # flags=cv2.CV_FEATURE_PARAMS_HOG

)

print("发现{0}个人脸!".format(len(faces)))


for (x, y, w, h) in faces:
    # cv2.rectangle(image,(x,y),(x+w,y+w),(0,255,0),2)

    cv2.circle(image, ((x + x + w) // 2, (y + y + h) // 2), w // 2, (0, 255, 0), 1)
plt.axis('off')
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.show()
#cv2.imshow("Find Faces!", image)

#cv2.waitKey(0)