from 感知器 import  Perceptron
def f(x):
    return x

# 一个使用线性函数的特定类
class LinerUnit(Perceptron):
    def __init__(self,input_num):
        Perceptron.__init__(self,input_num,f)

def get_train_dataset():
    '''
        捏造5个人的收入数据
        '''
    # 构建训练数据
    # 输入向量列表，每一项是工作年限
    input_vecs =  [[5], [3], [8], [1.4], [10.1]]
    # 期望的输出列表，月薪，注意要与输入一一对应
    labels = [5500, 2300, 7600, 1800, 11400]
    return input_vecs,labels

def train_linear_unit():
    # 创建感知器，设置特征数目为1个
    liUnit = LinerUnit(1)
    # 进行训练 , 迭代10轮，学习速率为0.01
    input_vecs , labels = get_train_dataset()
    liUnit.train(input_vecs,labels,10,0.01)
    return liUnit

if __name__ == '__main__':
    # 训练感知器，获得感知器
    linear_unit = train_linear_unit()
    # 输出感知器的参数
    print(linear_unit)
    # 测试
    print('Work 3.4 years, monthly salary = %.2f' % linear_unit.predict([3.4]))
    print('Work 15 years, monthly salary = %.2f' % linear_unit.predict([15]))
    print('Work 1.5 years, monthly salary = %.2f' % linear_unit.predict([1.5]))
    print('Work 6.3 years, monthly salary = %.2f' % linear_unit.predict([6.3]))
