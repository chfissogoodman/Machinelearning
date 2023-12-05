import numpy as np
from utils.features import  prepare_for_training

class LinearRegression:
    """
    1.对数据进行预处理
    2.先得到所有的特征个数
    3.初始化参数矩阵
    """
    def __init__(self, data, labels,polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
        (data_processed,
         features_mean,
         features_deviation)=prepare_for_training(data, polynomial_degree=0, sinusoid_degree=0, normalize_data=True)
        self.data=data_processed
        self.labels=labels
        self.features_mean=features_mean
        self.features_deviation=features_deviation
        self.polynomial_degree=polynomial_degree
        self.sinusoid_degree=sinusoid_degree
        self.normalize_data=normalize_data

        num_features=self.data.shape[1]
        self.theta=np.zeros((num_features,1))
    def train(self,alpha, num_iterations=500):

    def gradient_descent(self, alpha, num_iterations=500):
        cost_history=[]
        for i in range(num_iterations):
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data,self.labels))

        return self.theta
    def gradient_step(self,alpha):
        """
        梯度下降算法
        """
        num_examples=self.data.shape[0]
        prediction=LinearRegression.hypothesis(self.data,self.theta)
        delta=prediction-self.labels
        theta=self.theta
        theta=theta-alpha*(1/num_examples)*(np.dot(delta.T,self.data)).T
        self.theta=theta
    def cost_function(self,data,labels):
        num_examples=data.shape[0]
        delta=LinearRegression.hypothesis(self.data,self.theta)-labels
        cost

    @staticmethod
    def hypothesis(data,theta):
        prediction=np.dot(data,theta)
        return prediction