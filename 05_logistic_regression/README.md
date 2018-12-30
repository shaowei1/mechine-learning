Logistic回归的目的是寻找一个非线性函数Sigmoid的最佳拟合参数，求解过程可以由最优化算法来完成。


# 过度拟合(over fit):
  当数据特征过多，当数据样本太小

# 模型选择算法(自动选择和减少特征向量)
  减少过拟合的发生，舍弃一部分数据，同时舍弃了问题中的一些信息
  1. Reduce number of features
    Manually select which features to keep
    Model selection algorithm(later in course)
  2. Regularization
    - keep all features, but reduce magnitude/values of parameters (sita)j
    - works well when we have a lot of features, each of which contributes a bit to predicting y.
  
  