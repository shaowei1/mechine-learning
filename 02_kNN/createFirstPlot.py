import whether_likeimport matplotlibimport numpy as npimport matplotlib.pyplot as pltfrom numpy import arrayfrom mpl_toolkits.mplot3d import Axes3Dfig = plt.figure()ax = fig.add_subplot(111)datingDataMat, datingLabels = whether_like.file2matrix('datingTestSet2.txt')# ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))# xmin, xmax, ymin, ymax = axis(list_arg)ax.axis([-2, 25, -0.2, 2.0])plt.xlabel('Percentage of Time Spent Playing Video Games')plt.ylabel('Liters of Ice Cream Consumed Per Week')plt.show()# fig = plt.figure()# ax = Axes3D(fig)# datingDataMat, datingLabels = whether_like.file2matrix('datingTestSet2.txt')# X = datingDataMat[:, 0]# Y = datingDataMat[:, 1]# X, Y = np.meshgrid(X, Y)# ax.plot_surface(X, Y, datingDataMat[:, 2])## plt.show()# ## from matplotlib import pyplot as plt# import numpy as np# from mpl_toolkits.mplot3d import Axes3D## fig = plt.figure()# ax = Axes3D(fig)# X = np.arange(-4, 4, 0.25)# Y = np.arange(-4, 4, 0.25)# X, Y = np.meshgrid(X, Y)# R = np.sqrt(X**2 + Y**2)# Z = np.sin(R)## # 具体函数方法可用 help(function) 查看，如：help(ax.plot_surface)# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')## plt.show()# from math import log## fig = plt.figure()# # x = np.linspace(0, 100, 1000)# x = np.arange(0.05, 3, 0.05)# y1 = [log(a, 2) for a in x]# plot1 = plt.plot(x, y1, '-g', label='log2(x)')# plt.legend(loc='lower right')# plt.show()