
print(__doc__)

# Author: Gael Varoquaux

# License: BSD 3 clause

# 导入绘图包matplotlib

import matplotlib.pyplot as plt

# 导入数据集、分类器及性能评估器

from sklearn import datasets, svm, metrics

# The digits dataset

digits = datasets.load_digits()

# 我们感兴趣的数据是由8x8的数字图像组成的，让我们

# 看一下存储在数据集的"images"属性中的前4张图像。处

# 理图像文件，则可以使用matplotlib.pyplot.imread加载它们。

# 请注意，每个图像必须具有相同的大小。对于这些图像，我们知道它们代表哪个数字：它在数据集的“target”中给出。

images_and_labels = list(zip(digits.images, digits.target))

index: int
for index, (image, label) in enumerate(images_and_labels[:4]):

 plt.subplot(2, 4, index + 1)

plt.axis('off')

plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')

plt.title('Training: %i' % label)

# 要对该数据应用分类器，我们需要将图像展平，以将数据转换为（样本，特征）矩阵：

n_samples = len(digits.images)

data = digits.images.reshape((n_samples, -1))

# 创建分类器: 一个支持向量机分类器

classifier = svm.SVC(gamma=0.001)

# 我们使用数据集的前半部分学习数字识别模型

classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

# 预测数据集的剩下部分的数字:

expected = digits.target[n_samples // 2:]

predicted = classifier.predict(data[n_samples // 2:])

print("Classification report for classifier %s:\n%s\n"

% (classifier, metrics.classification_report(expected, predicted)))

print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))

for index, (image, prediction) in enumerate(images_and_predictions[:4]):

 plt.subplot(2, 4, index + 5)

plt.axis('off')

plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')

plt.title('Prediction: %i' % prediction)

plt.show()