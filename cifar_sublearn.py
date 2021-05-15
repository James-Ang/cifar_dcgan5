# example of loading the cifar10 dataset

from keras.datasets.cifar10 import load_data
from matplotlib import pyplot
# load the images into memory
(trainX, trainy), (testX, testy) = load_data()
# summarize the shape of the dataset
print('Train', trainX.shape, trainy.shape)
print('Test', testX.shape, testy.shape)

train_images_float = trainX.astype('float32')
trainX[0].min(), trainX.max()
train_images_float[0].min(), train_images_float.max()

pyplot.imshow(train_images_float[0].astype('uint8'))

# plot images from the training dataset
for i in range(49):
	# define subplot
	pyplot.subplot(7, 7, 1 + i)
	# turn off axis
	pyplot.axis('off')
	# plot raw pixel data
	pyplot.imshow(trainX[i])
pyplot.show()
#######################################
from numpy.random import randint
x = randint(1,5)
print(x)
#######################################
from numpy import ones
y = ones((10, 1))
print(y)

#######################################
from numpy.random import randn
x = randn(3,4)
x.reshape(3,4)
#######################################

#######################################

#######################################

#######################################
