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
pyplot.figure(figsize=(10,10))
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
from numpy import trace
trace([[1,3,3],[2,4,6],[4,5,6]]) # the sum of the elements along the main diagonal of the square matrix.

#######################################
import numpy
from cifar_lib import calculate_fid1
from numpy import cov
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm

act1 = random(10*2048)
act1 = act1.reshape((10,2048)) # 10 samples, 2048 features
act1.shape
act2 = random(10*2048)
act2 = act2.reshape((10,2048))
act2.shape

mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
mu1.shape, sigma1.shape

ssdiff = numpy.sum((mu1 - mu2)**2.0)

# calculate sqrt of product between cov
covmean = sqrtm(sigma1.dot(sigma2))
# check and correct imaginary numbers from sqrt
if iscomplexobj(covmean):
	covmean = covmean.real
# calculate score
fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)

#######################################
# define two collections of activations
act1 = random(10000*2048)
act1 = act1.reshape((10000,2048))
act2 = random(10000*2048)
act2 = act2.reshape((10000,2048))
# fid between act1 and act1
fid1 = calculate_fid1(act1, act1)
print('FID (same): %.3f' % fid1)
# fid between act1 and act2
fid2 = calculate_fid1(act1, act2)
print('FID (different): %.3f' % fid2)
# increasing samples form 10 --> 100 --> 1000 --> 10000
# FID = 354 --> 280 --> 140 --> 17.5 gets smaller when more samples are used.
#######################################

from numpy.random import randint
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize
from cifar_lib import scale_images, calculate_fid


# prepare the inception v3 model
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

# define two fake collections of images
images1 = randint(0, 255, 10*32*32*3)
images1 = images1.reshape((10,32,32,3))
# images1.min(), images1.max()
images2 = randint(0, 255, 10*32*32*3)
images2 = images2.reshape((10,32,32,3))
# images2.min(), images2.max()

# convert integer to floating point values
images1 = images1.astype('float32')
images2 = images2.astype('float32')
# resize images
images1 = scale_images(images1, (299,299,3))
images2 = scale_images(images2, (299,299,3))
print('Scaled', images1.shape, images2.shape)

# pre-process images
images1 = preprocess_input(images1) # convert (0-255) to (-1,1)
images2 = preprocess_input(images2)
# fid between images1 and images1
fid1 = calculate_fid(model, images1, images1)
print('FID (same): %.3f' % fid1)
# fid between images1 and images2
fid2 = calculate_fid(model, images1, images2)
print('FID (different): %.3f' % fid2)
#######################################
# CALCULATE FID ON REAL IMAGES
from numpy.random import shuffle

(images1, _), (images2, _) = load_data()
shuffle(images1)
images1 = images1[:10000]
print('Loaded', images1.shape, images2.shape)
# convert integer to floating point values
images1 = images1.astype('float32')
images2 = images2.astype('float32')
# resize images
images1 = scale_images(images1, (299,299,3))
images2 = scale_images(images2, (299,299,3))
bnmprint('Scaled', images1.shape, images2.shape)
# pre-process images
images1 = preprocess_input(images1)
images2 = preprocess_input(images2)
# calculate fid
fid = calculate_fid(model, images1, images2)
print('FID: %.3f' % fid)
#######################################
import numpy
arr=numpy.arange(10)

shuffle(arr)

#######################################


#######################################

#######################################

#######################################
