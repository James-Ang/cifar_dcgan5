# USING NUMPY TO CALCULATE FID
import numpy
from cifar_lib import calculate_fid1
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm

act1 = random(10*2048)
act1 = act1.reshape((10,2048)) # 10 samples, 2048 features, shape (10, 2048)

act2 = random(10*2048)
act2 = act2.reshape((10,2048)) # shape (10, 2048)


mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
# mu1.shape, sigma1.shape # ((2048,), (2048, 2048))

ssdiff = numpy.sum((mu1 - mu2)**2.0)

# calculate sqrt of product between cov
covmean = sqrtm(sigma1.dot(sigma2))
# check and correct imaginary numbers from sqrt
if iscomplexobj(covmean):
	covmean = covmean.real
# calculate score
fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)

# NUMPY FID - 10000 IMAGES ######################################
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

## USING KERAS to CALCULATE FID #####################################

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
# images1.min(), images1.max() #(0.0, 254.0)
images2 = randint(0, 255, 10*32*32*3)
images2 = images2.reshape((10,32,32,3))
# images2.min(), images2.max() #(0.0, 254.0)

# convert integer to floating point values
images1 = images1.astype('float32')
images2 = images2.astype('float32')

# RESIZE IMAGES to 299,299,3 from 32,32,3
images1 = scale_images(images1, (299,299,3))
# images1.min(), images1.max() #(0.0, 254.0)
images2 = scale_images(images2, (299,299,3))
# images2.min(), images2.max() #(0.0, 254.0)
print('Scaled', images1.shape, images2.shape) # Scaled (10, 299, 299, 3) (10, 299, 299, 3)

# pre-process images
images1 = preprocess_input(images1) # convert (0-255) to (-1,1)
images2 = preprocess_input(images2) # convert (0-255) to (-1,1)

# fid between images1 and images1
fid1 = calculate_fid(model, images1, images1)
print('FID (same): %.3f' % fid1) # FID (same): -0.000
# fid between images1 and images2
fid2 = calculate_fid(model, images1, images2)
print('FID (different): %.3f' % fid2) # FID (different): 41.376

#######################################
# USING KERAS TO CALCULATE FID ON REAL IMAGES
from numpy.random import shuffle

(images1, _), (images2, _) = load_data() # LOADING MNIST DATA
shuffle(images1)
images1 = images1[:10000]
print('Loaded', images1.shape, images2.shape) # Loaded (10000, 28, 28) (10000, 28, 28)

# convert integer to floating point values
images1 = images1.astype('float32')
images2 = images2.astype('float32')
# images1.min(), images1.max() #(0.0, 255.0)


# resize images - WILL TAKE A LONG TIME HERE
images1 = scale_images(images1, (299,299,3))
images2 = scale_images(images2, (299,299,3))
print('Scaled', images1.shape, images2.shape) # Scaled (10000, 299, 299, 3) (10000, 299, 299, 3)
# images1.min(), images1.max() # (0.0, 255.0)

# pre-process images
images1 = preprocess_input(images1)
images2 = preprocess_input(images2)

# calculate fid - TAKES A LONG TIME HERE
fid = calculate_fid(model, images1, images2)
print('FID: %.3f' % fid) # FID: 1.766
#######################################
import numpy
arr=numpy.arange(10)

shuffle(arr)

#######################################


#######################################

#######################################

#######################################
