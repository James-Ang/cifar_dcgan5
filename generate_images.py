# example of loading the generator model and generating images
from keras.models import load_model
from numpy.random import randn
from matplotlib import pyplot as plt
from cifar_lib import generate_latent_points, create_plot, scale_images, calculate_fid, save_generated_image, save_original_image

# LOAD TRAINED GENERATOR
model = load_model(r'C:\Users\User\Documents\virtual\cifar_dcgan5\result1\generator_model_200.h5')
print('LOAD TRAINED GENERATOR')

# INITIALISATION
latent_dim = 100
n_samples = 10000
print('INITIALISATION FOR GENERATOR')

# GENERATE LATENT POINTS
latent_points = generate_latent_points(latent_dim, n_samples)
print('GENERATE LATENT POINTS')


# GENERATE IMAGES FROM TRAINED GENERATOR
X = model.predict(latent_points)
print('GENERATE IMAGES FROM TRAINED GENERATOR')
# X.shape # (10000, 32, 32, 3)
# X.min(), X.max() # (-0.9999704, 0.99999994)

# scale from [-1,1] to [0,1]
# X = (X + 1) / 2.0
# X.min(), X.max() #(0.001031071, 0.9999937)
# X[0].shape

# SAVE GENERATED IMAGES
# save_generated_image(X)



# RESIZE IMAGES - WILL TAKE A LONG TIME HERE
images2 = scale_images(X, (299,299,3))
print('RESIZE IMAGES')
# images2.shape
# images2.min(), images2.max() #(-0.9999704, 0.99999994)
# pyplot.imshow(images2[0])

# plot the result
# create_plot(X, 10)
# create_plot(images2, 10)

## Calculate FID #########################################
# USING KERAS TO CALCULATE FID ON REAL IMAGES
from numpy.random import shuffle
from keras.datasets.cifar10 import load_data
from numpy.random import randint
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize

# LOAD INCEPTION MODEL
model_inception = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
print('LOAD INCEPTION MODEL')

# LOAD CIFAR DATA
(images1, _), (_, _) = load_data()
print('LOAD CIFAR DATA')
# dtype('uint8'), (0, 255), (50000, 32, 32, 3)

shuffle(images1)
images1 = images1[:n_samples]
# print('Loaded', images1.shape, images2.shape) # Loaded (10000, 28, 28) (10000, 28, 28)

# SAVE IMAGES
# save_original_image(images1)


# CONVERT UINT8 TO FLOATING POINT VALUE
images1 = images1.astype('float32')
print('CONVERT UINT8 TO FLOATING POINT VALUE')
# images2 = images2.astype('float32')
# images1.min(), images1.max() #(0.0, 255.0)


# RESIZE IMAGES to 299,299,3 (Inception input) from 32,32,3 - WILL TAKE A LONG TIME HERE
images1 = scale_images(images1, (299,299,3))
print('RESIZE CIFAR IMAGES')
# (10000, 299, 299, 3), (0.0, 255.0)
# images2 = scale_images(images2, (299,299,3))
# print('Scaled', images1.shape, images2.shape) # Scaled (10000, 299, 299, 3) (10000, 299, 299, 3)
# images1.min(), images1.max() # (0.0, 255.0)

# PRE-PROCESS IMAGES # convert (0-255) to (-1,1) #(10000, 299, 299, 3)
images1 = preprocess_input(images1)
print('PRE-PROCESS CIFAR IMAGES # convert (0-255) to (-1,1)')

# plot the result
# create_plot(images1, 10)

# calculate fid - TAKES A LONG TIME HERE
fid = calculate_fid(model_inception, images1, images2)
print('FID: %.3f' % fid) # FID: 34.880
