# Ref: https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-cifar-10-small-object-photographs-from-scratch/

from matplotlib import pyplot
from keras.utils.vis_utils import plot_model
from cifar_lib import define_discriminator, load_real_samples, generate_real_samples\
	,train_discriminator, define_generator, generate_fake_samples\
	,define_gan, save_plot, summarize_performance, train

# define model
model = define_discriminator()

# SUMMARIZE THE MODEL
# model.summary()

# PLOT THE MODEL
# plot_model(model, to_file='discriminator_plot.png', show_shapes = True, show_layer_names = True)

##############################################################

# load and prepare cifar10 training images
X = load_real_samples() # Dataset
#X.shape[0]
#print(X.min(),X.max())

## DISCRIMINATOR ############################################################

# define the discriminator model
model = define_discriminator()
# load image data
dataset = load_real_samples()

X_real, y_real = generate_real_samples(dataset, 10)

# fit the model
train_discriminator(model, dataset)

## GENERATOR ############################################################

# define the size of the latent space
latent_dim = 100
# define the generator model
model = define_generator(latent_dim)
# summarize the model
#model.summary()
# plot the model
#plot_model(model, to_file='generator_plot.png', show_shapes=True, show_layer_names=True)

# GENERATE SAMPLES
n_samples = 49
X, _ = generate_fake_samples(model, latent_dim, n_samples)
# X.shape # (49, 32, 32, 3)

# scale pixel values from [-1,1] to [0,1]
X = (X + 1) / 2.0
X.min(), X.max()

# PLOT the generated samples
pyplot.figure(figsize=(10,10))
for i in range(n_samples):
	# define subplot
	pyplot.subplot(7, 7, 1 + i)
	# turn off axis labels
	pyplot.axis('off')
	# plot single image
	pyplot.imshow(X[i])
# show the figure
pyplot.show()

######################################################
# size of the latent space
latent_dim = 100
# create the discriminator
d_model = define_discriminator()

# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
# gan_model.summary()
# load image data
dataset = load_real_samples()
# train model
train(g_model, d_model, gan_model, dataset, latent_dim)
