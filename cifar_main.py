# Ref: https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-cifar-10-small-object-photographs-from-scratch/

from matplotlib import pyplot
from keras.utils.vis_utils import plot_model
from cifar_lib import define_discriminator, load_real_samples, generate_real_samples\
	,train_discriminator, define_generator, generate_fake_samples\
	,define_gan, save_plot, summarize_performance, train

######################################################
# size of the latent space
latent_dim = 100

# create the discriminator
d_model = define_discriminator()
d_model.summary() # 522,497 Non-Trainable (swithed off)

# create the generator
g_model = define_generator(latent_dim)
g_model.summary() # 1,466,115

# create the gan
gan_model = define_gan(g_model, d_model)
# gan_model.summary()

# load image data
dataset = load_real_samples()
# dataset.shape[0]

# train model
train(g_model, d_model, gan_model, dataset, latent_dim,n_epochs=4)
