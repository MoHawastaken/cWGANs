# cWGANs

This is a paper in progress. I recommend to stop by in a few days...


We will document the used code in this repository.

Note that this code is not yet cleaned up!

We use Pytorch models.
A short explanation of the features of wgan_base:

- get_cdataloader(data, y, BATCH_SIZE=64, tt_split = 1, shuffle = True):

  data: the data points X (in the paper), should have shape (num_samples, d)
  
  y: the to data corresponding conditional values, should have shape (num_samples, d_Y)
  
  tt_split: proportion of training to test data in data.
  
  If shuffle =FALSE, training data = data\[:int(np.ceil(num_samples\*tt_split),:\], test data = data\[int(np.ceil(num_samples\*tt_split):,:\]


- class WGAN(nn.Module), class cWGAN(nn.Module): contains a generator .G and a critic/discriminator .D

- class cWGANTrainer: initialize with  (model, trains, testset = None, gantype = 'wgangp', noise = 'normal'):

  model: A Pytorch-trainable wrapper class containing networks .G and .D
  
  trains: Training dataloader
  
  testset: Testset dataloader
  
  gantype: Only 'wgangp' implemented but easily extendable
  
  noise: only standard normal 'normal' and uniform 'unif' implemented, but easily extendable.

- cWGANTrainer method: train_estim(num_epochs = 1, penalty = 0.1, G_lr=1e-4, D_lr=1e-4, G_wd = 0, D_wd = 0, D_steps_standard=5, num_batches = 1, num_estims = 10):

  num_epochs: number of training epochs

  penalty: gradient penalty
  
  G_lr: learning rate of the generator
  
  D_lr: learning rate of the discriminator
  
  G_wd: L2-weight decay in the ADAM optimizer for the generator
  
  D_wd: L2-weight decay in the ADAM optimizer for the discriminator
  
  D_steps_standard: how many iterations to train the discriminator network for one generator iteration,
  
  num_batches: number of batches to sample from training and test set to compute the empirical optimal transport estimates
  
  num_estims: number of runs of the OT estimate to then compute the standard deviation
 
 - cWGANTrainer method: generate_images(y, num_outputs = 64): generate num_outputs images given y



- similar for classes WGAN and WGANTrainer. WGANTrainer supports more GAN variants and exclusive discriminator training...
