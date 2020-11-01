# cWGANs

We use Pytorch models.
A short explanation of the features of wgan_base:

- class WGAN(nn.Module), class cWGAN(nn.Module): contains a generator .G and a critic/discriminator .D

- class cWGANTrainer: initialize with  (model, trains, traint = None, datas = None, datat = None,testset = None, gantype = 'wgangp', noise = 'normal'):
 - model: A Pytorch-trainable wrapper class containing networks .G and .D
 - trains: Training dataloader
 - testset: Testset dataloader
 - gantype: Only 'wgangp' implemented but easily extendable
 - noise: only standard normal 'normal' and uniform 'unif' implemented, but easily extendable.
