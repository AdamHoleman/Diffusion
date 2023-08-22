import time
from collections import defaultdict
import torch
from torch.utils.data.dataloader import DataLoader
from diffusion import diff_model, diffusion_process



#--------------------------------------------------------- Adapted from the MinGPT github --------------------------------------------------------------------------

class diffusion_trainer():
  @staticmethod
  def get_default_config():

    config = {
        'device' : 'auto', #device to train on
        'num_workers' : 2, #number of workers


        #diffusion parameters
        'timesteps' : 1000,
        'beta_0' : .0001,
        'beta_T' : .02,

        # optimizer parameters
        'batch_size' : 64,
        'learning_rate' : 5e-5,
        'betas' : (0.9, 0.99),
        'weight_decay' : 0,
        'grad_norm_clip' : 1.0,
        }

    return config


  def __init__(self, config, model, trainset):

    self.config = config
    self.model = model
    self.optimizer = None
    self.train_dataset = trainset


    # determine the device we'll train on
    if config['device'] == 'auto':
      self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
      self.device = config.device

    self.model = self.model.to(self.device)
    print("running on device", self.device)


    # variables that will be assigned to trainer class later for logging
    self.iter_num = 0
    self.iter_time = 0.0
    self.iter_dt = 0.0



  def run(self, epochs = 100, printing = True):
    model, config, device = self.model, self.config, self.device

    # setup the optimizer
    self.optimizer = torch.optim.Adam(
        	model.parameters(),
        	lr = config['learning_rate'],
        	betas = config['betas'])


    # setup the dataloader
    trainloader = DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
        )


    #initialize a diffusion model with the given network as the denoiser
    dif_proc = diffusion_process(config['timesteps'], config['beta_0'], config['beta_T'])
    dif_mod = diff_model(dif_proc, model, self.device)


    tot_loss = torch.zeros(epochs).to(device)
    dif_mod.denoiser.train()
    mse = torch.nn.MSELoss()

    start_time = time.time()

    for i in range(epochs):
    	for batch, _ in trainloader:

    	  #Follow Algorithm 1 from Ho et. al
         batch_size = batch.size(0)

      	#sample timestep and feed the image into the forward process
         t = torch.randint(1, dif_mod.T, (batch_size,1))
         x_t, eps = dif_mod.dif_proc.add_noise(batch, t)

      	#move everything to the device to train the network.
         t = t.to(device)
         x_t = x_t.to(device)
         eps = eps.to(device)

         self.optimizer.zero_grad()
         loss = mse(eps, dif_mod.denoiser(x_t, t))
         loss.backward()
         self.optimizer.step()
         tot_loss[i] += loss.detach()


         self.iter_num += 1
         tnow = time.time()
         self.iter_dt = tnow - self.iter_time
         self.iter_time = tnow

         if printing:
           if self.iter_num % 100 == 0:
             print(f"iter_dt {self.iter_dt * 1000:.2f}ms; iter {self.iter_num}: train loss {loss.item():.5f}")






