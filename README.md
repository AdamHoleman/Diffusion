# A Basic Diffusion Model

This is a simple implementation of a denoising diffusion model following the paper 'Denoising Diffusion Probabilistic Models' by Jonathan Ho, Ajay Jain, and Pieter Abbeel. It's all written in PyTorch, and is more or less from scratch. This was purely an exercise for me to get my hands dirty with generative modeling, so I didn't go out of my way to produce super high-quality samples.

There are three main scripts: modules.py, diffusion.py, and trainer.py. The modules script is my implementation of the UNET architecture used for the denoising process, the diffusion script collects a handful of utilities (mainly used in computing the loss function and generating new samples), and the trainer is my loop for training the model.

The jupyter notebook is a brief demo on the CIFAR-10 dataset. To speed up training, I selected a single class (deer) from the dataset and trained the model for 1000 epochs on this subset to demonstrate that the model actually works. 



# Acknowledgements

This is an offshoot of a class project for Comp_Sci 449 at Northwestern, taken in the Spring of 2023. In that project, I worked with Jason Huang and Divy Kumar to compare existing implementations of diffusion models and GANs on CIFAR-10, and I certainly benefitted from conversations with them. That being said, all the code here is my own.

A deer that looks like a moose?

![SingleDeerSmall](https://github.com/AdamHoleman/Diffusion/assets/121455892/8d3c24c5-a646-4b36-81f1-f87fe87dfdbd)
