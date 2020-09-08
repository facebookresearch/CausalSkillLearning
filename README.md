# Learning Robot Skills with Temporal Variational Inference

### What is this? ###

This repository has code for our ICML 2020 paper on [Learning Robot Skills with Temporal Variational Inference](https://proceedings.icml.cc/static/paper_files/icml/2020/2847-Paper.pdf), authored by Tanmay Shankar and Abhinav Gupta.

### I want a TL;DR of what this paper does. ###

Our paper presents a way to jointly learn robot skills and how to use them from demonstrations in an unsupervised manner. 
The code implements the training procedure for this across 3 different datasets, and provides tools to visualize the learnt skills.

### Cool. Can I use your code? ###

Yes! If you would like to use our code, please cite our paper and this repository in your work.
Also, be aware of the license for this repository: the Creative Commons Attribution-NonCommercial 4.0 International. Details may be viewed in the License file. 

### I need help, or I have brilliant ideas to make this code even better. ###

Great! Feel free to mail Tanmay (tanmay.shankar@gmail.com), for help, suggestions, questions and feedback. You can also create issues in the repository, if you feel like the problem is pertinent to others. 

### How do I set up this repository? ###

You will need a few packages to be able to run the code in this repository.
For Robotic environments, you will need to install Mujoco, Mujoco_Py, OpenAI Gym, and Robosuite. [Here](https://docs.google.com/document/d/1V6BJf4R-2TXKO_IEOII5rLJbGj0jrJPptjtBCfczPk8/edit?usp=sharing) is a list of instructions on how to set these up. 

You will also need some standard deep learning packages, Pytorch, Tensorflow, Tensorboard, and TensorboardX. Usually you can just pip install these packages. We recommend using a virtual environment for them. 

### Tell me how to run the code already! ###

We are compiling a list of commands to run pre-training and joint skill learning on the various datasets used in our paper. 
Stay tuned for more! 
