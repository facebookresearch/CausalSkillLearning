# Learning Robot Skills with Temporal Variational Inference [![Python](https://img.shields.io/badge/Python-brightgreen.svg)](https://www.python.org)

### Research ###

This repository has code for our ICML 2020 paper on [Learning Robot Skills with Temporal Variational Inference](https://proceedings.icml.cc/static/paper_files/icml/2020/2847-Paper.pdf), authored by Tanmay Shankar and Abhinav Gupta.

[PDF](https://arxiv.org/pdf/2006.16232.pdf)

### Description ###

Our paper presents a way to jointly learn robot skills by placing low-level control policies on higher skills and how to use them from demonstrations in an unsupervised manner. 
The code implements the training procedure for this across 3 different datasets, and provides tools to visualize the learnt skills.

### Referencing ###

If you would like to use our code, please cite our paper and this repository in your work.

@misc{shankar2020learning,
      title={Learning Robot Skills with Temporal Variational Inference}, 
      author={Tanmay Shankar and Abhinav Gupta},
      year={2020},
      eprint={2006.16232},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

Also, be aware of the license for this repository: the Creative Commons Attribution-NonCommercial 4.0 International. Details may be viewed in the License file. 

### Contributing ###

Feel free to mail Tanmay (tanmay.shankar@gmail.com), for help, suggestions, questions and feedback. You can create issues in the repository, if you feel like the problem is pertinent to others. 

You can also set pull requests using this guide:
https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request

### Set up repository ###

#### Dependencies ####
You will need a few packages to be able to run the code in this repository.
For Robotic environments, you will need to install Mujoco, Mujoco_Py, OpenAI Gym, and Robosuite. [Here](https://docs.google.com/document/d/1V6BJf4R-2TXKO_IEOII5rLJbGj0jrJPptjtBCfczPk8/edit?usp=sharing) is a list of instructions on how to set these up. 

You will also need some standard deep learning packages, Pytorch, Tensorflow, Tensorboard, and TensorboardX. Usually you can just pip install these packages.

Upgrade pip:
```
pip install --upgrade pip
```
Pytorch: 
```
pip3 install torch torchvision
```
Tensorflow: 
```
pip install tensorflow
```
Tensorboard: 
```
pip install tensorboard
```
TensorboardX: 
```
pip install tensorboardX
```

We recommend using a virtual environment for them. 

#### Data ####
We run our model on various publicly available datasets, i.e. the [MIME dataset](https://sites.google.com/view/mimedataset), the [Roboturk dataset](https://sites.google.com/view/mimedataset), and the [CMU Mocap dataset](http://mocap.cs.cmu.edu/). In the case of the MIME and Roboturk datasets, we collate relevant data modalities and store them in quickly accessible formats for our code. You can find the links to these files below.

[MIME Dataset]()
[Roboturk Dataset]()
[CMU Mocap Dataset]()

Once you have downloaded this data locally, you will want to feed the path to these datasets in the `--dataset_directory` command line flag when you run your code.

### How to run the code ###

Here is a list of commands to run pre-training and joint skill learning on the various datasets used in our paper. The hyper-parameters specified here are used in our paper. Depending on your use case, you may want to play with these values. For a full list of the hyper-parameters, look at `Experiments/Master.py`. 

#### The MIME Dataset ####
For the MIME dataset, to run pre-training of the low-level policy: 

```
python Master.py --train=1 --setting=pretrain_sub --name=MIME_Pretraining --data=MIME --number_layers=8 --hidden_size=128 --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --normalization=meanvar
```
  
This should automatically run some evaluation and visualization tools every few epochs, and you can view the results in Experimental_Logs/<Run_Name>/. 
Once you've run this pre-training, you can run the joint training using: 

```
python Master.py --train=1 --setting=learntsub --name=J100 --normalization=meanvar --kl_weight=0.0001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/<MIME_Pretraining>/saved_models/Model_epoch480 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1 --training_phase_size=200000  
```

#### The Roboturk Dataset ####
For the Roboturk dataset, to run pre-training of the low-level policy: 

```
python Master.py --train=1 --setting=pretrain_sub --name=Roboturk_Pretraining --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128
```

Just as in the case of the MIME dataset, you can then run the joint training using: 

```
python Master.py --train=1 --setting=learntsub --name=RJ80 --latent_loss_weight=1. --latentpolicy_ratio=0.01 --kl_weight=0.0001 --subpolicy_ratio=0.1 --b_probability_factor=0.001 --data=Roboturk --subpolicy_model=Experiment_Logs/<Roboturk_Pretraining>/saved_models/Model_epoch20 --z_dimensions=64 --traj_length=-1 --var_skill_length=1 --number_layers=8 --hidden_size=128
```
Stay tuned for more! 
