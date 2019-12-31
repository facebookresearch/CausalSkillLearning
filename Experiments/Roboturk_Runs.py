# Start pretraining roboturk.

# Debug run.
python Master.py --train=1 --setting=pretrain_sub --name=Rdebug --data=Roboturk --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --debug=1

# M46 for comparison
# python Master.py --train=1 --setting=pretrain_sub --name=M48 --data=MIME --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64

# Runs
python cluster_run.py --partition=learnfair --name=R1 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R1 --data=Roboturk --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64'

python cluster_run.py --partition=learnfair --name=R2 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R2 --data=Roboturk --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64'

python cluster_run.py --partition=learnfair --name=R3 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R3 --data=Roboturk --kl_weight=0.1 --var_skill_length=1 --z_dimensions=64'

python cluster_run.py --partition=learnfair --name=R4 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R4 --data=Roboturk --kl_weight=1. --var_skill_length=1 --z_dimensions=64'

python cluster_run.py --partition=learnfair --name=R5 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R5 --data=Roboturk --action_scale_factor=10 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64'

python cluster_run.py --partition=learnfair --name=R6 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R6 --data=Roboturk --action_scale_factor=10 --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64'

python cluster_run.py --partition=learnfair --name=R7 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R7 --data=Roboturk --action_scale_factor=10 --kl_weight=0.1 --var_skill_length=1 --z_dimensions=64'

python cluster_run.py --partition=learnfair --name=R8 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R8 --data=Roboturk --action_scale_factor=10 --kl_weight=1. --var_skill_length=1 --z_dimensions=64'

python cluster_run.py --partition=learnfair --name=R9 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R9 --data=Roboturk --action_scale_factor=100 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64'

python cluster_run.py --partition=learnfair --name=R10 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R10 --data=Roboturk --action_scale_factor=100 --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64'

python cluster_run.py --partition=learnfair --name=R11 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R11 --data=Roboturk --action_scale_factor=100 --kl_weight=0.1 --var_skill_length=1 --z_dimensions=64'

python cluster_run.py --partition=learnfair --name=R12 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R12 --data=Roboturk --action_scale_factor=100 --kl_weight=1. --var_skill_length=1 --z_dimensions=64'

# Eval
python cluster_run.py --partition=learnfair --name=R1 --cmd='python Master.py --train=0 --setting=pretrain_sub --name=R1 --data=Roboturk --model=Experiment_Logs/R1/saved_models/Model_epoch199 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64'

python cluster_run.py --partition=learnfair --name=R2 --cmd='python Master.py --train=0 --setting=pretrain_sub --name=R2 --data=Roboturk --model=Experiment_Logs/R2/saved_models/Model_epoch199 --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64'

python cluster_run.py --partition=learnfair --name=R3 --cmd='python Master.py --train=0 --setting=pretrain_sub --name=R3 --data=Roboturk --model=Experiment_Logs/R3/saved_models/Model_epoch199 --kl_weight=0.1 --var_skill_length=1 --z_dimensions=64'

python cluster_run.py --partition=learnfair --name=R4 --cmd='python Master.py --train=0 --setting=pretrain_sub --name=R4 --data=Roboturk --model=Experiment_Logs/R4/saved_models/Model_epoch199 --kl_weight=1. --var_skill_length=1 --z_dimensions=64'

python cluster_run.py --partition=learnfair --name=R5 --cmd='python Master.py --train=0 --setting=pretrain_sub --name=R5 --data=Roboturk --model=Experiment_Logs/R5/saved_models/Model_epoch199 --action_scale_factor=10 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64'

python cluster_run.py --partition=learnfair --name=R6 --cmd='python Master.py --train=0 --setting=pretrain_sub --name=R6 --data=Roboturk --model=Experiment_Logs/R6/saved_models/Model_epoch199 --action_scale_factor=10 --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64'

python cluster_run.py --partition=learnfair --name=R7 --cmd='python Master.py --train=0 --setting=pretrain_sub --name=R7 --data=Roboturk --model=Experiment_Logs/R7/saved_models/Model_epoch199 --action_scale_factor=10 --kl_weight=0.1 --var_skill_length=1 --z_dimensions=64'

python cluster_run.py --partition=learnfair --name=R8 --cmd='python Master.py --train=0 --setting=pretrain_sub --name=R8 --data=Roboturk --model=Experiment_Logs/R8/saved_models/Model_epoch199 --action_scale_factor=10 --kl_weight=1. --var_skill_length=1 --z_dimensions=64'

python cluster_run.py --partition=learnfair --name=R9 --cmd='python Master.py --train=0 --setting=pretrain_sub --name=R9 --data=Roboturk --model=Experiment_Logs/R9/saved_models/Model_epoch199 --action_scale_factor=100 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64'

python cluster_run.py --partition=learnfair --name=R10 --cmd='python Master.py --train=0 --setting=pretrain_sub --name=R10 --data=Roboturk --model=Experiment_Logs/R10/saved_models/Model_epoch199 --action_scale_factor=100 --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64'

python cluster_run.py --partition=learnfair --name=R11 --cmd='python Master.py --train=0 --setting=pretrain_sub --name=R11 --data=Roboturk --model=Experiment_Logs/R11/saved_models/Model_epoch199 --action_scale_factor=100 --kl_weight=0.1 --var_skill_length=1 --z_dimensions=64'

python cluster_run.py --partition=learnfair --name=R12 --cmd='python Master.py --train=0 --setting=pretrain_sub --name=R12 --data=Roboturk --model=Experiment_Logs/R12/saved_models/Model_epoch199 --action_scale_factor=100 --kl_weight=1. --var_skill_length=1 --z_dimensions=64'

# Runs with increased capacity
python cluster_run.py --partition=learnfair --name=R13 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R13 --data=Roboturk --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128'

python cluster_run.py --partition=learnfair --name=R14 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R14 --data=Roboturk --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128'

python cluster_run.py --partition=learnfair --name=R15 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R15 --data=Roboturk --kl_weight=0.1 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128'

python cluster_run.py --partition=learnfair --name=R16 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R16 --data=Roboturk --kl_weight=1. --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128'

# Eval
python cluster_run.py --partition=learnfair --name=R13 --cmd='python Master.py --train=0 --model=Experiment_Logs/R13/saved_models/Model_epoch199 --setting=pretrain_sub --name=R13 --data=Roboturk --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128'

python cluster_run.py --partition=learnfair --name=R14 --cmd='python Master.py --train=0 --model=Experiment_Logs/R14/saved_models/Model_epoch199 --setting=pretrain_sub --name=R14 --data=Roboturk --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128'

python cluster_run.py --partition=learnfair --name=R15 --cmd='python Master.py --train=0 --model=Experiment_Logs/R15/saved_models/Model_epoch199 --setting=pretrain_sub --name=R15 --data=Roboturk --kl_weight=0.1 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128'

python cluster_run.py --partition=learnfair --name=R16 --cmd='python Master.py --train=0 --model=Experiment_Logs/R16/saved_models/Model_epoch194 --setting=pretrain_sub --name=R16 --data=Roboturk --kl_weight=1. --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128'

# Repeat to test new implementation.
python Master.py --train=1 --setting=pretrain_sub --name=R17_rerun13 --data=Roboturk --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128
# Eval
python Master.py --train=0 --setting=pretrain_sub --name=R17_rerun13 --data=Roboturk --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --model=Experiment_Logs/R17_rerun13/saved_models/Model_epoch0

# Rerun with speedy gonzales runs. 
python cluster_run.py --partition=learnfair --name=R18 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R18 --data=Roboturk --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128'

python cluster_run.py --partition=learnfair --name=R19 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R19 --data=Roboturk --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128'

####################################################
####################################################
############ Joint training on roboturk ############
####################################################
####################################################

python Master.py --train=1 --setting=learntsub --name=RJ1_loadR13 --kl_weight=0.0001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=Roboturk --subpolicy_model=Experiment_Logs/R13/saved_models/Model_epoch199 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1 --number_layers=8 --hidden_size=128
python Master.py --train=1 --setting=learntsub --name=RJtrial_deb --kl_weight=0.0001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=Roboturk --subpolicy_model=Experiment_Logs/R13/saved_models/Model_epoch199 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1 --number_layers=8 --hidden_size=128 --debug=1

# Actually running joint training with roboturk.
python cluster_run.py --partition=learnfair --name=RJ1 --cmd='python Master.py --train=1 --setting=learntsub --name=RJ1 --kl_weight=0.0001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=Roboturk --subpolicy_model=Experiment_Logs/R13/saved_models/Model_epoch199 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1 --number_layers=8 --hidden_size=128'

python cluster_run.py --partition=learnfair --name=RJ2 --cmd='python Master.py --train=1 --setting=learntsub --name=RJ2 --kl_weight=0.0001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=Roboturk --subpolicy_model=Experiment_Logs/R14/saved_models/Model_epoch199 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1 --number_layers=8 --hidden_size=128'

python cluster_run.py --partition=learnfair --name=RJ3 --cmd='python Master.py --train=1 --setting=learntsub --name=RJ3 --kl_weight=0.001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=Roboturk --subpolicy_model=Experiment_Logs/R13/saved_models/Model_epoch199 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1 --number_layers=8 --hidden_size=128'

python cluster_run.py --partition=learnfair --name=RJ4 --cmd='python Master.py --train=1 --setting=learntsub --name=RJ4 --kl_weight=0.001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=Roboturk --subpolicy_model=Experiment_Logs/R14/saved_models/Model_epoch199 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1 --number_layers=8 --hidden_size=128'

####################################################
############## Downstream RL Training ##############
####################################################

python Master.py --train=1 --setting='downstreamRL' --name=RLdebug --number_layers=8 --hidden_size=128 --data=MIME --environment=SawyerLift --epsilon_over=1000 --z_dimensions=0

python Master.py --train=1 --setting='downstreamRL' --name=RLdebug_2 --number_layers=8 --hidden_size=128 --data=MIME --environment=SawyerLift --epsilon_over=1000 --z_dimensions=0 --display_freq=50