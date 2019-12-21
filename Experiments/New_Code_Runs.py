# Training on Continuous Directed NonZero data with Reparam. 
python cluster_run.py --partition=learnfair --name=S180 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S180 --data=ContinuousDirNZ --kl_weight=0.1 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1'
# Eval 
python Master.py --train=0 --setting=pretrain_sub --name=S180_m5 --data=ContinuousDirNZ --kl_weight=0.1 --z_dimensions=8 --model=Experiment_Logs/S180/saved_models/Model_epoch5 &&
python Master.py --train=0 --setting=pretrain_sub --name=S180_m10 --data=ContinuousDirNZ --kl_weight=0.1 --z_dimensions=8 --model=Experiment_Logs/S180/saved_models/Model_epoch10 &&
python Master.py --train=0 --setting=pretrain_sub --name=S180_m15 --data=ContinuousDirNZ --kl_weight=0.1 --z_dimensions=8 --model=Experiment_Logs/S180/saved_models/Model_epoch15 &&
python Master.py --train=0 --setting=pretrain_sub --name=S180_m20 --data=ContinuousDirNZ --kl_weight=0.1 --z_dimensions=8 --model=Experiment_Logs/S180/saved_models/Model_epoch20 &&
python Master.py --train=0 --setting=pretrain_sub --name=S180_m25 --data=ContinuousDirNZ --kl_weight=0.1 --z_dimensions=8 --model=Experiment_Logs/S180/saved_models/Model_epoch25 && 
python Master.py --train=0 --setting=pretrain_sub --name=S180_m30 --data=ContinuousDirNZ --kl_weight=0.1 --z_dimensions=8 --model=Experiment_Logs/S180/saved_models/Model_epoch30 &&
python Master.py --train=0 --setting=pretrain_sub --name=S180_m35 --data=ContinuousDirNZ --kl_weight=0.1 --z_dimensions=8 --model=Experiment_Logs/S180/saved_models/Model_epoch35 &&
python Master.py --train=0 --setting=pretrain_sub --name=S180_m40 --data=ContinuousDirNZ --kl_weight=0.1 --z_dimensions=8 --model=Experiment_Logs/S180/saved_models/Model_epoch40


python cluster_run.py --partition=learnfair --name=S181 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S181 --data=ContinuousDirNZ --kl_weight=0.1 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1'
# Eval
python Master.py --train=0 --setting=pretrain_sub --name=S181_m5 --data=ContinuousDirNZ --kl_weight=0.1 --z_dimensions=64 --model=Experiment_Logs/S181/saved_models/Model_epoch5

#######################################################
# Training on Separable Data with Reparam with New Repo
#######################################################
python cluster_run.py --partition=learnfair --name=S200_Separable --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S200_Separable --entropy=0 --data=Separable --kl_weight=0.1 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1'
# Eval
python Master.py --train=0 --setting=pretrain_sub --name=S200_Separable_M5 --entropy=0 --data=Separable --kl_weight=0.1 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --model=Experiment_Logs/S200_Separable/saved_models/Model_epoch5 && python Master.py --train=0 --setting=pretrain_sub --name=S200_Separable_M10 --entropy=0 --data=Separable --kl_weight=0.1 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --model=Experiment_Logs/S200_Separable/saved_models/Model_epoch10 && python Master.py --train=0 --setting=pretrain_sub --name=S200_Separable_M15 --entropy=0 --data=Separable --kl_weight=0.1 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --model=Experiment_Logs/S200_Separable/saved_models/Model_epoch15 && python Master.py --train=0 --setting=pretrain_sub --name=S200_Separable_M20 --entropy=0 --data=Separable --kl_weight=0.1 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --model=Experiment_Logs/S200_Separable/saved_models/Model_epoch20 && python Master.py --train=0 --setting=pretrain_sub --name=S200_Separable_M25 --entropy=0 --data=Separable --kl_weight=0.1 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --model=Experiment_Logs/S200_Separable/saved_models/Model_epoch25 && python Master.py --train=0 --setting=pretrain_sub --name=S200_Separable_M30 --entropy=0 --data=Separable --kl_weight=0.1 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --model=Experiment_Logs/S200_Separable/saved_models/Model_epoch30

python cluster_run.py --partition=learnfair --name=S201_Separable --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S201_Separable --entropy=0 --data=Separable --kl_weight=0.1 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1'

python cluster_run.py --partition=learnfair --name=S203 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S203 --entropy=0 --data=ContinuousNonZero --kl_weight=0.1 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1'

# Eval
python Master.py --train=0 --setting=pretrain_sub --name=S203 --entropy=0 --data=ContinuousNonZero --kl_weight=0.1 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --model=Experiment_Logs/S203/saved_models/Model_epoch6

python cluster_run.py --partition=learnfair --name=S204 --cmd='python Master.py --train=1 --setting=oldpretrain_sub --name=S204 --entropy=0 --data=ContinuousNonZero --kl_weight=0.1 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1'
# Eval
python Master.py --train=0 --setting=oldpretrain_sub --name=S204 --entropy=0 --data=ContinuousNonZero --kl_weight=0.1 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --model=Experiment_Logs/S204/saved_models/Model_epoch6

python cluster_run.py --partition=learnfair --name=S205 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S205 --entropy=0 --data=ContinuousNonZero --kl_weight=0.01 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1'
# Eval
python Master.py --train=0 --setting=pretrain_sub --name=S205 --entropy=0 --data=ContinuousNonZero --kl_weight=0.01 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --model=Experiment_Logs/S205/saved_models/Model_epoch3
# 
python Master.py --train=0 --setting=pretrain_sub --name=S205_m9 --entropy=0 --data=ContinuousNonZero --kl_weight=0.01 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --model=Experiment_Logs/S205/saved_models/Model_epoch9
python Master.py --train=0 --setting=pretrain_sub --name=S205_m10 --entropy=0 --data=ContinuousNonZero --kl_weight=0.01 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --model=Experiment_Logs/S205/saved_models/Model_epoch10
python Master.py --train=0 --setting=pretrain_sub --name=S205_m15 --entropy=0 --data=ContinuousNonZero --kl_weight=0.01 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --model=Experiment_Logs/S205/saved_models/Model_epoch15 && python Master.py --train=0 --setting=pretrain_sub --name=S205_m20 --entropy=0 --data=ContinuousNonZero --kl_weight=0.01 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --model=Experiment_Logs/S205/saved_models/Model_epoch20 && python Master.py --train=0 --setting=pretrain_sub --name=S205_m25 --entropy=0 --data=ContinuousNonZero --kl_weight=0.01 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --model=Experiment_Logs/S205/saved_models/Model_epoch25

python cluster_run.py --partition=learnfair --name=S206 --cmd='python Master.py --train=1 --setting=oldpretrain_sub --name=S206 --entropy=0 --data=ContinuousNonZero --kl_weight=0.01 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1'
# Eval
python Master.py --train=0 --setting=oldpretrain_sub --name=S206 --entropy=0 --data=ContinuousNonZero --kl_weight=0.01 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --model=Experiment_Logs/S206/saved_models/Model_epoch3
# 
python Master.py --train=0 --setting=oldpretrain_sub --name=S206_m9 --entropy=0 --data=ContinuousNonZero --kl_weight=0.01 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --model=Experiment_Logs/S206/saved_models/Model_epoch9
python Master.py --train=0 --setting=oldpretrain_sub --name=S206_m10 --entropy=0 --data=ContinuousNonZero --kl_weight=0.01 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --model=Experiment_Logs/S206/saved_models/Model_epoch10
python Master.py --train=0 --setting=oldpretrain_sub --name=S206_m15 --entropy=0 --data=ContinuousNonZero --kl_weight=0.01 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --model=Experiment_Logs/S206/saved_models/Model_epoch15 && python Master.py --train=0 --setting=oldpretrain_sub --name=S206_m20 --entropy=0 --data=ContinuousNonZero --kl_weight=0.01 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --model=Experiment_Logs/S206/saved_models/Model_epoch20 && python Master.py --train=0 --setting=oldpretrain_sub --name=S206_m25 --entropy=0 --data=ContinuousNonZero --kl_weight=0.01 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --model=Experiment_Logs/S206/saved_models/Model_epoch25

python cluster_run.py --partition=learnfair --name=S207 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S207 --entropy=0 --data=ContinuousNonZero --kl_weight=0.5 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1'

python cluster_run.py --partition=learnfair --name=S208 --cmd='python Master.py --train=1 --setting=oldpretrain_sub --name=S208 --entropy=0 --data=ContinuousNonZero --kl_weight=0.5 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1'

# Pretrain with KL. 
python cluster_run.py --partition=learnfair --name=S209 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S209 --entropy=0 --data=Separable --kl_weight=0.01 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1'

python cluster_run.py --partition=learnfair --name=S210 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S210 --entropy=0 --data=Separable --kl_weight=0.01 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1'

#######################################################
#######################################################

# Using S170 model on ContinuousDirNZ data. These use the old loss.
python cluster_run.py --name=C300 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=20 --name=C300_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=ContinuousDirNZ --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1.'

python cluster_run.py --name=C301 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=20 --name=C301_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=ContinuousDirNZ --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1.'
# Rollout latent policy with samples.
python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=20 --name=C301_rollout --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=ContinuousDirNZ --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --model=Experiment_Logs/C301_loadS170/saved_models/Model_epoch19 --display_freq=100

# Running with S170 on DirNZ data with new latent loss.
python cluster_run.py --name=C302 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=20 --name=C302_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=ContinuousDirNZ --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1.'

python cluster_run.py --name=C303 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=20 --name=C303_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=ContinuousDirNZ --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1.'

# Running with S170 on DirNZ data with reduced training phase size.
python cluster_run.py --name=C304 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=20 --name=C304_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=ContinuousDirNZ --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --training_phase_size=100000'

python cluster_run.py --name=C305 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=20 --name=C305_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=ContinuousDirNZ --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --training_phase_size=100000'

# New code runs with goal directed dataset. 
# Debug
python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=20 --name=Cdebug --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=GoalDirected --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1.

python cluster_run.py --name=C306 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=20 --name=C306_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=GoalDirected --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1.'

python cluster_run.py --name=C307 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=20 --name=C307_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=GoalDirected --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1.'

# New code runs with goal directed dataset with training_phase_size reduced. 
python cluster_run.py --name=C308 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=20 --name=C308_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=GoalDirected --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --training_phase_size=100000'

python cluster_run.py --name=C309 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=20 --name=C309_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=GoalDirected --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --training_phase_size=100000'

# Rerun C306-C309 with full length.
python cluster_run.py --name=C310 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C310_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=GoalDirected --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1.'

python cluster_run.py --name=C311 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C311_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=GoalDirected --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1.'

python cluster_run.py --name=C312 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C312_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=GoalDirected --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --training_phase_size=100000'

python cluster_run.py --name=C313 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C313_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=GoalDirected --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --training_phase_size=100000'

# Rerun C310-C313 for repeatability, to see if previous runs visualizations were garbage or not, and to check that viz still works with new MIME stuff added.
python cluster_run.py --name=C314 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C314_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=GoalDirected --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1.'

python cluster_run.py --name=C315 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C315_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=GoalDirected --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1.'

python cluster_run.py --name=C316 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C316_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=GoalDirected --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --training_phase_size=100000'

python cluster_run.py --name=C317 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C317_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=GoalDirected --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --training_phase_size=100000'

# Rerun C314-C317 with correct conditional information representation supplied to the latent policy. 
python cluster_run.py --name=C318 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C318_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=GoalDirected --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --condition_size=4'

python cluster_run.py --name=C319 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C319_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=GoalDirected --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --condition_size=4'

python cluster_run.py --name=C320 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C320_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=GoalDirected --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --training_phase_size=100000 --condition_size=4'

python cluster_run.py --name=C321 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C321_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=GoalDirected --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --training_phase_size=100000 --condition_size=4'

# Now running on DEterministic Goal Directed. 
# DEbug
python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=Cdetergoal --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=DeterGoal --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --condition_size=4

python cluster_run.py --name=C322 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C322_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=DeterGoal --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --condition_size=4'

python cluster_run.py --name=C323 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C323_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=DeterGoal --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --condition_size=4'

python cluster_run.py --name=C324 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C324_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=DeterGoal --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --training_phase_size=100000 --condition_size=4'

python cluster_run.py --name=C325 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C325_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=DeterGoal --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --training_phase_size=100000 --condition_size=4'

# Try debugging the latent policy rollout. 
# DEbug
python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=Cdetergoal --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=DeterGoal --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --condition_size=4

# Rerun with latent policy loss weights. 
python cluster_run.py --name=C326 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C326_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=DeterGoal --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --condition_size=4'

python cluster_run.py --name=C327 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C327_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=DeterGoal --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --condition_size=4'

python cluster_run.py --name=C328 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C328_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=DeterGoal --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --training_phase_size=100000 --condition_size=4'

python cluster_run.py --name=C329 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C329_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=DeterGoal --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --training_phase_size=100000 --condition_size=4'

# Rerun with further decreased latent z loss weight.
python cluster_run.py --name=C330 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C330_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=DeterGoal --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --condition_size=4 --lat_z_wt=0.01'

python cluster_run.py --name=C331 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C331_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=DeterGoal --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --condition_size=4 --lat_z_wt=0.01'

python cluster_run.py --name=C332 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C332_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=DeterGoal --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --training_phase_size=100000 --condition_size=4 --lat_z_wt=0.01'

python cluster_run.py --name=C333 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C333_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=DeterGoal --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --training_phase_size=100000 --condition_size=4 --lat_z_wt=0.01'

# Run with new separable data. 
python cluster_run.py --name=C334 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C334_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=Separable --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --condition_size=4'

python cluster_run.py --name=C335 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C335_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=Separable --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --condition_size=4'

python cluster_run.py --name=C336 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C336_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=Separable --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --training_phase_size=100000 --condition_size=4'

python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C336_loadS170_debug --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=Separable --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --training_phase_size=100000 --condition_size=4 --model=Experiment_Logs/C336_loadS170/saved_models/Model_epoch25

python cluster_run.py --name=C337 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C337_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=Separable --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --training_phase_size=100000 --condition_size=4'

# Run with separable data, with conditional information being provided from the first timestep. Additionally, increasing latent_z loss weight for subsequent phases of training. 
python cluster_run.py --name=C338 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C338_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=Separable --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --condition_size=4'

python cluster_run.py --name=C339 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C339_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=Separable --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --condition_size=4'

python cluster_run.py --name=C340 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C340_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=Separable --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --training_phase_size=100000 --condition_size=4'

python cluster_run.py --name=C341 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C341_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=Separable --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --training_phase_size=100000 --condition_size=4'

# Trying out ... CausalSkillLearning Repo
python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=CCSL_T1 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=Separable --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --training_phase_size=100000 --condition_size=4

# Rerun C338-C341 with greedy selection in latent rollout.. which is what we should have been doing all this while. :O
python cluster_run.py --name=C342 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C342_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=Separable --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --condition_size=4'

python cluster_run.py --name=C343 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C343_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=Separable --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --condition_size=4'

python cluster_run.py --name=C344 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C344_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=Separable --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --training_phase_size=100000 --condition_size=4'

# Visualize every 1000 images. 
python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C344_loadS170_viz --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=Separable --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --training_phase_size=100000 --condition_size=4 --display_freq=1000 --model=Experiment_Logs/C344_loadS170/saved_models/Model_epoch19

python cluster_run.py --name=C345 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C345_loadS170 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=Separable --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S170/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --training_phase_size=100000 --condition_size=4'

# We've been training joint objective loading S170. Now we will try to load S205, S206, so that it's in distribution. 
python cluster_run.py --name=C346 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C346_loadS205_m10 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=Separable --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S205/saved_models/Model_epoch10 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --condition_size=4'

python cluster_run.py --name=C347 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C347_loadS205_m10 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=Separable --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S205/saved_models/Model_epoch10 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --condition_size=4'

python cluster_run.py --name=C348 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C348_loadS205_m10 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=Separable --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S205/saved_models/Model_epoch10 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --training_phase_size=100000 --condition_size=4'

python cluster_run.py --name=C349 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C349_loadS205_m10 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=Separable --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S205/saved_models/Model_epoch10 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --training_phase_size=100000 --condition_size=4'

# Running with S205_m15 model. # THESE HAVE ALSO BEEN RUN WITH NEW UPDATE POLICY FUNCTION
python cluster_run.py --name=C350 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C350_loadS205_m15 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=Separable --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S205/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --condition_size=4'

python cluster_run.py --name=C351 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C351_loadS205_m15 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=Separable --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S205/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --condition_size=4'

python cluster_run.py --name=C352 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C352_loadS205_m15 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=Separable --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S205/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --training_phase_size=100000 --condition_size=4'

python cluster_run.py --name=C353 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C353_loadS205_m15 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=Separable --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S205/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --training_phase_size=100000 --condition_size=4'

# Rerun C350-C353 with baseline and baseline_target as scalars. 
python cluster_run.py --name=C354 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C354_loadS205_m15 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=Separable --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S205/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --condition_size=4'

python cluster_run.py --name=C355 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C355_loadS205_m15 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=Separable --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S205/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --condition_size=4'

python cluster_run.py --name=C356 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C356_loadS205_m15 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=Separable --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S205/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --training_phase_size=100000 --condition_size=4'

python cluster_run.py --name=C357 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C357_loadS205_m15 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=Separable --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S205/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --training_phase_size=100000 --condition_size=4'

# Run C354-C357 again without clamps on either subpolicy or latent policy likelihoods. 
python cluster_run.py --name=C358 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C358_loadS205_m15 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=Separable --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S205/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --condition_size=4'

python cluster_run.py --name=C359 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C359_loadS205_m15 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=Separable --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S205/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --condition_size=4'

python cluster_run.py --name=C360 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C360_loadS205_m15 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=Separable --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S205/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --training_phase_size=100000 --condition_size=4'

python cluster_run.py --name=C361 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C361_loadS205_m15 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=Separable --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S205/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --training_phase_size=100000 --condition_size=4'

# Run C358-C361 with clamps on subpolicy but without clamps on latent policy. 
python cluster_run.py --name=C362 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C362_loadS205_m15 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=Separable --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S205/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --condition_size=4'

python cluster_run.py --name=C363 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C363_loadS205_m15 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=Separable --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S205/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --condition_size=4'

python cluster_run.py --name=C364 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C364_loadS205_m15 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=Separable --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S205/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --training_phase_size=100000 --condition_size=4'

python cluster_run.py --name=C365 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=-1 --name=C365_loadS205_m15 --ent_weight=0. --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0. --b_probability_factor=0.01 --min_variance_bias=0.01 --data=Separable --kl_weight=0.01 --epsilon_from=0.2 --epsilon_to=0.05 --epsilon_over=30 --fix_subpolicy=1 --var_loss_weight=1.0 --subpolicy_model=Experiment_Logs/S205/saved_models/Model_epoch15 --subpolicy_clamp_value=-5 --latent_loss_weight=1. --training_phase_size=100000 --condition_size=4'






###############################################################
################### NEW MIME PRETRAINING ######################
###############################################################
#Debug
python Master.py --train=1 --setting=pretrain_sub --name=M15_trial --entropy=0 --data=MIME --kl_weight=0.01 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar

python cluster_run.py --partition=learnfair --name=M15 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M15 --entropy=0 --data=MIME --kl_weight=0.01 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M16 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M16 --entropy=0 --data=MIME --kl_weight=0.1 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M17 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M17 --entropy=0 --data=MIME --kl_weight=0.5 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M18 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M18 --entropy=0 --data=MIME --kl_weight=1.0 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M19 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M19 --entropy=0 --data=MIME --kl_weight=0.01 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M20 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M20 --entropy=0 --data=MIME --kl_weight=0.1 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M21 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M21 --entropy=0 --data=MIME --kl_weight=0.5 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M22 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M22 --entropy=0 --data=MIME --kl_weight=1.0 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=minmax'

python Master.py --train=0 --setting=pretrain_sub --name=M15 --entropy=0 --data=MIME --kl_weight=0.01 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar --model=Experiment_Logs/M15/saved_models/Model_epoch85

# Eval at whatever was done 
python cluster_run.py --partition=learnfair --name=M15_eval --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M15 --entropy=0 --data=MIME --kl_weight=0.01 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar --model=Experiment_Logs/M15/saved_models/Model_epoch70'

python cluster_run.py --partition=learnfair --name=M16_eval --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M16 --entropy=0 --data=MIME --kl_weight=0.1 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar --model=Experiment_Logs/M16/saved_models/Model_epoch70'

python cluster_run.py --partition=learnfair --name=M17_eval --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M17 --entropy=0 --data=MIME --kl_weight=0.5 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar --model=Experiment_Logs/M17/saved_models/Model_epoch70'

python cluster_run.py --partition=learnfair --name=M18_eval --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M18 --entropy=0 --data=MIME --kl_weight=1.0 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar --model=Experiment_Logs/M18/saved_models/Model_epoch70'

python cluster_run.py --partition=learnfair --name=M19_eval --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M19 --entropy=0 --data=MIME --kl_weight=0.01 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=minmax --model=Experiment_Logs/M19/saved_models/Model_epoch70'

python cluster_run.py --partition=learnfair --name=M20_eval --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M20 --entropy=0 --data=MIME --kl_weight=0.1 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=minmax --model=Experiment_Logs/M20/saved_models/Model_epoch70'

python cluster_run.py --partition=learnfair --name=M21_eval --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M21 --entropy=0 --data=MIME --kl_weight=0.5 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=minmax --model=Experiment_Logs/M21/saved_models/Model_epoch70'

python cluster_run.py --partition=learnfair --name=M22_eval --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M22 --entropy=0 --data=MIME --kl_weight=1.0 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=minmax --model=Experiment_Logs/M22/saved_models/Model_epoch70'

# Eval at 190
python cluster_run.py --partition=learnfair --name=M15_eval --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M15 --entropy=0 --data=MIME --kl_weight=0.01 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar --model=Experiment_Logs/M15/saved_models/Model_epoch190'

python cluster_run.py --partition=learnfair --name=M16_eval --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M16 --entropy=0 --data=MIME --kl_weight=0.1 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar --model=Experiment_Logs/M16/saved_models/Model_epoch190'

python cluster_run.py --partition=learnfair --name=M17_eval --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M17 --entropy=0 --data=MIME --kl_weight=0.5 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar --model=Experiment_Logs/M17/saved_models/Model_epoch190'

python cluster_run.py --partition=learnfair --name=M18_eval --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M18 --entropy=0 --data=MIME --kl_weight=1.0 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar --model=Experiment_Logs/M18/saved_models/Model_epoch190'

python cluster_run.py --partition=learnfair --name=M19_eval --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M19 --entropy=0 --data=MIME --kl_weight=0.01 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=minmax --model=Experiment_Logs/M19/saved_models/Model_epoch190'

python cluster_run.py --partition=learnfair --name=M20_eval --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M20 --entropy=0 --data=MIME --kl_weight=0.1 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=minmax --model=Experiment_Logs/M20/saved_models/Model_epoch190'

python cluster_run.py --partition=learnfair --name=M21_eval --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M21 --entropy=0 --data=MIME --kl_weight=0.5 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=minmax --model=Experiment_Logs/M21/saved_models/Model_epoch190'

python cluster_run.py --partition=learnfair --name=M22_eval --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M22 --entropy=0 --data=MIME --kl_weight=1.0 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=minmax --model=Experiment_Logs/M22/saved_models/Model_epoch190'

# Rerun M15-M22 with the mean losses instead of sum losses. 
python cluster_run.py --partition=learnfair --name=M23 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M23 --entropy=0 --data=MIME --kl_weight=0.01 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M24 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M24 --entropy=0 --data=MIME --kl_weight=0.1 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M25 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M25 --entropy=0 --data=MIME --kl_weight=0.5 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M26 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M26 --entropy=0 --data=MIME --kl_weight=1.0 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M27 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M27 --entropy=0 --data=MIME --kl_weight=0.01 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M28 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M28 --entropy=0 --data=MIME --kl_weight=0.1 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M29 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M29 --entropy=0 --data=MIME --kl_weight=0.5 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M30 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M30 --entropy=0 --data=MIME --kl_weight=1.0 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=minmax'

# Eval
python cluster_run.py --partition=learnfair --name=M23_eval --cmd='python Master.py --train=0 --model=Experiment_Logs/M23/saved_models/Model_epoch70 --setting=pretrain_sub --name=M23 --entropy=0 --data=MIME --kl_weight=0.01 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M24_eval --cmd='python Master.py --train=0 --model=Experiment_Logs/M24/saved_models/Model_epoch70 --setting=pretrain_sub --name=M24 --entropy=0 --data=MIME --kl_weight=0.1 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M25_eval --cmd='python Master.py --train=0 --model=Experiment_Logs/M25/saved_models/Model_epoch70 --setting=pretrain_sub --name=M25 --entropy=0 --data=MIME --kl_weight=0.5 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M26_eval --cmd='python Master.py --train=0 --model=Experiment_Logs/M26/saved_models/Model_epoch70 --setting=pretrain_sub --name=M26 --entropy=0 --data=MIME --kl_weight=1.0 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M27_eval --cmd='python Master.py --train=0 --model=Experiment_Logs/M27/saved_models/Model_epoch70 --setting=pretrain_sub --name=M27 --entropy=0 --data=MIME --kl_weight=0.01 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M28_eval --cmd='python Master.py --train=0 --model=Experiment_Logs/M28/saved_models/Model_epoch70 --setting=pretrain_sub --name=M28 --entropy=0 --data=MIME --kl_weight=0.1 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M29_eval --cmd='python Master.py --train=0 --model=Experiment_Logs/M29/saved_models/Model_epoch70 --setting=pretrain_sub --name=M29 --entropy=0 --data=MIME --kl_weight=0.5 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M30_eval --cmd='python Master.py --train=0 --model=Experiment_Logs/M30/saved_models/Model_epoch70 --setting=pretrain_sub --name=M30 --entropy=0 --data=MIME --kl_weight=1.0 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M23_eval --cmd='python Master.py --train=0 --model=Experiment_Logs/M23/saved_models/Model_epoch190 --setting=pretrain_sub --name=M23 --entropy=0 --data=MIME --kl_weight=0.01 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M24_eval --cmd='python Master.py --train=0 --model=Experiment_Logs/M24/saved_models/Model_epoch190 --setting=pretrain_sub --name=M24 --entropy=0 --data=MIME --kl_weight=0.1 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M25_eval --cmd='python Master.py --train=0 --model=Experiment_Logs/M25/saved_models/Model_epoch190 --setting=pretrain_sub --name=M25 --entropy=0 --data=MIME --kl_weight=0.5 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M26_eval --cmd='python Master.py --train=0 --model=Experiment_Logs/M26/saved_models/Model_epoch190 --setting=pretrain_sub --name=M26 --entropy=0 --data=MIME --kl_weight=1.0 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M27_eval --cmd='python Master.py --train=0 --model=Experiment_Logs/M27/saved_models/Model_epoch190 --setting=pretrain_sub --name=M27 --entropy=0 --data=MIME --kl_weight=0.01 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M28_eval --cmd='python Master.py --train=0 --model=Experiment_Logs/M28/saved_models/Model_epoch190 --setting=pretrain_sub --name=M28 --entropy=0 --data=MIME --kl_weight=0.1 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M29_eval --cmd='python Master.py --train=0 --model=Experiment_Logs/M29/saved_models/Model_epoch190 --setting=pretrain_sub --name=M29 --entropy=0 --data=MIME --kl_weight=0.5 --discrete_z=0 --z_dimensions=8 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M30_eval --cmd='python Master.py --train=0 --model=Experiment_Logs/M30/saved_models/Model_epoch190 --setting=pretrain_sub --name=M30 --entropy=0 --data=MIME --kl_weight=1.0 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=minmax'

# Running with higher z dimensionality. For some reason we weren't doing this. 
python cluster_run.py --partition=learnfair --name=M31 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M31 --entropy=0 --data=MIME --kl_weight=0.01 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M32 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M32 --entropy=0 --data=MIME --kl_weight=0.1 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M33 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M33 --entropy=0 --data=MIME --kl_weight=0.5 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar'

python cluster_run.py --partition=leranfair --name=M34 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M34 --entropy=0 --data=MIME --kl_weight=1. --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M35 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M35 --entropy=0 --data=MIME --kl_weight=0.01 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M36 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M36 --entropy=0 --data=MIME --kl_weight=0.1 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M37 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M37 --entropy=0 --data=MIME --kl_weight=0.5 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M38 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M38 --entropy=0 --data=MIME --kl_weight=1. --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M39 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M39 --entropy=0 --data=MIME --kl_weight=0.01 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20'

python cluster_run.py --partition=learnfair --name=M40 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M40 --entropy=0 --data=MIME --kl_weight=0.1 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20'

python cluster_run.py --partition=learnfair --name=M41 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M41 --entropy=0 --data=MIME --kl_weight=0.5 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20'

python cluster_run.py --partition=learnfair --name=M42 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M42 --entropy=0 --data=MIME --kl_weight=1. --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20'

# Eval at m15, m30, m75
python cluster_run.py --partition=learnfair --name=M31_eval --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M31 --entropy=0 --data=MIME --model=Experiment_Logs/M31/saved_models/Model_epoch75 --kl_weight=0.01 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M32_eval --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M32 --entropy=0 --data=MIME --model=Experiment_Logs/M32/saved_models/Model_epoch75 --kl_weight=0.1 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M33_eval --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M33 --entropy=0 --data=MIME --model=Experiment_Logs/M33/saved_models/Model_epoch75 --kl_weight=0.5 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M34_eval --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M34 --entropy=0 --data=MIME --model=Experiment_Logs/M34/saved_models/Model_epoch75 --kl_weight=1. --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M35_eval --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M35 --entropy=0 --data=MIME --model=Experiment_Logs/M35/saved_models/Model_epoch75 --kl_weight=0.01 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M36_eval --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M36 --entropy=0 --data=MIME --model=Experiment_Logs/M36/saved_models/Model_epoch75 --kl_weight=0.1 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M37_eval --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M37 --entropy=0 --data=MIME --model=Experiment_Logs/M37/saved_models/Model_epoch75 --kl_weight=0.5 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M38_eval --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M38 --entropy=0 --data=MIME --model=Experiment_Logs/M38/saved_models/Model_epoch75 --kl_weight=1. --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M39_eval --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M39 --entropy=0 --data=MIME --model=Experiment_Logs/M39/saved_models/Model_epoch75 --kl_weight=0.01 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20'

python cluster_run.py --partition=learnfair --name=M40_eval --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M40 --entropy=0 --data=MIME --model=Experiment_Logs/M40/saved_models/Model_epoch75 --kl_weight=0.1 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20'

python cluster_run.py --partition=learnfair --name=M41_eval --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M41 --entropy=0 --data=MIME --model=Experiment_Logs/M41/saved_models/Model_epoch75 --kl_weight=0.5 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20'

python cluster_run.py --partition=learnfair --name=M42_eval --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M42 --entropy=0 --data=MIME --model=Experiment_Logs/M42/saved_models/Model_epoch75 --kl_weight=1. --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20'

# Eval runs with KL=0.01 at m80 with GT and 100 samples. 
python cluster_run.py --partition=learnfair --name=M31_eval --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M31 --entropy=0 --data=MIME --model=Experiment_Logs/M31/saved_models/Model_epoch80 --kl_weight=0.01 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar'
python cluster_run.py --partition=learnfair --name=M31_eval --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M31 --entropy=0 --data=MIME --model=Experiment_Logs/M31/saved_models/Model_epoch100 --kl_weight=0.01 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar'
python cluster_run.py --partition=learnfair --name=M31_eval_m120 --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M31 --entropy=0 --data=MIME --model=Experiment_Logs/M31/saved_models/Model_epoch120 --kl_weight=0.01 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M35_eval --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M35 --entropy=0 --data=MIME --model=Experiment_Logs/M35/saved_models/Model_epoch80 --kl_weight=0.01 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=minmax'
python cluster_run.py --partition=learnfair --name=M35_eval --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M35 --entropy=0 --data=MIME --model=Experiment_Logs/M35/saved_models/Model_epoch100 --kl_weight=0.01 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=minmax'
python cluster_run.py --partition=learnfair --name=M35_eval_m125 --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M35 --entropy=0 --data=MIME --model=Experiment_Logs/M35/saved_models/Model_epoch125 --kl_weight=0.01 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M39_eval --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M39 --entropy=0 --data=MIME --model=Experiment_Logs/M39/saved_models/Model_epoch80 --kl_weight=0.01 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20'
python cluster_run.py --partition=learnfair --name=M39_eval --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M39 --entropy=0 --data=MIME --model=Experiment_Logs/M39/saved_models/Model_epoch100 --kl_weight=0.01 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20'
python cluster_run.py --partition=learnfair --name=M39_eval_m125 --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M39 --entropy=0 --data=MIME --model=Experiment_Logs/M39/saved_models/Model_epoch125 --kl_weight=0.01 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20'

# Eval at m150
python cluster_run.py --partition=learnfair --name=M31_eval_m155 --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M31 --entropy=0 --data=MIME --model=Experiment_Logs/M31/saved_models/Model_epoch155 --kl_weight=0.01 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M35_eval_m160 --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M35 --entropy=0 --data=MIME --model=Experiment_Logs/M35/saved_models/Model_epoch160 --kl_weight=0.01 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M39_eval_m165 --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M39 --entropy=0 --data=MIME --model=Experiment_Logs/M39/saved_models/Model_epoch165 --kl_weight=0.01 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20'

# Eval at m185+
python cluster_run.py --partition=learnfair --name=M31_eval_m198 --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M31 --entropy=0 --data=MIME --model=Experiment_Logs/M31/saved_models/Model_epoch198 --kl_weight=0.01 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M35_eval_m199 --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M35 --entropy=0 --data=MIME --model=Experiment_Logs/M35/saved_models/Model_epoch199 --kl_weight=0.01 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M39_eval_m185 --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M39 --entropy=0 --data=MIME --model=Experiment_Logs/M39/saved_models/Model_epoch185 --kl_weight=0.01 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20'
python cluster_run.py --partition=learnfair --name=M39_eval_m199 --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M39 --entropy=0 --data=MIME --model=Experiment_Logs/M39/saved_models/Model_epoch199 --kl_weight=0.01 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20'

# Running with KL magnitude decreased even further than 0.01, because maybe KL divergence for larger dimensional z spaces is just higher magnitude. 
python cluster_run.py --partition=learnfair --name=M43 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M43 --entropy=0 --data=MIME --kl_weight=0.001 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M44 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M44 --entropy=0 --data=MIME --kl_weight=0.001 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M45 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M45 --entropy=0 --data=MIME --kl_weight=0.001 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20'

python cluster_run.py --partition=learnfair --name=M43_eval --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M43 --model=Experiment_Logs/M43/saved_models/Model_epoch75 --entropy=0 --data=MIME --kl_weight=0.001 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M44_eval --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M44 --model=Experiment_Logs/M44/saved_models/Model_epoch75 --entropy=0 --data=MIME --kl_weight=0.001 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M45_eval --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M45 --model=Experiment_Logs/M45/saved_models/Model_epoch75 --entropy=0 --data=MIME --kl_weight=0.001 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20'

# Eval at m100-125
python cluster_run.py --partition=learnfair --name=M43_eval --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M43 --model=Experiment_Logs/M43/saved_models/Model_epoch118 --entropy=0 --data=MIME --kl_weight=0.001 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M44_eval --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M44 --model=Experiment_Logs/M44/saved_models/Model_epoch100 --entropy=0 --data=MIME --kl_weight=0.001 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M45_eval --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M45 --model=Experiment_Logs/M45/saved_models/Model_epoch125 --entropy=0 --data=MIME --kl_weight=0.001 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20'

# Eval at m158+
python cluster_run.py --partition=learnfair --name=M43_eval --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M43 --model=Experiment_Logs/M43/saved_models/Model_epoch158 --entropy=0 --data=MIME --kl_weight=0.001 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M45_eval --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M45 --model=Experiment_Logs/M45/saved_models/Model_epoch167 --entropy=0 --data=MIME --kl_weight=0.001 --discrete_z=0 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --reparam=1 --traj_length=20'

# Running with variable length pretraining to reduce intersection of skills. 
python cluster_run.py --partition=learnfair --name=M46 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M46 --data=MIME --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M47 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M47 --data=MIME --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M48 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M48 --data=MIME --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64'

python cluster_run.py --partition=learnfair --name=M49 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M49 --data=MIME --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M50 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M50 --data=MIME --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M51 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M51 --data=MIME --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64'

# Eval M46-M51 at m70ish. 
# debug 
python Master.py --train=0 --setting=pretrain_sub --name=M46 --model=Experiment_Logs/M46/saved_models/Model_epoch74 --data=MIME --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --normalization=meanvar

# Run
python cluster_run.py --partition=learnfair --name=M46_eval_m180 --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M46 --model=Experiment_Logs/M46/saved_models/Model_epoch180 --data=MIME --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M47_eval_m172 --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M47 --model=Experiment_Logs/M47/saved_models/Model_epoch172 --data=MIME --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M48_eval_m180 --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M48 --model=Experiment_Logs/M48/saved_models/Model_epoch180 --data=MIME --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64'

python cluster_run.py --partition=learnfair --name=M49_eval_m180 --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M49 --model=Experiment_Logs/M49/saved_models/Model_epoch180 --data=MIME --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M50_eval_m180 --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M50 --model=Experiment_Logs/M50/saved_models/Model_epoch180 --data=MIME --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M51_eval_m180 --cmd='python Master.py --train=0 --setting=pretrain_sub --name=M51 --model=Experiment_Logs/M51/saved_models/Model_epoch180 --data=MIME --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64'

# Debug automatic eval 
python Master.py --train=1 --setting=pretrain_sub --name=Mdebug_autoeval --data=MIME --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --normalization=meanvar

# Rerun M46-M51 with auto-eval. Auto-eval was just screwing up, so treat this as... a low var run if need it. 
python cluster_run.py --partition=learnfair --name=M52 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M52 --data=MIME --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M53 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M53 --data=MIME --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M54 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M54 --data=MIME --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64'

python cluster_run.py --partition=learnfair --name=M55 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M55 --data=MIME --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M56 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M56 --data=MIME --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M57 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M57 --data=MIME --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64'

# Running with variable length pretraining to reduce intersection of skills. 
python cluster_run.py --partition=learnfair --name=M46 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M46 --data=MIME --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M47 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M47 --data=MIME --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M48 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M48 --data=MIME --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64'

python cluster_run.py --partition=learnfair --name=M49 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M49 --data=MIME --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M50 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M50 --data=MIME --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M51 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M51 --data=MIME --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64'

# Debug auto eval
python Master.py --train=1 --setting=pretrain_sub --name=Mdebug_autoeval2 --data=MIME --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --normalization=meanvar --transformer=1

# Running M46-51 with transformer. 
python cluster_run.py --partition=learnfair --name=M58 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M58 --transformer=1 --data=MIME --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M59 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M59 --transformer=1 --data=MIME --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M60 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M60 --transformer=1 --data=MIME --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64'

python cluster_run.py --partition=learnfair --name=M61 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M61 --transformer=1 --data=MIME --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M62 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M62 --transformer=1 --data=MIME --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M63 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M63 --transformer=1 --data=MIME --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64'

# Eval
python cluster_run.py --partition=learnfair --name=M58_eval_m20 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M58 --transformer=1 --data=MIME --model=Experiment_Logs/M58/saved_models/Model_epoch20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M59_eval_m20 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M59 --transformer=1 --data=MIME --model=Experiment_Logs/M59/saved_models/Model_epoch20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M60_eval_m20 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M60 --transformer=1 --data=MIME --model=Experiment_Logs/M60/saved_models/Model_epoch20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64'

python cluster_run.py --partition=learnfair --name=M61_eval_m20 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M61 --transformer=1 --data=MIME --model=Experiment_Logs/M61/saved_models/Model_epoch20 --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M62_eval_m20 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M62 --transformer=1 --data=MIME --model=Experiment_Logs/M62/saved_models/Model_epoch20 --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M63_eval_m20 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M63 --transformer=1 --data=MIME --model=Experiment_Logs/M63/saved_models/Model_epoch20 --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64'

# Rerun M46-M51 with transformer argument in autoeval.
python cluster_run.py --partition=learnfair --name=M64 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M64 --transformer=1 --data=MIME --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M65 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M65 --transformer=1 --data=MIME --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M66 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M66 --transformer=1 --data=MIME --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64'

python cluster_run.py --partition=learnfair --name=M67 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M67 --transformer=1 --data=MIME --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M68 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M68 --transformer=1 --data=MIME --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M69 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M69 --transformer=1 --data=MIME --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64'

#### Runs with biased sampling, and various parameters. 
# Debug
python Master.py --train=1 --setting=pretrain_sub --name=Mdebug_biasing --data=MIME --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --normalization=meanvar --pretrain_bias_sampling=0.1 --pretrain_bias_sampling_prob=0.3

# Rerun M46-M51 without transformer with biased sampling. 
python cluster_run.py --partition=learnfair --name=M70 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M70 --data=MIME --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --normalization=meanvar --pretrain_bias_sampling=0.1 --pretrain_bias_sampling_prob=0.3'

python cluster_run.py --partition=learnfair --name=M71 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M71 --data=MIME --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --normalization=minmax --pretrain_bias_sampling=0.1 --pretrain_bias_sampling_prob=0.3'

python cluster_run.py --partition=learnfair --name=M72 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M72 --data=MIME --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --pretrain_bias_sampling=0.1 --pretrain_bias_sampling_prob=0.3'

python cluster_run.py --partition=learnfair --name=M73 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M73 --data=MIME --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --normalization=meanvar --pretrain_bias_sampling=0.1 --pretrain_bias_sampling_prob=0.3'

python cluster_run.py --partition=learnfair --name=M74 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M74 --data=MIME --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --normalization=minmax --pretrain_bias_sampling=0.1 --pretrain_bias_sampling_prob=0.3'

python cluster_run.py --partition=learnfair --name=M75 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M75 --data=MIME --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --pretrain_bias_sampling=0.1 --pretrain_bias_sampling_prob=0.3'

# Now running with 20% of the trajectory and 30% probability of sampling biased.
python cluster_run.py --partition=learnfair --name=M76 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M76 --data=MIME --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --normalization=meanvar --pretrain_bias_sampling=0.2 --pretrain_bias_sampling_prob=0.3'

python cluster_run.py --partition=learnfair --name=M77 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M77 --data=MIME --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --normalization=minmax --pretrain_bias_sampling=0.2 --pretrain_bias_sampling_prob=0.3'

python cluster_run.py --partition=learnfair --name=M78 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M78 --data=MIME --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --pretrain_bias_sampling=0.2 --pretrain_bias_sampling_prob=0.3'

python cluster_run.py --partition=learnfair --name=M79 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M79 --data=MIME --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --normalization=meanvar --pretrain_bias_sampling=0.2 --pretrain_bias_sampling_prob=0.3'

python cluster_run.py --partition=learnfair --name=M80 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M80 --data=MIME --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --normalization=minmax --pretrain_bias_sampling=0.2 --pretrain_bias_sampling_prob=0.3'

python cluster_run.py --partition=learnfair --name=M81 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M81 --data=MIME --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --pretrain_bias_sampling=0.2 --pretrain_bias_sampling_prob=0.3'


###############################################################
##################### MIME JOINT TRIALS #######################
###############################################################

python cluster_run.py --name=MJ1 --cmd='python Master.py --train=1 --setting=learntsub --name=MJ1_loadM10 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.1 --data=MIME --subpolicy_model=Experiment_Logs/M10/saved_models/Model_epoch190 --latent_loss_weight=0.01 --z_dimensions=8 --traj_length=20'

python cluster_run.py --name=MJ2 --cmd='python Master.py --train=1 --setting=learntsub --name=MJ2_loadM14 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.1 --data=MIME --subpolicy_model=Experiment_Logs/M14/saved_models/Model_epoch100 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=20'

# Running with correctly set max limits for MIME data. 
python cluster_run.py --name=MJ3 --cmd='python Master.py --train=1 --setting=learntsub --name=MJ3_loadM10 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.1 --data=MIME --subpolicy_model=Experiment_Logs/M10/saved_models/Model_epoch190 --latent_loss_weight=0.01 --z_dimensions=8 --traj_length=20'

python cluster_run.py --name=MJ4 --cmd='python Master.py --train=1 --setting=learntsub --name=MJ4_loadM14 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.1 --data=MIME --subpolicy_model=Experiment_Logs/M14/saved_models/Model_epoch100 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=20'

# Trying visualization.
python Master.py --train=1 --setting=learntsub --name=MJ_debugvisuals --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.1 --data=MIME --subpolicy_model=Experiment_Logs/M14/saved_models/Model_epoch100 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1

## Running with correctly set max limits for MIME data. 
python cluster_run.py --name=MJ5 --cmd='python Master.py --train=1 --setting=learntsub --name=MJ5_loadM10 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.1 --data=MIME --subpolicy_model=Experiment_Logs/M10/saved_models/Model_epoch190 --latent_loss_weight=0.01 --z_dimensions=8 --traj_length=-1'

python cluster_run.py --name=MJ6 --cmd='python Master.py --train=1 --setting=learntsub --name=MJ6_loadM14 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.1 --data=MIME --subpolicy_model=Experiment_Logs/M14/saved_models/Model_epoch100 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1'

# Debug weights
python Master.py --train=1 --setting=learntsub --name=MJ_debug --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.1 --data=MIME --subpolicy_model=Experiment_Logs/M14/saved_models/Model_epoch100 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --debug=1

# Run MJ5 and MJ6 with skill length in prior set to 20., and probability factor naturally reduced to 0.01
python cluster_run.py --name=MJ7 --cmd='python Master.py --train=1 --setting=learntsub --name=MJ7_loadM10 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M10/saved_models/Model_epoch190 --latent_loss_weight=0.01 --z_dimensions=8 --traj_length=-1 --skill_length=20'

python cluster_run.py --name=MJ8 --cmd='python Master.py --train=1 --setting=learntsub --name=MJ8_loadM14 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M14/saved_models/Model_epoch100 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --skill_length=20'

# Debug everything in life
python Master.py --train=1 --setting=learntsub --name=MJ_debug --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M14/saved_models/Model_epoch100 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --skill_length=20 --debug=200

# New MIME Joint trials. 
python cluster_run.py --name=MJ9 --cmd='python Master.py --train=1 --setting=learntsub --name=MJ9_loadM10 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M10/saved_models/Model_epoch190 --latent_loss_weight=0.01 --z_dimensions=8 --traj_length=-1 --skill_length=20'

python cluster_run.py --name=MJ10 --cmd='python Master.py --train=1 --setting=learntsub --name=MJ10_loadM14 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M14/saved_models/Model_epoch100 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --skill_length=20'

python Master.py --train=1 --setting=learntsub --name=MJ11_loadM14_debugsingle --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M14/saved_models/Model_epoch100 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --skill_length=20

python Master.py --train=1 --setting=learntsub --name=MJdebug_loadM14 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M14/saved_models/Model_epoch100 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --skill_length=20

# New MIME Joint trials with new_update_policy.
python cluster_run.py --name=J12 --cmd='python Master.py --train=1 --setting=learntsub --name=J12_loadM10 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M10/saved_models/Model_epoch190 --latent_loss_weight=0.01 --z_dimensions=8 --traj_length=-1 --skill_length=20'

python cluster_run.py --name=J13 --cmd='python Master.py --train=1 --setting=learntsub --name=J13_loadM14 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M14/saved_models/Model_epoch100 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --skill_length=20'

python Master.py --train=1 --setting=learntsub --name=MJdebug_loadM14_singledp --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M14/saved_models/Model_epoch100 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --skill_length=20

# Running with subpolicy_ratio set to 1, new_update_policies, and clamp_value=0.
python cluster_run.py --name=J14 --cmd='python Master.py --train=1 --setting=learntsub --name=J14_loadM10 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M10/saved_models/Model_epoch190 --latent_loss_weight=0.01 --z_dimensions=8 --traj_length=-1 --skill_length=20'

python cluster_run.py --name=J15 --cmd='python Master.py --train=1 --setting=learntsub --name=J15_loadM14 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M14/saved_models/Model_epoch100 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --skill_length=20'

python cluster_run.py --name=J16 --cmd='python Master.py --train=1 --setting=learntsub --name=J16_loadM10 --subpolicy_ratio=0.01 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M10/saved_models/Model_epoch190 --latent_loss_weight=0.01 --z_dimensions=8 --traj_length=-1 --skill_length=20'

python cluster_run.py --name=J17 --cmd='python Master.py --train=1 --setting=learntsub --name=J17_loadM14 --subpolicy_ratio=0.01 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M14/saved_models/Model_epoch100 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --skill_length=20'

# New runs with M47+. 
# Debug
python Master.py --train=1 --setting=learntsub --name=Jdebug_loadM47_m172 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M47/saved_models/Model_epoch172 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1

python cluster_run.py --name=J18 --cmd='python Master.py --train=1 --setting=learntsub --name=J18_loadM47_m172 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M47/saved_models/Model_epoch172 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1'

python cluster_run.py --name=J19 --cmd='python Master.py --train=1 --setting=learntsub --name=J19_loadM47_m160 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M47/saved_models/Model_epoch160 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1'

python cluster_run.py --name=J20 --cmd='python Master.py --train=1 --setting=learntsub --name=J20_loadM47_m140 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M47/saved_models/Model_epoch140 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1'

python cluster_run.py --name=J21 --cmd='python Master.py --train=1 --setting=learntsub --name=J21_loadM50_m199 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M50/saved_models/Model_epoch199 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1'

python cluster_run.py --name=J22 --cmd='python Master.py --train=1 --setting=learntsub --name=J22_loadM50_m180 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M50/saved_models/Model_epoch180 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1'

python cluster_run.py --name=J23 --cmd='python Master.py --train=1 --setting=learntsub --name=J23_loadM50_m160 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M50/saved_models/Model_epoch160 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1'

# Rerun J18-23 with KL weight = 0.001 and 0.0001.
python cluster_run.py --name=J24 --cmd='python Master.py --train=1 --setting=learntsub --name=J24_loadM47_m172 --kl_weight=0.001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M47/saved_models/Model_epoch172 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1'

python cluster_run.py --name=J25 --cmd='python Master.py --train=1 --setting=learntsub --name=J25_loadM47_m160 --kl_weight=0.001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M47/saved_models/Model_epoch160 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1'

python cluster_run.py --name=J26 --cmd='python Master.py --train=1 --setting=learntsub --name=J26_loadM47_m140 --kl_weight=0.001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M47/saved_models/Model_epoch140 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1'

python cluster_run.py --name=J27 --cmd='python Master.py --train=1 --setting=learntsub --name=J27_loadM50_m199 --kl_weight=0.001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M50/saved_models/Model_epoch199 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1'

python cluster_run.py --name=J28 --cmd='python Master.py --train=1 --setting=learntsub --name=J28_loadM50_m180 --kl_weight=0.001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M50/saved_models/Model_epoch180 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1'

python cluster_run.py --name=J29 --cmd='python Master.py --train=1 --setting=learntsub --name=J29_loadM50_m160 --kl_weight=0.001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M50/saved_models/Model_epoch160 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1'

# With KL weight 0.0001.
python cluster_run.py --name=J30 --cmd='python Master.py --train=1 --setting=learntsub --name=J30_loadM47_m172 --kl_weight=0.0001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M47/saved_models/Model_epoch172 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1'

python cluster_run.py --name=J31 --cmd='python Master.py --train=1 --setting=learntsub --name=J31_loadM47_m160 --kl_weight=0.0001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M47/saved_models/Model_epoch160 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1'

python cluster_run.py --name=J32 --cmd='python Master.py --train=1 --setting=learntsub --name=J32_loadM47_m140 --kl_weight=0.0001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M47/saved_models/Model_epoch140 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1'

python cluster_run.py --name=J33 --cmd='python Master.py --train=1 --setting=learntsub --name=J33_loadM50_m199 --kl_weight=0.0001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M50/saved_models/Model_epoch199 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1'

python cluster_run.py --name=J34 --cmd='python Master.py --train=1 --setting=learntsub --name=J34_loadM50_m180 --kl_weight=0.0001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M50/saved_models/Model_epoch180 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1'

python cluster_run.py --name=J35 --cmd='python Master.py --train=1 --setting=learntsub --name=J35_loadM50_m160 --kl_weight=0.0001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M50/saved_models/Model_epoch160 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1'

# We forgot to run with normalization. 
# Debug
python Master.py --train=1 --setting=learntsub --name=J35_loadM47_m172 --normalization=minmax --kl_weight=0.01 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M47/saved_models/Model_epoch172 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1 --debug=300

#KL Weight 0.01
python cluster_run.py --name=J35 --cmd='python Master.py --train=1 --setting=learntsub --name=J35_loadM47_m172 --normalization=minmax --kl_weight=0.01 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M47/saved_models/Model_epoch172 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1'

python cluster_run.py --name=J36 --cmd='python Master.py --train=1 --setting=learntsub --name=J36_loadM47_m160 --normalization=minmax --kl_weight=0.01 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M47/saved_models/Model_epoch160 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1'

python cluster_run.py --name=J37 --cmd='python Master.py --train=1 --setting=learntsub --name=J37_loadM47_m140 --normalization=minmax --kl_weight=0.01 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M47/saved_models/Model_epoch140 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1'

python cluster_run.py --name=J38 --cmd='python Master.py --train=1 --setting=learntsub --name=J38_loadM50_m199 --normalization=minmax --kl_weight=0.01 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M50/saved_models/Model_epoch199 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1'

python cluster_run.py --name=J39 --cmd='python Master.py --train=1 --setting=learntsub --name=J39_loadM50_m180 --normalization=minmax --kl_weight=0.01 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M50/saved_models/Model_epoch180 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1'

python cluster_run.py --name=J40 --cmd='python Master.py --train=1 --setting=learntsub --name=J40_loadM50_m160 --normalization=minmax --kl_weight=0.01 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M50/saved_models/Model_epoch160 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1'

# KL Weight 0.001.
python cluster_run.py --name=J41 --cmd='python Master.py --train=1 --setting=learntsub --name=J41_loadM47_m172 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M47/saved_models/Model_epoch172 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1'

python cluster_run.py --name=J42 --cmd='python Master.py --train=1 --setting=learntsub --name=J42_loadM47_m160 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M47/saved_models/Model_epoch160 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1'

python cluster_run.py --name=J43 --cmd='python Master.py --train=1 --setting=learntsub --name=J43_loadM47_m140 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M47/saved_models/Model_epoch140 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1'

python cluster_run.py --name=J44 --cmd='python Master.py --train=1 --setting=learntsub --name=J44_loadM50_m199 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M50/saved_models/Model_epoch199 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1'

python cluster_run.py --name=J45 --cmd='python Master.py --train=1 --setting=learntsub --name=J45_loadM50_m180 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M50/saved_models/Model_epoch180 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1'

python cluster_run.py --name=J46 --cmd='python Master.py --train=1 --setting=learntsub --name=J46_loadM50_m160 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M50/saved_models/Model_epoch160 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1'

# KL Weight 0.0001
python cluster_run.py --name=J47 --cmd='python Master.py --train=1 --setting=learntsub --name=J47_loadM47_m172 --normalization=minmax --kl_weight=0.0001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M47/saved_models/Model_epoch172 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1'

python cluster_run.py --name=J48 --cmd='python Master.py --train=1 --setting=learntsub --name=J48_loadM47_m160 --normalization=minmax --kl_weight=0.0001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M47/saved_models/Model_epoch160 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1'

python cluster_run.py --name=J49 --cmd='python Master.py --train=1 --setting=learntsub --name=J49_loadM47_m140 --normalization=minmax --kl_weight=0.0001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M47/saved_models/Model_epoch140 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1'

python cluster_run.py --name=J50 --cmd='python Master.py --train=1 --setting=learntsub --name=J50_loadM50_m199 --normalization=minmax --kl_weight=0.0001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M50/saved_models/Model_epoch199 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1'

python cluster_run.py --name=J51 --cmd='python Master.py --train=1 --setting=learntsub --name=J51_loadM50_m180 --normalization=minmax --kl_weight=0.0001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M50/saved_models/Model_epoch180 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1'

python cluster_run.py --name=J52 --cmd='python Master.py --train=1 --setting=learntsub --name=J52_loadM50_m160 --normalization=minmax --kl_weight=0.0001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M50/saved_models/Model_epoch160 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1'

# Debug the transformer joint training. 
python Master.py --train=1 --setting=learntsub --name=Jdebug_transformer --normalization=minmax --kl_weight=0.0001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=MIME --subpolicy_model=Experiment_Logs/M50/saved_models/Model_epoch160 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1 --transformer=1


########################################################
# Pretraining prior for experiments on diversity

# Debug
python Master.py --train=1 --setting=pretrain_prior --name=Pdebug --data=MIME --var_skill_length=1 --z_dimensions=0

python cluster_run.py --name=P1 --cmd='python Master.py --train=1 --setting=pretrain_prior --name=P1 --data=MIME --var_skill_length=1 --z_dimensions=0'

python cluster_run.py --name=P2 --cmd='python Master.py --train=1 --setting=pretrain_prior --name=P2 --data=MIME --var_skill_length=1 --z_dimensions=0 --normalization=meanvar'

python cluster_run.py --name=P3 --cmd='python Master.py --train=1 --setting=pretrain_prior --name=P3 --data=MIME --var_skill_length=1 --z_dimensions=0 --normalization=minmax'

# Run again with differently named model. 
python cluster_run.py --name=P4 --cmd='python Master.py --train=1 --setting=pretrain_prior --name=P4 --data=MIME --var_skill_length=1 --z_dimensions=0'

# Eval
python Master.py --train=0 --setting=pretrain_prior --name=P4 --data=MIME --var_skill_length=1 --z_dimensions=0 --model=Experiment_Logs/P4/saved_models/Model_epoch199

python cluster_run.py --name=P5 --cmd='python Master.py --train=1 --setting=pretrain_prior --name=P5 --data=MIME --var_skill_length=1 --z_dimensions=0 --normalization=meanvar'

python cluster_run.py --name=P6 --cmd='python Master.py --train=1 --setting=pretrain_prior --name=P6 --data=MIME --var_skill_length=1 --z_dimensions=0 --normalization=minmax'

