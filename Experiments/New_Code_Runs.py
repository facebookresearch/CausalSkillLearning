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