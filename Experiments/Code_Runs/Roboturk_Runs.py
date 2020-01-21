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
python Master.py --train=1 --setting=pretrain_sub --name=R17_rerun13 --data=Roboturk --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --debug=1
# Eval
python Master.py --train=0 --setting=pretrain_sub --name=R17_rerun13 --data=Roboturk --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --model=Experiment_Logs/R17_rerun13/saved_models/Model_epoch0

# Rerun with speedy gonzales runs. 
python cluster_run.py --partition=learnfair --name=R18 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R18 --data=Roboturk --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128'

python cluster_run.py --partition=learnfair --name=R19 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R19 --data=Roboturk --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128'

# Rerun with new dataloader / dataset. 
python cluster_run.py --partition=learnfair --name=R20 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R20 --data=Roboturk --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128'

python cluster_run.py --partition=learnfair --name=R21 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R21 --data=Roboturk --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128'

# Eval and debug.
python Master.py --train=0 --setting=pretrain_sub --name=Rdebug --data=Roboturk --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --model=Experiment_Logs/R20/saved_models/Model_epoch0

# Rerun with new segmented AND SMOOTHED dataset. 
python cluster_run.py --partition=learnfair --name=R22 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R22 --data=Roboturk --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128'

python cluster_run.py --partition=learnfair --name=R23 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R23 --data=Roboturk --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128'

# Rerunning R22-R23. 
# Rerun with new segmented AND SMOOTHED dataset. 
python cluster_run.py --partition=learnfair --name=R24 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R24 --data=Roboturk --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128'

python cluster_run.py --partition=learnfair --name=R25 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R25 --data=Roboturk --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128'

# Debug
python Master.py --train=0 --setting=pretrain_sub --name=Rdebug --data=Roboturk --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --model=Experiment_Logs/R24/saved_models/Model_epoch0

python Master.py --train=0 --setting=pretrain_sub --name=Rdebug_2 --data=Roboturk --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --model=Experiment_Logs/R24/saved_models/Model_epoch0

python Master.py --train=0 --setting=pretrain_sub --name=Rdebug_2 --data=Roboturk --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --gripper=0

# Training with and without gripper for comparison. 
python cluster_run.py --partition=learnfair --name=R26 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R26 --data=Roboturk --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --gripper=0'

python cluster_run.py --partition=learnfair --name=R27 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R27 --data=Roboturk --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --gripper=0'

# Rerun with new segmented SMOOTHED dataset and scaling. 
python cluster_run.py --partition=learnfair --name=R28 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R28 --data=Roboturk --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --action_scale_factor=10'
# Debug 
python Master.py --train=1 --setting=pretrain_sub --name=R28_eval --data=Roboturk --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --action_scale_factor=10 --model=Experiment_Logs/R28/saved_models/Model_epoch300 --debug=1
# compare with a MIME run
python Master.py --train=0 --setting=pretrain_sub --name=M47_eval --model=Experiment_Logs/M47/saved_models/Model_epoch172 --data=MIME --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --normalization=minmax --debug=1

# Eval - # Run eval with new rollout.. 
python cluster_run.py --partition=learnfair --name=R28_m300 --cmd='python Master.py --train=0 --setting=pretrain_sub --name=R28_Eval --data=Roboturk --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --action_scale_factor=10 --model=Experiment_Logs/R28/saved_models/Model_epoch300'

python cluster_run.py --partition=learnfair --name=R28_m360 --cmd='python Master.py --train=0 --setting=pretrain_sub --name=R28_Eval --data=Roboturk --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --action_scale_factor=10 --model=Experiment_Logs/R28/saved_models/Model_epoch360'

python cluster_run.py --partition=learnfair --name=R29 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R29 --data=Roboturk --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --action_scale_factor=10'
# Run eval with new rollout.. 
python cluster_run.py --partition=learnfair --name=R29_m300 --cmd='python Master.py --train=0 --setting=pretrain_sub --name=R29_Eval --data=Roboturk --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --action_scale_factor=10 --model=Experiment_Logs/R29/saved_models/Model_epoch300'

python cluster_run.py --partition=learnfair --name=R29_m360 --cmd='python Master.py --train=0 --setting=pretrain_sub --name=R29_Eval --data=Roboturk --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --action_scale_factor=10 --model=Experiment_Logs/R29/saved_models/Model_epoch360'


# Scaling, new data, and no gripper. 
python cluster_run.py --partition=learnfair --name=R30 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R30 --data=Roboturk --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --action_scale_factor=10 --gripper=0'

python cluster_run.py --partition=learnfair --name=R31 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R31 --data=Roboturk --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --action_scale_factor=10 --gripper=0'

# Downsampling further. 
python cluster_run.py --partition=learnfair --name=R32 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R32 --data=Roboturk --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --ds_freq=2'

python cluster_run.py --partition=learnfair --name=R33 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R33 --data=Roboturk --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --ds_freq=2'

########################################################################
################# Trying normalization with roboturk. ##################
########################################################################

# Debug
python Master.py --train=1 --setting=pretrain_sub --name=Rdebugnorm --data=Roboturk --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --debug=1

# MEANVAR NORM.
# Smooth data, with gripper, no scaling.
python cluster_run.py --partition=learnfair --name=R34 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R34 --data=Roboturk --normalization=meanvar --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128'

python cluster_run.py --partition=learnfair --name=R35 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R35 --data=Roboturk --normalization=meanvar --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128'

# Smooth data, no gripper, no scaling.
python cluster_run.py --partition=learnfair --name=R36 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R36 --data=Roboturk --normalization=meanvar --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --gripper=0'

python cluster_run.py --partition=learnfair --name=R37 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R37 --data=Roboturk --normalization=meanvar --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --gripper=0'

# Smooth data, with gripper, yes scaling.
python cluster_run.py --partition=learnfair --name=R38 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R38 --data=Roboturk --normalization=meanvar --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --action_scale_factor=10'

python cluster_run.py --partition=learnfair --name=R39 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R39 --data=Roboturk --normalization=meanvar --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --action_scale_factor=10'

# Eval
python cluster_run.py --partition=learnfair --name=R38 --cmd='python Master.py --train=0 --setting=pretrain_sub --name=R38_Eval --data=Roboturk --normalization=meanvar --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --action_scale_factor=10 --model=Experiment_Logs/R38/saved_models/Model_epoch300'

python cluster_run.py --partition=learnfair --name=R39 --cmd='python Master.py --train=0 --setting=pretrain_sub --name=R39_Eval --data=Roboturk --normalization=meanvar --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --action_scale_factor=10 --model=Experiment_Logs/R39/saved_models/Model_epoch300'


# Smooth data, no gripper, yes scaling.
python cluster_run.py --partition=learnfair --name=R40 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R40 --data=Roboturk --normalization=meanvar --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --action_scale_factor=10 --gripper=0'

python cluster_run.py --partition=learnfair --name=R41 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R41 --data=Roboturk --normalization=meanvar --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --action_scale_factor=10 --gripper=0'

# Smooth data, with gripper, no scaling, downsampling.
python cluster_run.py --partition=learnfair --name=R42 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R42 --data=Roboturk --normalization=meanvar --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --ds_freq=2'

python cluster_run.py --partition=learnfair --name=R43 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R43 --data=Roboturk --normalization=meanvar --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --ds_freq=2'

# MINMAX NORM.
# Smooth data, with gripper, no scaling.
python cluster_run.py --partition=learnfair --name=R44 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R44 --data=Roboturk --normalization=minmax --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128'

python cluster_run.py --partition=learnfair --name=R45 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R45 --data=Roboturk --normalization=minmax --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128'

# Smooth data, no gripper, no scaling.
python cluster_run.py --partition=learnfair --name=R46 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R46 --data=Roboturk --normalization=minmax --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --gripper=0'

python cluster_run.py --partition=learnfair --name=R47 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R47 --data=Roboturk --normalization=minmax --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --gripper=0'

# Smooth data, with gripper, yes scaling.
python cluster_run.py --partition=learnfair --name=R48 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R48 --data=Roboturk --normalization=minmax --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --action_scale_factor=10'

python cluster_run.py --partition=learnfair --name=R49 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R49 --data=Roboturk --normalization=minmax --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --action_scale_factor=10'

# Smooth data, no gripper, yes scaling.
python cluster_run.py --partition=learnfair --name=R50 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R50 --data=Roboturk --normalization=minmax --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --action_scale_factor=10 --gripper=0'

python cluster_run.py --partition=learnfair --name=R51 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R51 --data=Roboturk --normalization=minmax --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --action_scale_factor=10 --gripper=0'

# Smooth data, with gripper, no scaling, downsampling.
python cluster_run.py --partition=learnfair --name=R52 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R52 --data=Roboturk --normalization=minmax --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --ds_freq=2'

python cluster_run.py --partition=learnfair --name=R53 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R53 --data=Roboturk --normalization=minmax --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --ds_freq=2'

###############
# Smooth, with gripper, scaling to 100. 
python cluster_run.py --partition=learnfair --name=R54 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R54 --data=Roboturk --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --action_scale_factor=100'

python cluster_run.py --partition=learnfair --name=R55 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R55 --data=Roboturk --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --action_scale_factor=100'

# Eval
python cluster_run.py --partition=learnfair --name=R54 --cmd='python Master.py --train=0 --setting=pretrain_sub --name=R54_Eval --data=Roboturk --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --action_scale_factor=100 --model=Experiment_Logs/R54/saved_models/Model_epoch140'

python cluster_run.py --partition=learnfair --name=R55 --cmd='python Master.py --train=0 --setting=pretrain_sub --name=R55_Eval --data=Roboturk --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --action_scale_factor=100 --model=Experiment_Logs/R55/saved_models/Model_epoch140'

# Original capacity, reduced epsilon. 
python cluster_run.py --partition=learnfair --name=R60 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R60 --data=Roboturk --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --action_scale_factor=10 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=10'

python cluster_run.py --partition=learnfair --name=R61 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R61 --data=Roboturk --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --action_scale_factor=10 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=10'

# Running with reduced KL in life. 
python cluster_run.py --partition=learnfair --name=R64 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R64 --data=Roboturk --kl_weight=0.00001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128'

python cluster_run.py --partition=learnfair --name=R65 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R65 --data=Roboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128'

########################################################
# Running with different kernel bandwidth values. 
python cluster_run.py --partition=learnfair --name=R66 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R66 --data=Roboturk --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --smoothing_kernel_bandwidth=1.'

python cluster_run.py --partition=learnfair --name=R67 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R67 --data=Roboturk --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --smoothing_kernel_bandwidth=1.'

python cluster_run.py --partition=learnfair --name=R68 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R68 --data=Roboturk --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --smoothing_kernel_bandwidth=1.5'

python cluster_run.py --partition=learnfair --name=R69 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R69 --data=Roboturk --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --smoothing_kernel_bandwidth=1.5'

python cluster_run.py --partition=learnfair --name=R70 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R70 --data=Roboturk --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --smoothing_kernel_bandwidth=2.'

python cluster_run.py --partition=learnfair --name=R71 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R71 --data=Roboturk --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --smoothing_kernel_bandwidth=2.'

# Rerunning these kernel bandwidth values with meanvar and minmax normalziation. 
python cluster_run.py --partition=learnfair --name=R72 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R72 --data=Roboturk --normalization=meanvar --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --smoothing_kernel_bandwidth=1.'

python cluster_run.py --partition=learnfair --name=R73 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R73 --data=Roboturk --normalization=meanvar --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --smoothing_kernel_bandwidth=1.'

python cluster_run.py --partition=learnfair --name=R74 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R74 --data=Roboturk --normalization=meanvar --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --smoothing_kernel_bandwidth=1.5'

python cluster_run.py --partition=learnfair --name=R75 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R75 --data=Roboturk --normalization=meanvar --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --smoothing_kernel_bandwidth=1.5'

python cluster_run.py --partition=learnfair --name=R76 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R76 --data=Roboturk --normalization=meanvar --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --smoothing_kernel_bandwidth=2.'

python cluster_run.py --partition=learnfair --name=R77 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R77 --data=Roboturk --normalization=meanvar --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --smoothing_kernel_bandwidth=2.'

# Minmax
python cluster_run.py --partition=learnfair --name=R78 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R78 --data=Roboturk --normalization=minmax --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --smoothing_kernel_bandwidth=1.'

python cluster_run.py --partition=learnfair --name=R79 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R79 --data=Roboturk --normalization=minmax --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --smoothing_kernel_bandwidth=1.'

python cluster_run.py --partition=learnfair --name=R80 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R80 --data=Roboturk --normalization=minmax --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --smoothing_kernel_bandwidth=1.5'

python cluster_run.py --partition=learnfair --name=R81 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R81 --data=Roboturk --normalization=minmax --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --smoothing_kernel_bandwidth=1.5'

python cluster_run.py --partition=learnfair --name=R82 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R82 --data=Roboturk --normalization=minmax --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --smoothing_kernel_bandwidth=2.'

python cluster_run.py --partition=learnfair --name=R83 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R83 --data=Roboturk --normalization=minmax --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --smoothing_kernel_bandwidth=2.'

# Debug
python Master.py --train=1 --setting=pretrain_sub --name=Rpreproc_full --data=FullRoboturk --debug=1

# Rerunning on original dataset. 
python cluster_run.py --partition=learnfair --name=R84 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R84 --data=OrigRoboturk --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128'

python cluster_run.py --partition=learnfair --name=R85 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R85 --data=OrigRoboturk --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128'

# Eval
python cluster_run.py --partition=learnfair --name=R84E --cmd='python Master.py --train=0 --setting=pretrain_sub --name=R84_Eval --data=OrigRoboturk --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128  --model=Experiment_Logs/R84/saved_models/Model_epoch20'

python cluster_run.py --partition=learnfair --name=R85E --cmd='python Master.py --train=0 --setting=pretrain_sub --name=R85_Eval --data=OrigRoboturk --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --model=Experiment_Logs/R85/saved_models/Model_epoch20'

# Rerunning on original dataset with smoothening. 
python cluster_run.py --partition=learnfair --name=R86 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R86 --data=OrigRoboturk --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --smoothen=1 --smoothing_kernel_bandwidth=1.5'

python cluster_run.py --partition=learnfair --name=R87 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R87 --data=OrigRoboturk --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --smoothen=1 --smoothing_kernel_bandwidth=1.5'

# Running R84-R87 on FullDataset (which should be a speedy gonsalez run). 
# Debug
python Master.py --train=1 --setting=pretrain_sub --name=Rdebug --data=FullRoboturk --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128

python cluster_run.py --partition=learnfair --name=R88 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R88 --data=FullRoboturk --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128'

python cluster_run.py --partition=learnfair --name=R89 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R89 --data=FullRoboturk --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128'

# Rerunning on original dataset with smoothening. 
python cluster_run.py --partition=learnfair --name=R90 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R90 --data=FullRoboturk --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --smoothen=1 --smoothing_kernel_bandwidth=1.5'

python cluster_run.py --partition=learnfair --name=R91 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R91 --data=FullRoboturk --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --smoothen=1 --smoothing_kernel_bandwidth=1.5'

# Rerunning 88-91 with lowered KL. 
python cluster_run.py --partition=learnfair --name=R92 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R92 --data=FullRoboturk --kl_weight=0.00001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128'

python cluster_run.py --partition=learnfair --name=R93 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R93 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128'

# Rerunning on original dataset with smoothening.
python cluster_run.py --partition=learnfair --name=R94 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R94 --data=FullRoboturk --kl_weight=0.00001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --smoothen=1 --smoothing_kernel_bandwidth=1.5'

python cluster_run.py --partition=learnfair --name=R95 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=R95 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=64 --number_layers=8 --hidden_size=128 --smoothen=1 --smoothing_kernel_bandwidth=1.5'

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

# Trying out training with conditional info.
python Master.py --train=1 --setting=learntsub --name=RJ_debugcond --kl_weight=0.001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=Roboturk --subpolicy_model=Experiment_Logs/R84/saved_models/Model_epoch60 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1 --number_layers=8 --hidden_size=128

# Running joint with R84+ models. 
python cluster_run.py --partition=learnfair --name=RJ5 --cmd='python Master.py --train=1 --setting=learntsub --name=RJ5 --kl_weight=0.0001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=Roboturk --subpolicy_model=Experiment_Logs/R84/saved_models/Model_epoch60 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1 --number_layers=8 --hidden_size=128'

python cluster_run.py --partition=learnfair --name=RJ6 --cmd='python Master.py --train=1 --setting=learntsub --name=RJ6 --kl_weight=0.001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=Roboturk --subpolicy_model=Experiment_Logs/R84/saved_models/Model_epoch60 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1 --number_layers=8 --hidden_size=128'

python cluster_run.py --partition=learnfair --name=RJ7 --cmd='python Master.py --train=1 --setting=learntsub --name=RJ7 --kl_weight=0.0001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=Roboturk --subpolicy_model=Experiment_Logs/R84/saved_models/Model_epoch80 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1 --number_layers=8 --hidden_size=128'

python cluster_run.py --partition=learnfair --name=RJ8 --cmd='python Master.py --train=1 --setting=learntsub --name=RJ8 --kl_weight=0.001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=Roboturk --subpolicy_model=Experiment_Logs/R84/saved_models/Model_epoch80 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1 --number_layers=8 --hidden_size=128'

python cluster_run.py --partition=learnfair --name=RJ9 --cmd='python Master.py --train=1 --setting=learntsub --name=RJ9 --kl_weight=0.0001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=Roboturk --subpolicy_model=Experiment_Logs/R88/saved_models/Model_epoch60 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1 --number_layers=8 --hidden_size=128'

python cluster_run.py --partition=learnfair --name=RJ10 --cmd='python Master.py --train=1 --setting=learntsub --name=RJ10 --kl_weight=0.001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=Roboturk --subpolicy_model=Experiment_Logs/R88/saved_models/Model_epoch60 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1 --number_layers=8 --hidden_size=128'

python cluster_run.py --partition=learnfair --name=RJ11 --cmd='python Master.py --train=1 --setting=learntsub --name=RJ11 --kl_weight=0.0001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=Roboturk --subpolicy_model=Experiment_Logs/R88/saved_models/Model_epoch160 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1 --number_layers=8 --hidden_size=128'

python cluster_run.py --partition=learnfair --name=RJ12 --cmd='python Master.py --train=1 --setting=learntsub --name=RJ12 --kl_weight=0.001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=Roboturk --subpolicy_model=Experiment_Logs/R88/saved_models/Model_epoch160 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1 --number_layers=8 --hidden_size=128'

# Running with R93 model 20. 
python cluster_run.py --partition=learnfair --name=RJ13 --cmd='python Master.py --train=1 --setting=learntsub --name=RJ13 --kl_weight=0.0001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=Roboturk --subpolicy_model=Experiment_Logs/R93/saved_models/Model_epoch20 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1 --number_layers=8 --hidden_size=128'

python cluster_run.py --partition=learnfair --name=RJ14 --cmd='python Master.py --train=1 --setting=learntsub --name=RJ14 --kl_weight=0.001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=Roboturk --subpolicy_model=Experiment_Logs/R93/saved_models/Model_epoch20 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1 --number_layers=8 --hidden_size=128'

# Running with R93 model 20 with smoothing. 
python cluster_run.py --partition=learnfair --name=RJ15 --cmd='python Master.py --train=1 --setting=learntsub --name=RJ15 --kl_weight=0.0001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=Roboturk --subpolicy_model=Experiment_Logs/R93/saved_models/Model_epoch20 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1 --number_layers=8 --hidden_size=128 --smoothen=1 --smoothing_kernel_bandwidth=1.5'

python cluster_run.py --partition=learnfair --name=RJ16 --cmd='python Master.py --train=1 --setting=learntsub --name=RJ16 --kl_weight=0.001 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_probability_factor=0.01 --data=Roboturk --subpolicy_model=Experiment_Logs/R93/saved_models/Model_epoch20 --latent_loss_weight=0.01 --z_dimensions=64 --traj_length=-1 --var_skill_length=1 --number_layers=8 --hidden_size=128 --smoothen=1 --smoothing_kernel_bandwidth=1.5'
