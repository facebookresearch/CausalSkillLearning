# Debug
python Master.py --train=1 --setting=pretrain_sub --name=Bdebug_MIME --data=MIME --number_layers=8 --hidden_size=128 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --traj_segments=0
# Debug eval
python Master.py --train=0 --setting=pretrain_sub --name=Bdebug_MIME --data=MIME --number_layers=8 --hidden_size=128 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --traj_segments=0 --model=Experiment_Logs/Bdebug_MIME/saved_models/Model_epoch0

# Running LSTM training on MIME data.
python cluster_run.py --partition=learnfair --name=B1_MIME --cmd='python Master.py --train=1 --setting=pretrain_sub --name=B1_MIME --data=MIME --number_layers=8 --hidden_size=128 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --traj_segments=0'

# Running LSTM training on Roboturk data.
python cluster_run.py --partition=learnfair --name=B2_RTurk --cmd='python Master.py --train=1 --setting=pretrain_sub --name=B2_RTurk --data=Roboturk --number_layers=8 --hidden_size=128 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --traj_segments=0'

# Running baseline on new Roboturk data. 
python cluster_run.py --partition=learnfair --name=B3_RTurk --cmd='python Master.py --train=1 --setting=pretrain_sub --name=B3_RTurk --data=Roboturk --number_layers=8 --hidden_size=128 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --traj_segments=0'

# Running baselines on new smoothed roboturk data, and mime data with a test set. 
python cluster_run.py --partition=learnfair --name=B4_MIME --cmd='python Master.py --train=1 --setting=pretrain_sub --name=B4_MIME --data=MIME --number_layers=8 --hidden_size=128 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --traj_segments=0'

python cluster_run.py --partition=learnfair --name=B5_RTurk --cmd='python Master.py --train=1 --setting=pretrain_sub --name=B5_RTurk --data=Roboturk --number_layers=8 --hidden_size=128 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --traj_segments=0'

# Rerun B4-B5 with proper viz. 
python cluster_run.py --partition=learnfair --name=B6_MIME --cmd='python Master.py --train=1 --setting=pretrain_sub --name=B6_MIME --data=MIME --number_layers=8 --hidden_size=128 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --traj_segments=0'

python cluster_run.py --partition=learnfair --name=B7_RTurk --cmd='python Master.py --train=1 --setting=pretrain_sub --name=B7_RTurk --data=Roboturk --number_layers=8 --hidden_size=128 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --traj_segments=0'

# Rerun B7 without visualization.
python Master.py --train=1 --setting=pretrain_sub --name=B8_RTurk --data=FullRoboturk --number_layers=8 --hidden_size=128 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --traj_segments=0


# Running Flat DMP Baseline on MIME Data. 
python Master.py --name=FlatDMP_MIME --setting=FlatDMP --data=MIME

# Running multiple baselines
python Master.py --name=FlatDMP_MIME --setting=DMP --data=MIME

python Master.py --name=FlatDMP_MIME --setting=DMP --data=Roboturk

# Running DMP baselines with various parameters. 
python Master.py --name=FlatDMP_MIME --setting=DMP --data=Roboturk

# RErunning DMP on FullDRoboturk
python Master.py --name=FlatDMP_MIME --setting=DMP --data=FullRoboturk --smoothen=1 --smoothing_kernel_bandwidth=1.5