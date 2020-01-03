# Debug
python Master.py --train=1 --setting=pretrain_sub --name=Bdebug_MIME --data=MIME --number_layers=8 --hidden_size=128 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --traj_segments=0

# Running LSTM training on MIME data.
python cluster_run.py --partition=learnfair --name=B1_MIME --cmd='python Master.py --train=1 --setting=pretrain_sub --name=B1_MIME --data=MIME --number_layers=8 --hidden_size=128 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --traj_segments=0'