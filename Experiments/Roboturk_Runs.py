# Start pretraining roboturk.

# Debug run.
python Master.py --train=1 --setting=pretrain_sub --name=R1 --data=Roboturk --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64

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