# Duplicating good pretraining run on new dataset. 
python cluster_run.py --partition=learnfair --name=M47 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M47 --data=MIME --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --normalization=minmax'

# Speedy gonzales runs. 
python cluster_run.py --partition=learnfair --name=M150 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M150 --data=MIME --number_layers=8 --hidden_size=128 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M151 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M151 --data=MIME --number_layers=8 --hidden_size=128 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M152 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M152 --data=MIME --number_layers=8 --hidden_size=128 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64'

python cluster_run.py --partition=learnfair --name=M153 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M153 --data=MIME --number_layers=8 --hidden_size=128 --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --normalization=meanvar'

python cluster_run.py --partition=learnfair --name=M154 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M154 --data=MIME --number_layers=8 --hidden_size=128 --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --normalization=minmax'

python cluster_run.py --partition=learnfair --name=M155 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M155 --data=MIME --number_layers=8 --hidden_size=128 --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64'


