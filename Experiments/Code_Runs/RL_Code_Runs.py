####################################################
############## Downstream RL Training ##############
####################################################

python Master.py --train=1 --setting='downstreamRL' --name=RLdebug --number_layers=8 --hidden_size=128 --data=MIME --environment=SawyerLift --epsilon_over=1000 --z_dimensions=0

python Master.py --train=1 --setting='downstreamRL' --name=RLdebug_2 --number_layers=8 --hidden_size=128 --data=MIME --environment=SawyerLift --epsilon_over=100000 --epsilon_from=0.4 --epsilon_to=0.1 --z_dimensions=0 --display_freq=50 

python Master.py --train=1 --setting='downstreamRL' --name=RLdebug_3 --number_layers=8 --hidden_size=128 --data=MIME --environment=BaxterLift --epsilon_over=1000 --z_dimensions=0 --display_freq=50

# Actual roboturk runs. 
python cluster_run.py --partition=learnfair --name=R1 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL1 --number_layers=8 --hidden_size=128 --data=MIME --environment=SawyerLift --epsilon_over=10000 --z_dimensions=0 --display_freq=50'

python cluster_run.py --partition=learnfair --name=R2 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL2 --number_layers=8 --hidden_size=128 --data=MIME --environment=SawyerStack --epsilon_over=10000 --z_dimensions=0 --display_freq=50'

python cluster_run.py --partition=learnfair --name=R3 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL3 --number_layers=8 --hidden_size=128 --data=MIME --environment=SawyerPickPlace --epsilon_over=10000 --z_dimensions=0 --display_freq=50'

python cluster_run.py --partition=learnfair --name=R4 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL4 --number_layers=8 --hidden_size=128 --data=MIME --environment=SawyerNutAssembly --epsilon_over=10000 --z_dimensions=0 --display_freq=50'

python cluster_run.py --partition=learnfair --name=R5 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL5 --number_layers=8 --hidden_size=128 --data=MIME --environment=BaxterLift --epsilon_over=10000 --z_dimensions=0 --display_freq=50'

python cluster_run.py --partition=learnfair --name=R6 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL6 --number_layers=8 --hidden_size=128 --data=MIME --environment=BaxterPegInHole --epsilon_over=10000 --z_dimensions=0 --display_freq=50'

# Run with increased number of episodes.
python cluster_run.py --partition=learnfair --name=R7 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL7 --number_layers=8 --hidden_size=128 --data=MIME --environment=SawyerLift --epsilon_over=10000 --z_dimensions=0 --display_freq=50'

python cluster_run.py --partition=learnfair --name=R8 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL8 --number_layers=8 --hidden_size=128 --data=MIME --environment=SawyerStack --epsilon_over=10000 --z_dimensions=0 --display_freq=50'

python cluster_run.py --partition=learnfair --name=R9 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL9 --number_layers=8 --hidden_size=128 --data=MIME --environment=SawyerPickPlace --epsilon_over=10000 --z_dimensions=0 --display_freq=50'

python cluster_run.py --partition=learnfair --name=R10 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL10 --number_layers=8 --hidden_size=128 --data=MIME --environment=SawyerNutAssembly --epsilon_over=10000 --z_dimensions=0 --display_freq=50'

python cluster_run.py --partition=learnfair --name=R11 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL11 --number_layers=8 --hidden_size=128 --data=MIME --environment=BaxterLift --epsilon_over=10000 --z_dimensions=0 --display_freq=50'

python cluster_run.py --partition=learnfair --name=R12 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL12 --number_layers=8 --hidden_size=128 --data=MIME --environment=BaxterPegInHole --epsilon_over=10000 --z_dimensions=0 --display_freq=50'

# Trying to debug....
python Master.py --train=1 --setting='downstreamRL' --name=RL13 --number_layers=8 --hidden_size=128 --data=MIME --environment=SawyerLift --epsilon_over=100000 --epsilon_from=0.4 --epsilon_to=0.1 --z_dimensions=0 --display_freq=50 

# running with actual epsilon.
python Master.py --train=1 --setting='downstreamRL' --name=RL14 --number_layers=8 --hidden_size=128 --data=MIME --environment=SawyerLift --epsilon_over=100000 --epsilon_from=0.4 --epsilon_to=0.1 --z_dimensions=0 --display_freq=50 

# Run with TD. 
python Master.py --train=1 --setting='downstreamRL' --name=RL15 --number_layers=8 --hidden_size=128 --data=MIME --environment=SawyerLift --epsilon_over=100000 --epsilon_from=0.4 --epsilon_to=0.1 --z_dimensions=0 --display_freq=50 --TD=1

# Rerun RL14 and RL15 with differentiable critic inputs. 
python Master.py --train=1 --setting='downstreamRL' --name=RL16 --number_layers=8 --hidden_size=128 --data=MIME --environment=SawyerLift --epsilon_over=100000 --epsilon_from=0.4 --epsilon_to=0.1 --z_dimensions=0 --display_freq=50 

# Run with TD. 
python Master.py --train=1 --setting='downstreamRL' --name=RL17 --number_layers=8 --hidden_size=128 --data=MIME --environment=SawyerLift --epsilon_over=100000 --epsilon_from=0.4 --epsilon_to=0.1 --z_dimensions=0 --display_freq=50 --TD=1

############################################
# Running RL on suite of tasks with and without TD. 
python cluster_run.py --partition=learnfair --name=R18 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL18 --number_layers=8 --hidden_size=128 --data=MIME --environment=SawyerLift --epsilon_over=100000 --epsilon_from=0.4 --epsilon_to=0.1 --z_dimensions=0 --display_freq=50'

python cluster_run.py --partition=learnfair --name=R19 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL19 --number_layers=8 --hidden_size=128 --data=MIME --environment=SawyerLift --epsilon_over=100000 --epsilon_from=0.4 --epsilon_to=0.1 --z_dimensions=0 --display_freq=50 --TD=1'

python cluster_run.py --partition=learnfair --name=R20 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL20 --number_layers=8 --hidden_size=128 --data=MIME --environment=SawyerStack --epsilon_over=100000 --epsilon_from=0.4 --epsilon_to=0.1 --z_dimensions=0 --display_freq=50'

python cluster_run.py --partition=learnfair --name=R21 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL21 --number_layers=8 --hidden_size=128 --data=MIME --environment=SawyerStack --epsilon_over=100000 --epsilon_from=0.4 --epsilon_to=0.1 --z_dimensions=0 --display_freq=50 --TD=1'

python cluster_run.py --partition=learnfair --name=R22 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL22 --number_layers=8 --hidden_size=128 --data=MIME --environment=SawyerPickPlace --epsilon_over=100000 --epsilon_from=0.4 --epsilon_to=0.1 --z_dimensions=0 --display_freq=50'

python cluster_run.py --partition=learnfair --name=R23 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL23 --number_layers=8 --hidden_size=128 --data=MIME --environment=SawyerPickPlace --epsilon_over=100000 --epsilon_from=0.4 --epsilon_to=0.1 --z_dimensions=0 --display_freq=50 --TD=1'

python cluster_run.py --partition=learnfair --name=R24 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL24 --number_layers=8 --hidden_size=128 --data=MIME --environment=SawyerNutAssembly --epsilon_over=100000 --epsilon_from=0.4 --epsilon_to=0.1 --z_dimensions=0 --display_freq=50'

python cluster_run.py --partition=learnfair --name=R25 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL25 --number_layers=8 --hidden_size=128 --data=MIME --environment=SawyerNutAssembly --epsilon_over=100000 --epsilon_from=0.4 --epsilon_to=0.1 --z_dimensions=0 --display_freq=50 --TD=1'

python cluster_run.py --partition=learnfair --name=RL26 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL26 --number_layers=8 --hidden_size=128 --data=MIME --environment=BaxterLift --epsilon_over=100000 --epsilon_from=0.4 --epsilon_to=0.1 --z_dimensions=0 --display_freq=50'

python cluster_run.py --partition=learnfair --name=RL27 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL27 --number_layers=8 --hidden_size=128 --data=MIME --environment=BaxterLift --epsilon_over=100000 --epsilon_from=0.4 --epsilon_to=0.1 --z_dimensions=0 --display_freq=50 --TD=1'

python cluster_run.py --partition=learnfair --name=RL28 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL28 --number_layers=8 --hidden_size=128 --data=MIME --environment=BaxterPegInHole --epsilon_over=100000 --epsilon_from=0.4 --epsilon_to=0.1 --z_dimensions=0 --display_freq=50'

python cluster_run.py --partition=learnfair --name=RL29 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL29 --number_layers=8 --hidden_size=128 --data=MIME --environment=BaxterPegInHole --epsilon_over=100000 --epsilon_from=0.4 --epsilon_to=0.1 --z_dimensions=0 --display_freq=50 --TD=1'
