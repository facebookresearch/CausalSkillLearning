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

############################################
############################################
# RL runs with memory.
############################################
############################################

# Debug
python Master.py --train=1 --setting=downstreamRL --name=RLmemdebug --number_layers=8 --hidden_size=128 --data=MIME --environment=SawyerLift --epsilon_over=10000 --epsilon_from=0.4 --epsilon_to=0.1 --z_dimensions=0 --display_freq=50 --save_freq=50

python cluster_run.py --partition=learnfair --name=RL30 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL30 --number_layers=8 --hidden_size=128 --data=MIME --environment=SawyerLift --epsilon_over=100000 --epsilon_from=0.4 --epsilon_to=0.1 --z_dimensions=0 --display_freq=50 --save_freq=50'

python cluster_run.py --partition=learnfair --name=RL31 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL31 --number_layers=8 --hidden_size=128 --data=MIME --environment=SawyerStack --epsilon_over=100000 --epsilon_from=0.4 --epsilon_to=0.1 --z_dimensions=0 --display_freq=50 --save_freq=50'

python cluster_run.py --partition=learnfair --name=RL32 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL32 --number_layers=8 --hidden_size=128 --data=MIME --environment=SawyerPickPlace --epsilon_over=100000 --epsilon_from=0.4 --epsilon_to=0.1 --z_dimensions=0 --display_freq=50 --save_freq=50'

python cluster_run.py --partition=learnfair --name=RL33 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL33 --number_layers=8 --hidden_size=128 --data=MIME --environment=SawyerNutAssembly --epsilon_over=100000 --epsilon_from=0.4 --epsilon_to=0.1 --z_dimensions=0 --display_freq=50 --save_freq=50'

python cluster_run.py --partition=learnfair --name=RL34 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL34 --number_layers=8 --hidden_size=128 --data=MIME --environment=BaxterLift --epsilon_over=100000 --epsilon_from=0.4 --epsilon_to=0.1 --z_dimensions=0 --display_freq=50 --save_freq=50'

python cluster_run.py --partition=learnfair --name=RL35 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL35 --number_layers=8 --hidden_size=128 --data=MIME --environment=BaxterPegInHole --epsilon_over=100000 --epsilon_from=0.4 --epsilon_to=0.1 --z_dimensions=0 --display_freq=50 --save_freq=50'

# Running with MLP policy. 
# debug
python Master.py --train=1 --setting=downstreamRL --name=RLdebugmlp --hidden_size=32 --data=MIME --environment=SawyerLift --epsilon_over=100000 --epsilon_from=0.4 --epsilon_to=0.1 --z_dimensions=0 --display_freq=50 --save_freq=50 --MLP_policy=1

python cluster_run.py --partition=learnfair --name=RL36 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL36 --hidden_size=32 --data=MIME --environment=SawyerLift --epsilon_over=100000 --epsilon_from=0.4 --epsilon_to=0.1 --z_dimensions=0 --display_freq=50 --save_freq=50 --MLP_policy=1'

python cluster_run.py --partition=learnfair --name=RL37 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL37 --hidden_size=32 --data=MIME --environment=SawyerStack --epsilon_over=100000 --epsilon_from=0.4 --epsilon_to=0.1 --z_dimensions=0 --display_freq=50 --save_freq=50 --MLP_policy=1'

python cluster_run.py --partition=learnfair --name=RL38 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL38 --hidden_size=32 --data=MIME --environment=SawyerPickPlace --epsilon_over=100000 --epsilon_from=0.4 --epsilon_to=0.1 --z_dimensions=0 --display_freq=50 --save_freq=50 --MLP_policy=1'

python cluster_run.py --partition=learnfair --name=RL39 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL39 --hidden_size=32 --data=MIME --environment=SawyerNutAssembly --epsilon_over=100000 --epsilon_from=0.4 --epsilon_to=0.1 --z_dimensions=0 --display_freq=50 --save_freq=50 --MLP_policy=1'

python cluster_run.py --partition=learnfair --name=RL40 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL40 --hidden_size=32 --data=MIME --environment=BaxterLift --epsilon_over=100000 --epsilon_from=0.4 --epsilon_to=0.1 --z_dimensions=0 --display_freq=50 --save_freq=50 --MLP_policy=1'

python cluster_run.py --partition=learnfair --name=RL41 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL41 --hidden_size=32 --data=MIME --environment=BaxterPegInHole --epsilon_over=100000 --epsilon_from=0.4 --epsilon_to=0.1 --z_dimensions=0 --display_freq=50 --save_freq=50 --MLP_policy=1'

# New updates, DDPG policy, and terminal state awareness.
# Debug
python Master.py --train=1 --setting=downstreamRL --name=RLmemdebug --number_layers=8 --hidden_size=128 --data=MIME --environment=SawyerLift --epsilon_over=10000 --epsilon_from=0.4 --epsilon_to=0.1 --z_dimensions=0 --display_freq=50 --save_freq=50

python cluster_run.py --partition=learnfair --name=RL42 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL42 --number_layers=8 --hidden_size=128 --data=MIME --environment=SawyerLift --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50'

python cluster_run.py --partition=learnfair --name=RL43 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL43 --number_layers=8 --hidden_size=128 --data=MIME --environment=SawyerStack --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50'

python cluster_run.py --partition=learnfair --name=RL44 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL44 --number_layers=8 --hidden_size=128 --data=MIME --environment=SawyerPickPlace --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50'

python cluster_run.py --partition=learnfair --name=RL45 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL45 --number_layers=8 --hidden_size=128 --data=MIME --environment=SawyerNutAssembly --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50'

python cluster_run.py --partition=learnfair --name=RL46 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL46 --number_layers=8 --hidden_size=128 --data=MIME --environment=BaxterLift --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50'

python cluster_run.py --partition=learnfair --name=RL47 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL47 --number_layers=8 --hidden_size=128 --data=MIME --environment=BaxterPegInHole --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50'

# MLP version
python cluster_run.py --partition=learnfair --name=RL48 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL48 --number_layers=8 --hidden_size=128 --data=MIME --MLP_policy=1 --environment=SawyerLift --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50'

python cluster_run.py --partition=learnfair --name=RL49 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL49 --number_layers=8 --hidden_size=128 --data=MIME --MLP_policy=1 --environment=SawyerStack --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50'

python cluster_run.py --partition=learnfair --name=RL50 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL50 --number_layers=8 --hidden_size=128 --data=MIME --MLP_policy=1 --environment=SawyerPickPlace --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50'

python cluster_run.py --partition=learnfair --name=RL51 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL51 --number_layers=8 --hidden_size=128 --data=MIME --MLP_policy=1 --environment=SawyerNutAssembly --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50'

python cluster_run.py --partition=learnfair --name=RL52 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL52 --number_layers=8 --hidden_size=128 --data=MIME --MLP_policy=1 --environment=BaxterLift --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50'

python cluster_run.py --partition=learnfair --name=RL53 --cmd='python Master.py --train=1 --setting=downstreamRL --name=RL53 --number_layers=8 --hidden_size=128 --data=MIME --MLP_policy=1 --environment=BaxterPegInHole --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50'

# MC Critic with memory.
# debug
python Master.py --train=1 --setting=downstreamRL --name=RLmemdebug --number_layers=8 --hidden_size=128 --data=MIME --environment=SawyerLift --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50 --TD=0 --burn_in_eps=2

# 
python cluster_run.py --partition=learnfair --name=RL54 --cmd='python Master.py --train=1 --setting=baselineRL --name=RL54 --number_layers=8 --hidden_size=128 --data=MIME --TD=0 --environment=SawyerLift --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50'

python cluster_run.py --partition=learnfair --name=RL55 --cmd='python Master.py --train=1 --setting=baselineRL --name=RL55 --number_layers=8 --hidden_size=128 --data=MIME --TD=0 --environment=SawyerStack --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50'

python cluster_run.py --partition=learnfair --name=RL56 --cmd='python Master.py --train=1 --setting=baselineRL --name=RL56 --number_layers=8 --hidden_size=128 --data=MIME --TD=0 --environment=SawyerPickPlace --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50'

python cluster_run.py --partition=learnfair --name=RL57 --cmd='python Master.py --train=1 --setting=baselineRL --name=RL57 --number_layers=8 --hidden_size=128 --data=MIME --TD=0 --environment=SawyerNutAssembly --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50'

python cluster_run.py --partition=learnfair --name=RL58 --cmd='python Master.py --train=1 --setting=baselineRL --name=RL58 --number_layers=8 --hidden_size=128 --data=MIME --TD=0 --environment=BaxterLift --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50'

python cluster_run.py --partition=learnfair --name=RL59 --cmd='python Master.py --train=1 --setting=baselineRL --name=RL59 --number_layers=8 --hidden_size=128 --data=MIME --TD=0 --environment=BaxterPegInHole --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50'

# Running with higher memory burn in, higher epsilon, and random memory burn in. 
python cluster_run.py --partition=learnfair --name=RL60 --cmd='python Master.py --train=1 --setting=baselineRL --name=RL60 --number_layers=8 --hidden_size=128 --data=MIME --epsilon_from=0.5 --burn_in_eps=100 --environment=SawyerLift --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50'

python cluster_run.py --partition=learnfair --name=RL61 --cmd='python Master.py --train=1 --setting=baselineRL --name=RL61 --number_layers=8 --hidden_size=128 --data=MIME --epsilon_from=0.5 --burn_in_eps=100 --environment=SawyerStack --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50'

python cluster_run.py --partition=learnfair --name=RL62 --cmd='python Master.py --train=1 --setting=baselineRL --name=RL62 --number_layers=8 --hidden_size=128 --data=MIME --epsilon_from=0.5 --burn_in_eps=100 --environment=SawyerPickPlace --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50'

python cluster_run.py --partition=learnfair --name=RL63 --cmd='python Master.py --train=1 --setting=baselineRL --name=RL63 --number_layers=8 --hidden_size=128 --data=MIME --epsilon_from=0.5 --burn_in_eps=100 --environment=SawyerNutAssembly --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50'

python cluster_run.py --partition=learnfair --name=RL64 --cmd='python Master.py --train=1 --setting=baselineRL --name=RL64 --number_layers=8 --hidden_size=128 --data=MIME --epsilon_from=0.5 --burn_in_eps=100 --environment=BaxterLift --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50'

python cluster_run.py --partition=learnfair --name=RL65 --cmd='python Master.py --train=1 --setting=baselineRL --name=RL65 --number_layers=8 --hidden_size=128 --data=MIME --epsilon_from=0.5 --burn_in_eps=100 --environment=BaxterPegInHole --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50'

# Now running with low epsilon.
python cluster_run.py --partition=learnfair --name=RL66 --cmd='python Master.py --train=1 --setting=baselineRL --name=RL66 --number_layers=8 --hidden_size=128 --data=MIME --epsilon_from=0.15 --burn_in_eps=100 --environment=SawyerLift --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50'

python cluster_run.py --partition=learnfair --name=RL67 --cmd='python Master.py --train=1 --setting=baselineRL --name=RL67 --number_layers=8 --hidden_size=128 --data=MIME --epsilon_from=0.15 --burn_in_eps=100 --environment=SawyerStack --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50'

python cluster_run.py --partition=learnfair --name=RL68 --cmd='python Master.py --train=1 --setting=baselineRL --name=RL68 --number_layers=8 --hidden_size=128 --data=MIME --epsilon_from=0.15 --burn_in_eps=100 --environment=SawyerPickPlace --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50'

python cluster_run.py --partition=learnfair --name=RL69 --cmd='python Master.py --train=1 --setting=baselineRL --name=RL69 --number_layers=8 --hidden_size=128 --data=MIME --epsilon_from=0.15 --burn_in_eps=100 --environment=SawyerNutAssembly --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50'

python cluster_run.py --partition=learnfair --name=RL70 --cmd='python Master.py --train=1 --setting=baselineRL --name=RL70 --number_layers=8 --hidden_size=128 --data=MIME --epsilon_from=0.15 --burn_in_eps=100 --environment=BaxterLift --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50'

python cluster_run.py --partition=learnfair --name=RL71 --cmd='python Master.py --train=1 --setting=baselineRL --name=RL71 --number_layers=8 --hidden_size=128 --data=MIME --epsilon_from=0.15 --burn_in_eps=100 --environment=BaxterPegInHole --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50'

# Now running with ever lower epsilon.
# debug
python Master.py --train=1 --setting=baselineRL --name=RLtrial --number_layers=8 --hidden_size=128 --data=MIME --epsilon_from=0.1 --epsilon_to=0.02 --environment=SawyerLift --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50

python cluster_run.py --partition=learnfair --name=RL72 --cmd='python Master.py --train=1 --setting=baselineRL --name=RL72 --number_layers=8 --hidden_size=128 --data=MIME --epsilon_from=0.1 --epsilon_to=0.02 --burn_in_eps=100 --environment=SawyerLift --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50'

python cluster_run.py --partition=learnfair --name=RL73 --cmd='python Master.py --train=1 --setting=baselineRL --name=RL73 --number_layers=8 --hidden_size=128 --data=MIME --epsilon_from=0.1 --epsilon_to=0.02 --burn_in_eps=100 --environment=SawyerStack --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50'

python cluster_run.py --partition=learnfair --name=RL74 --cmd='python Master.py --train=1 --setting=baselineRL --name=RL74 --number_layers=8 --hidden_size=128 --data=MIME --epsilon_from=0.1 --epsilon_to=0.02 --burn_in_eps=100 --environment=SawyerPickPlace --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50'

python cluster_run.py --partition=learnfair --name=RL75 --cmd='python Master.py --train=1 --setting=baselineRL --name=RL75 --number_layers=8 --hidden_size=128 --data=MIME --epsilon_from=0.1 --epsilon_to=0.02 --burn_in_eps=100 --environment=SawyerNutAssembly --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50'

python cluster_run.py --partition=learnfair --name=RL76 --cmd='python Master.py --train=1 --setting=baselineRL --name=RL76 --number_layers=8 --hidden_size=128 --data=MIME --epsilon_from=0.1 --epsilon_to=0.02 --burn_in_eps=100 --environment=BaxterLift --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50'

python cluster_run.py --partition=learnfair --name=RL77 --cmd='python Master.py --train=1 --setting=baselineRL --name=RL77 --number_layers=8 --hidden_size=128 --data=MIME --epsilon_from=0.1 --epsilon_to=0.02 --burn_in_eps=100 --environment=BaxterPegInHole --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50'

# Also run with tiny tiny epsilon. 
python cluster_run.py --partition=learnfair --name=RL78 --cmd='python Master.py --train=1 --setting=baselineRL --name=RL78 --number_layers=8 --hidden_size=128 --data=MIME --epsilon_from=0.07 --epsilon_to=0.01 --burn_in_eps=100 --environment=SawyerLift --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50'

python cluster_run.py --partition=learnfair --name=RL79 --cmd='python Master.py --train=1 --setting=baselineRL --name=RL79 --number_layers=8 --hidden_size=128 --data=MIME --epsilon_from=0.07 --epsilon_to=0.01 --burn_in_eps=100 --environment=SawyerStack --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50'

python cluster_run.py --partition=learnfair --name=RL80 --cmd='python Master.py --train=1 --setting=baselineRL --name=RL80 --number_layers=8 --hidden_size=128 --data=MIME --epsilon_from=0.07 --epsilon_to=0.01 --burn_in_eps=100 --environment=SawyerPickPlace --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50'

python cluster_run.py --partition=learnfair --name=RL81 --cmd='python Master.py --train=1 --setting=baselineRL --name=RL81 --number_layers=8 --hidden_size=128 --data=MIME --epsilon_from=0.07 --epsilon_to=0.01 --burn_in_eps=100 --environment=SawyerNutAssembly --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50'

python cluster_run.py --partition=learnfair --name=RL82 --cmd='python Master.py --train=1 --setting=baselineRL --name=RL82 --number_layers=8 --hidden_size=128 --data=MIME --epsilon_from=0.07 --epsilon_to=0.01 --burn_in_eps=100 --environment=BaxterLift --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50'

python cluster_run.py --partition=learnfair --name=RL83 --cmd='python Master.py --train=1 --setting=baselineRL --name=RL83 --number_layers=8 --hidden_size=128 --data=MIME --epsilon_from=0.07 --epsilon_to=0.01 --burn_in_eps=100 --environment=BaxterPegInHole --epsilon_over=10000 --z_dimensions=0 --display_freq=50 --save_freq=50'


