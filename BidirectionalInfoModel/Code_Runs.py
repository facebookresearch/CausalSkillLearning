# New experiments to run: 

# T50 with variational loss.
python cluster_run.py --name=T50_justa_vbloss_len10 --cmd='python Master.py --train=1 --just_actions=1 --traj_length=10 --name=T50_justa_vbloss_len10'

# T51:
python cluster_run.py --name=T51_justa_vbloss_lenfull --cmd='python Master.py --train=1 --just_actions=1 --traj_length=-1 --name=T51_justa_vbloss_lenfull'

# T52: Rerunning T51 to make sure it wasn't a fluke.
python cluster_run.py --name=T52_justa_vbloss_len10 --cmd='python Master.py --train=1 --just_actions=1 --traj_length=10 --name=T52_justa_vbloss_len10'

# T53: Now actually running with full length - RERUN.
python cluster_run.py --name=T53_justa_vbloss_lenfull --cmd='python Master.py --train=1 --just_actions=1 --traj_length=-1 --name=T53_justa_vbloss_lenfull'

# T54: Running T51, T52 with different lambda likelihood penalty to try to learn latent_b correctly. 
python cluster_run.py --name=T54_justa_vblosswt5_len10 --cmd='python Master.py --train=1 --just_actions=1 --traj_length=10 --likelihood_penalty=5 --name=T54_justa_vblosswt5_len10'

# T55: Running T51, T52 with different lambda likelihood penalty to try to learn latent_b correctly. 
python cluster_run.py --name=T55_justa_vblosswt2_len10 --cmd='python Master.py --train=1 --just_actions=1 --traj_length=10 --likelihood_penalty=5 --name=T55_justa_vblosswt2_len10'

# # T56: Running T51, T52 with different lambda likelihood penalty to try to learn latent_b correctly with full traj len. 
# python cluster_run.py --name=T56_justa_vblosswt5_lenfull --cmd='python Master.py --train=1 --just_actions=1 --traj_length=-1 --likelihood_penalty=5 --name=T56_justa_vblosswt5_lenfull'

# # New T56: Running with trajectory length 20. 
python cluster_run.py --name=T56_justa_len20 --cmd='python Master.py --train=1 --just_actions=1 --traj_length=20 --name=T56_justa_len20'

# # T57: Running T51, T52 with different lambda likelihood penalty to try to learn latent_b correctly with full traj len.
# python cluster_run.py --name=T57_justa_vblosswt2_lenfull --cmd='python Master.py --train=1 --just_actions=1 --traj_length=-1 --likelihood_penalty=5 --name=T57_justa_vblosswt2_lenfull'

# T58: Running T51, T52 with different lambda likelihood penalty to try to learn latent_b correctly. 
python cluster_run.py --name=T58_justa_vblosswt2_len10 --cmd='python Master.py --train=1 --just_actions=1 --traj_length=10 --likelihood_penalty=2 --name=T58_justa_vblosswt2_len10'

# T58 eval:
python Master.py --train=0 --just_actions=1 --traj_length=10 --likelihood_penalty=2 --name=T58_justa_vblosswt2_len10_eval --model=Experiment_Logs/T58_justa_vblosswt2_len10/Model_epoch30

# # T59: Running T51, T52 with different lambda likelihood penalty to try to learn latent_b correctly with full traj len.
# python cluster_run.py --name=T59_justa_vblosswt2_lenfull --cmd='python Master.py --train=1 --just_actions=1 --traj_length=-1 --likelihood_penalty=2 --name=T59_justa_vblosswt2_lenfull'

# T60, T61, T62, T63: Running with len 10 and latent policies, and latent and subpolicy likelihoods, without variational entropy.
python cluster_run.py --name=T60_latentz_gtsub_len10_sbrat0pt01_llpen10 --cmd='python Master.py --train=1 --just_actions=0 --traj_length=10 --likelihood_penalty=10 --subpolicy_ratio=0.01 --name=T60_latentz_gtsub_len10_sbrat0pt01_llpen10'

python cluster_run.py --name=T61_latentz_gtsub_len10_sbrat0pt1_llpen10 --cmd='python Master.py --train=1 --just_actions=0 --traj_length=10 --likelihood_penalty=10 --subpolicy_ratio=0.1 --name=T61_latentz_gtsub_len10_sbrat0pt1_llpen10'

python cluster_run.py --name=T62_latentz_gtsub_len10_sbrat0pt001_llpen10 --cmd='python Master.py --train=1 --just_actions=0 --traj_length=10 --likelihood_penalty=10 --subpolicy_ratio=0.001 --name=T62_latentz_gtsub_len10_sbrat0pt001_llpen10'

python cluster_run.py --name=T63_latentz_gtsub_len10_sbrat1pt0_llpen10 --cmd='python Master.py --train=1 --just_actions=0 --traj_length=10 --likelihood_penalty=10 --subpolicy_ratio=1.0 --name=T63_latentz_gtsub_len10_sbrat1pt0_llpen10'

# Try T60-T63 with variational entropy term.
python cluster_run.py --name=T64_latentz_gtsub_varent_len10_sbrat0pt01_llpen10 --cmd='python Master.py --train=1 --just_actions=0 --traj_length=10 --likelihood_penalty=10 --subpolicy_ratio=0.01 --name=T64_latentz_gtsub_varent_len10_sbrat0pt01_llpen10 --entropy=1'

python cluster_run.py --name=T65_latentz_gtsub_varent_len10_sbrat0pt1_llpen10 --cmd='python Master.py --train=1 --just_actions=0 --traj_length=10 --likelihood_penalty=10 --subpolicy_ratio=0.1 --name=T65_latentz_gtsub_varent_len10_sbrat0pt1_llpen10 --entropy=1'

python cluster_run.py --name=T66_latentz_gtsub_varent_len10_sbrat0pt001_llpen10 --cmd='python Master.py --train=1 --just_actions=0 --traj_length=10 --likelihood_penalty=10 --subpolicy_ratio=0.001 --name=T66_latentz_gtsub_varent_len10_sbrat0pt001_llpen10 --entropy=1'

python cluster_run.py --name=T67_latentz_gtsub_varent_len10_sbrat1pt0_llpen10 --cmd='python Master.py --train=1 --just_actions=0 --traj_length=10 --likelihood_penalty=10 --subpolicy_ratio=1.0 --name=T67_latentz_gtsub_varent_len10_sbrat1pt0_llpen10 --entropy=1'

# # DEBUG RUNS AND T68 are useless.
# # These "debugging" tries ended up working. DebugObj is really good. 
# python Master.py --train=1 --just_actions=1 --traj_length=10 --likelihood_penalty=2 --name=DebugObj --ent_weight=0.
# # python Master.py --train=0 --just_actions=1 --traj_length=10 --likelihood_penalty=2 --name=DebugObj_Eval --ent_weight=0. --model=Experiment_Logs/DebugObj/Model_epoch20


# python Master.py --train=1 --just_actions=1 --traj_length=10 --likelihood_penalty=2 --name=DebugObjEnt --ent_weight=0. --entropy=1
# python Master.py --train=1 --just_actions=1 --traj_length=10 --likelihood_penalty=2 --name=DebugObjEntRat01 --ent_weight=0. --entropy=1 --subpolicy_ratio=0.1

# # Run T68 onwards without subpolicy and latent entropy, but with variational entropy. 
# python cluster_run.py --name=T68_latentz_gtsub_varent_nolatent_subrat0pt10 --cmd='python Master.py --train=1 --just_actions=1 --traj_length=10 --likelihood_penalty=2 --name=T68_latentz_gtsub_varent_nolatent_subrat0pt10 --ent_weight=0. --entropy=0 --var_entropy=1'

python Master.py --train=1 --just_actions=0 --traj_length=10 --likelihood_penalty=2 --name=T69_DebugObj --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1

python Master.py --train=1 --just_actions=0 --traj_length=10 --likelihood_penalty=2 --name=T70_DebugObj_NoEnt --ent_weight=0. --entropy=0 --var_entropy=0 --subpolicy_ratio=0.1

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --just_actions=0 --traj_length=5 --likelihood_penalty=2 --name=T71_DebugObj_len5 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1
# CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --just_actions=0 --traj_length=5 --likelihood_penalty=2 --name=T71_DebugObj_len5_eval --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --model=Experiment_Logs/T71_DebugObj_len5/saved_models/Model_epoch2

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --just_actions=0 --traj_length=5 --likelihood_penalty=2 --name=T72_DebugObj_NoEnt_len5 --ent_weight=0. --entropy=0 --var_entropy=0 --subpolicy_ratio=0.1
# CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --just_actions=0 --traj_length=5 --likelihood_penalty=2 --name=T72_DebugObj_NoEnt_len5_eval --ent_weight=0. --entropy=0 --var_entropy=0 --subpolicy_ratio=0.1 --model=Experiment_Logs/T72_DebugObj_NoEnt_len5/saved_models/Model_epoch3

#
CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --just_actions=0 --traj_length=10 --likelihood_penalty=2 --name=T73_DebugObj_len10_subrat0pt5 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --just_actions=0 --traj_length=10 --likelihood_penalty=2 --name=T74_DebugObj_NoEnt_len10_subrat0pt5 --ent_weight=0. --entropy=0 --var_entropy=0 --subpolicy_ratio=0.5

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --just_actions=0 --traj_length=5 --likelihood_penalty=2 --name=T75_DebugObj_len5_subrat0pt5 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --just_actions=0 --traj_length=5 --likelihood_penalty=2 --name=T76_DebugObj_NoEnt_len5_subrat0pt5 --ent_weight=0. --entropy=0 --var_entropy=0 --subpolicy_ratio=0.5

# python Master.py --train=1 --just_actions=0 --traj_length=5 --likelihood_penalty=2 --name=Tdebug_latent --ent_weight=0. --entropy=0 --var_entropy=0 --subpolicy_ratio=0.5

# T77 running with new initialization of LatentPolicyNetwork LSTM. 
python Master.py --train=1 --just_actions=0 --traj_length=5 --likelihood_penalty=2 --name=T77_trial --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1
# Eval: 
# python Master.py --train=0 --just_actions=0 --traj_length=5 --likelihood_penalty=2 --name=T77_trial_eval --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --model=Experiment_Logs/T77_trial/saved_models/Model_epoch1

# T78 run to evaluate latent likelihoods and reweight them. 
python Master.py --train=1 --just_actions=0 --traj_length=5 --likelihood_penalty=2 --name=T78_trial --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1
# Eval:
# python Master.py --train=1 --just_actions=0 --traj_length=5 --likelihood_penalty=2 --name=T78_trial_eval --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --model=Experiment_Logs/T78_trial/saved_models/Model_epoch1

# T79 run to explore better b's.
python Master.py --train=1 --just_actions=0 --traj_length=5 --likelihood_penalty=2 --name=T79_trial_bexbias0pt2 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3
# Eval:
# python Master.py --train=0 --just_actions=0 --traj_length=5 --likelihood_penalty=2 --name=T79_trial_bexbias0pt2_eval --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --model=Experiment_Logs/T79_trial_bexbias0pt2/saved_models/Model_epoch1

# T80 run to explore better b's by increasing likelihood penalty.
python Master.py --train=1 --just_actions=0 --traj_length=5 --likelihood_penalty=5 --name=T80_trial_bexbias0pt2_llp5 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3
# Eval:
# python Master.py --train=0 --just_actions=0 --traj_length=5 --likelihood_penalty=5 --name=T80_trial_bexbias0pt2_llp5_eval --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --model=Experiment_Logs/T80_trial_bexbias0pt2_llp5/saved_models/Model_epoch1

T81:
python -m backports.pdb Master.py --train=1 --just_actions=0 --traj_length=5 --likelihood_penalty=5 --name=T81_trial_bexbias0pt2_llp5 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3

# T82 With Likelihood penalty = 5, and b exploration bias = 0.3:
python Master.py --train=1 --just_actions=0 --traj_length=5 --likelihood_penalty=5 --name=T82_trial_bexbias0pt2_llp5 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3
# Eval
python Master.py --train=0 --just_actions=0 --traj_length=5 --likelihood_penalty=5 --name=T82_trial_bexbias0pt2_llp5_eval --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --model=Experiment_Logs/T82_trial_bexbias0pt2_llp5/saved_models/Model_epoch3


# T83: Rerunning with T82 settings.
python cluster_run.py --name=T83_latentz_gtsub_varent_len5_bexbias0pt3_llp5 --cmd='python Master.py --train=1 --just_actions=0 --traj_length=5 --likelihood_penalty=5 --name=T83_latentz_gtsub_varent_len5_bexbias0pt3_llp5 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3'

# T84: Running with T82 settings with trajectory length = 10.
python cluster_run.py --name=T84_latentz_gtsub_varent_len10_bexbias0pt3_llp5 --cmd='python Master.py --train=1 --just_actions=0 --traj_length=10 --likelihood_penalty=5 --name=T84_latentz_gtsub_varent_len10_bexbias0pt3_llp5 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3'
# Eval
python Master.py --train=0 --just_actions=0 --traj_length=10 --likelihood_penalty=5 --name=T84_latentz_gtsub_varent_len10_bexbias0pt3_llp5_eval --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --model=Experiment_Logs/T84_latentz_gtsub_varent_len10_bexbias0pt3_llp5/saved_models/Model_epoch7

# T85: Running with T82 settings with full trajectory length.
python cluster_run.py --name=T85_latentz_gtsub_varent_lenfull_bexbias0pt3_llp5 --cmd='python Master.py --train=1 --just_actions=0 --traj_length=-1 --likelihood_penalty=5 --name=T85_latentz_gtsub_varent_lenfull_bexbias0pt3_llp5 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3'
# Eval
python Master.py --train=0 --just_actions=0 --traj_length=-1 --likelihood_penalty=5 --name=T85_latentz_gtsub_varent_lenfull_bexbias0pt3_llp5_eval --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --model=Experiment_Logs/T85_latentz_gtsub_varent_lenfull_bexbias0pt3_llp5/saved_models/Model_epoch4


# Running things with new trajectory representation. 
python cluster_run.py --name=T86_latentz_gtsub_varent_len5_bexbias0pt3_llp5_newrep --cmd='python Master.py --train=1 --setting=gtsub --traj_length=5 --likelihood_penalty=5 --name=T86_latentz_gtsub_varent_len5_bexbias0pt3_llp5_newrep --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3'
# Eval
python Master.py --train=0 --setting=gtsub --traj_length=5 --likelihood_penalty=5 --name=T86_latentz_gtsub_varent_len5_bexbias0pt3_llp5_newrep_eval --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --model=Experiment_Logs/T86_latentz_gtsub_varent_len5_bexbias0pt3_llp5_newrep/saved_models/Model_epoch2

python cluster_run.py --name=T87_latentz_gtsub_varent_len10_bexbias0pt3_llp5_newrep --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T87_latentz_gtsub_varent_len10_bexbias0pt3_llp5_newrep --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3'
# Eval 
python Master.py --train=0 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T87_latentz_gtsub_varent_len10_bexbias0pt3_llp5_newrep_eval --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --model=Experiment_Logs/T87_latentz_gtsub_varent_len10_bexbias0pt3_llp5_newrep/saved_models/Model_epoch4

python cluster_run.py --name=T88_latentz_gtsub_varent_lenfull_bexbias0pt3_llp5_newrep --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T88_latentz_gtsub_varent_lenfull_bexbias0pt3_llp5_newrep --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3'

# Running things with updated correct trajectory representation. 
python cluster_run.py --name=T89_latentz_gtsub_varent_len5_bexbias0pt3_llp5_newrep --cmd='python Master.py --train=1 --setting=gtsub --traj_length=5 --likelihood_penalty=5 --name=T89_latentz_gtsub_varent_len5_bexbias0pt3_llp5_newrep --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3'

python cluster_run.py --name=T90_latentz_gtsub_varent_len10_bexbias0pt3_llp5_newrep --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T90_latentz_gtsub_varent_len10_bexbias0pt3_llp5_newrep --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3'

python cluster_run.py --name=T91_latentz_gtsub_varent_lenfull_bexbias0pt3_llp5_newrep --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T91_latentz_gtsub_varent_lenfull_bexbias0pt3_llp5_newrep --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3'


# Running things with correct trajectory representation and large initialization for the latent policy. 
python cluster_run.py --name=T92_latentz_gtsub_varent_len5_bexbias0pt3_llp5_newrep --cmd='python Master.py --train=1 --setting=gtsub --traj_length=5 --likelihood_penalty=5 --name=T92_latentz_gtsub_varent_len5_bexbias0pt3_llp5_newrep --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3'
# Eval
python Master.py --train=0 --setting=gtsub --traj_length=5 --likelihood_penalty=5 --name=T92_latentz_gtsub_varent_len5_bexbias0pt3_llp5_newrep_eval --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --model=Experiment_Logs/T92_latentz_gtsub_varent_len5_bexbias0pt3_llp5_newrep/saved_models/Model_epoch5

python cluster_run.py --name=T93_latentz_gtsub_varent_len10_bexbias0pt3_llp5_newrep --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T93_latentz_gtsub_varent_len10_bexbias0pt3_llp5_newrep --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3'

python cluster_run.py --name=T94_latentz_gtsub_varent_lenfull_bexbias0pt3_llp5_newrep --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T94_latentz_gtsub_varent_lenfull_bexbias0pt3_llp5_newrep --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3'


# # Debugging CUDNN error.
# python Master.py --train=1 --setting=gtsub --traj_length=5 --likelihood_penalty=5 --name=T89_latentz_gtsub_varent_len5_bexbias0pt3_llp5_newrep_r2 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3

# CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T90_latentz_gtsub_varent_len10_bexbias0pt3_llp5_newrep_r2 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3

# python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T91_latentz_gtsub_varent_lenfull_bexbias0pt3_llp5_newrep_r2 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3

# # Debugging CUDNN error again.
# CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=gtsub --traj_length=5 --likelihood_penalty=5 --name=T89_r3 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3

# CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T90_r3 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3

# python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T91_r3 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3

# Running things with correct trajectory representation and large initialization for the latent policy. 
python cluster_run.py --name=T100_latentz_gtsub_varent_len5 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=5 --likelihood_penalty=5 --name=T100_latentz_gtsub_varent_len5 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir'
# Eval
python Master.py --train=0 --setting=gtsub --traj_length=5 --likelihood_penalty=5 --name=T100_latentz_gtsub_varent_len5_eval --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --model=Experiment_Logs/T100_latentz_gtsub_varent_len5/saved_models/Model_epoch10

python cluster_run.py --name=T101_latentz_gtsub_varent_len10 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T101_latentz_gtsub_varent_len10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir'
# Eval
python Master.py --train=0 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T101_latentz_gtsub_varent_len10_eval --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --model=Experiment_Logs/T101_latentz_gtsub_varent_len10/saved_models/Model_epoch30

python cluster_run.py --name=T102_latentz_gtsub_varent_lenfull --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T102_latentz_gtsub_varent_lenfull --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir'

# Running on directed continuous trajectories with z_exploration bias. 
python cluster_run.py --name=T103_latentz_gtsub_varent_len10_zex0pt3 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T103_latentz_gtsub_varent_len10_zex0pt3 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3'
# Eval
python Master.py --train=0 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T103_latentz_gtsub_varent_len10_zex0pt3_eval --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3 --model=Experiment_Logs/T103_latentz_gtsub_varent_len10_zex0pt3/saved_models/Model_epoch12

python cluster_run.py --name=T104_latentz_gtsub_varent_len10_zex0pt5 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T104_latentz_gtsub_varent_len10_zex0pt5 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.5'

python cluster_run.py --name=T105_latentz_gtsub_varent_len10_zex1pt0 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T105_latentz_gtsub_varent_len10_zex1pt0 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=1.'

# Loading len5 model and trying.
python cluster_run.py --name=T106_latentz_gtsub_varent_len10_loadlen5 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T106_latentz_gtsub_varent_len10_loadlen5 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --model=Experiment_Logs/T100_latentz_gtsub_varent_len5/saved_models/Model_epoch20'

# Running on directed continuous trajectories with z_exploration bias with fulllen.
python cluster_run.py --name=T107_latentz_gtsub_varent_lenfull_zex0pt3 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T107_latentz_gtsub_varent_lenfull_zex0pt3 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3'

python cluster_run.py --name=T108_latentz_gtsub_varent_lenfull_zex0pt5 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T108_latentz_gtsub_varent_lenfull_zex0pt5 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.5'

python cluster_run.py --name=T109_latentz_gtsub_varent_lenfull_zex1pt0 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T109_latentz_gtsub_varent_lenfull_zex1pt0 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=1.'

# Loading len5 model and trying.
python cluster_run.py --name=T110_latentz_gtsub_varent_lenfull_loadlen5 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T110_latentz_gtsub_varent_lenfull_loadlen5 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --model=Experiment_Logs/T100_latentz_gtsub_varent_len5/saved_models/Model_epoch20'

# Running on directed continuous trajectories with z_exploration bias with fulllen.
python cluster_run.py --name=T111_latentz_gtsub_varent_len7_zex0 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=7 --likelihood_penalty=5 --name=T111_latentz_gtsub_varent_len7_zex0 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.'

python cluster_run.py --name=T112_latentz_gtsub_varent_len7_zex0pt3 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=7 --likelihood_penalty=5 --name=T112_latentz_gtsub_varent_len7_zex0pt3 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3'
# Eval:
python Master.py --train=0 --setting=gtsub --traj_length=7 --likelihood_penalty=5 --name=T112_latentz_gtsub_varent_len7_zex0pt3_eval --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3 --model=Experiment_Logs/T112_latentz_gtsub_varent_len7_zex0pt3/saved_models/Model_epoch11

python cluster_run.py --name=T113_latentz_gtsub_varent_len7_zex0pt5 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=7 --likelihood_penalty=5 --name=T113_latentz_gtsub_varent_len7_zex0pt5 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.5'

python cluster_run.py --name=T114_latentz_gtsub_varent_len7_zex1pt0 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=7 --likelihood_penalty=5 --name=T114_latentz_gtsub_varent_len7_zex1pt0 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=1.'

# Actually Loading len5 model and trying.
python cluster_run.py --name=T115_latentz_gtsub_varent_len10_loadlen5 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T115_latentz_gtsub_varent_len10_loadlen5 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --model=Experiment_Logs/T100_latentz_gtsub_varent_len5/saved_models/Model_epoch20'

# Actually Loading len5 model and trying.
python cluster_run.py --name=T116_latentz_gtsub_varent_lenfull_loadlen5 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T116_latentz_gtsub_varent_lenfull_loadlen5 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --model=Experiment_Logs/T100_latentz_gtsub_varent_len5/saved_models/Model_epoch20'

# Running with lowered subpolicy ratio for exploration bias = 0. and 0.3.
python cluster_run.py --name=T117_latentz_gtsub_len10_zex0_subrat0pt05 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T117_latentz_gtsub_len10_zex0_subrat0pt05 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.05 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.'

python cluster_run.py --name=T118_latentz_gtsub_len10_zex0_subrat0pt01 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T118_latentz_gtsub_len10_zex0_subrat0pt01 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.01 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.'

python cluster_run.py --name=T119_latentz_gtsub_len10_zex0_subrat0pt005 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T119_latentz_gtsub_len10_zex0_subrat0pt005 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.005 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.'

python cluster_run.py --name=T120_latentz_gtsub_len10_zex0_subrat0pt001 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T120_latentz_gtsub_len10_zex0_subrat0pt001 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.001 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.'

# With z ex bias = 0.3
python cluster_run.py --name=T121_latentz_gtsub_len10_zex0pt3_subrat0pt05 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T121_latentz_gtsub_len10_zex0pt3_subrat0pt05 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.05 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3'

python cluster_run.py --name=T122_latentz_gtsub_len10_zex0pt3_subrat0pt01 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T122_latentz_gtsub_len10_zex0pt3_subrat0pt01 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.01 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3'

python cluster_run.py --name=T123_latentz_gtsub_len10_zex0pt3_subrat0pt005 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T123_latentz_gtsub_len10_zex0pt3_subrat0pt005 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.005 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3'

python cluster_run.py --name=T124_latentz_gtsub_len10_zex0pt3_subrat0pt001 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T124_latentz_gtsub_len10_zex0pt3_subrat0pt001 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.001 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3'


# Run with increased subpolicy ratio.
python cluster_run.py --name=T125_latentz_gtsub_len10_zex0_subrat0pt5 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T125_latentz_gtsub_len10_zex0_subrat0pt5 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.'

python cluster_run.py --name=T126_latentz_gtsub_len10_zex0pt3_subrat0pt5 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T126_latentz_gtsub_len10_zex0pt3_subrat0pt5 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3'

python cluster_run.py --name=T127_latentz_gtsub_len10_zex0_subrat1pt0 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T127_latentz_gtsub_len10_zex0_subrat1pt0 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.'

python cluster_run.py --name=T128_latentz_gtsub_len10_zex0pt3_subrat1pt0 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T128_latentz_gtsub_len10_zex0pt3_subrat1pt0 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3'

# Running just actions on new data.
python cluster_run.py --name=T129_len10_justa --cmd='python Master.py --train=1 --setting=just_actions --traj_length=10 --likelihood_penalty=5 --name=T129_len10_justa --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.'

python cluster_run.py --name=T130_lenfull_justa --cmd='python Master.py --train=1 --setting=just_actions --traj_length=-1 --likelihood_penalty=5 --name=T130_lenfull_justa --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.'

###########
# Running with old concatenated trajectory.

# Running with lowered subpolicy ratio for exploration bias = 0. and 0.3.
python cluster_run.py --name=T131_latentz_gtsub_len10_zex0_subrat0pt05 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T131_latentz_gtsub_len10_zex0_subrat0pt05 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.05 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.'

python cluster_run.py --name=T132_latentz_gtsub_len10_zex0_subrat0pt01 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T132_latentz_gtsub_len10_zex0_subrat0pt01 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.01 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.'

python cluster_run.py --name=T133_latentz_gtsub_len10_zex0_subrat0pt005 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T133_latentz_gtsub_len10_zex0_subrat0pt005 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.005 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.'

python cluster_run.py --name=T134_latentz_gtsub_len10_zex0_subrat0pt001 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T134_latentz_gtsub_len10_zex0_subrat0pt001 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.001 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.'

# With z ex bias = 0.3
python cluster_run.py --name=T135_latentz_gtsub_len10_zex0pt3_subrat0pt05 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T135_latentz_gtsub_len10_zex0pt3_subrat0pt05 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.05 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3'

python cluster_run.py --name=T136_latentz_gtsub_len10_zex0pt3_subrat0pt01 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T136_latentz_gtsub_len10_zex0pt3_subrat0pt01 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.01 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3'

python cluster_run.py --name=T137_latentz_gtsub_len10_zex0pt3_subrat0pt005 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T137_latentz_gtsub_len10_zex0pt3_subrat0pt005 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.005 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3'

python cluster_run.py --name=T138_latentz_gtsub_len10_zex0pt3_subrat0pt001 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T138_latentz_gtsub_len10_zex0pt3_subrat0pt001 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.001 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3'

python cluster_run.py --name=T139_latentz_gtsub_len10_zex0_subrat0pt5 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T139_latentz_gtsub_len10_zex0_subrat0pt5 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.'

python cluster_run.py --name=T140_latentz_gtsub_len10_zex0pt3_subrat0pt5 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T140_latentz_gtsub_len10_zex0pt3_subrat0pt5 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3'

python cluster_run.py --name=T141_latentz_gtsub_len10_zex0_subrat1pt0 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T141_latentz_gtsub_len10_zex0_subrat1pt0 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.'

python cluster_run.py --name=T142_latentz_gtsub_len10_zex0pt3_subrat1pt0 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T142_latentz_gtsub_len10_zex0pt3_subrat1pt0 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3'

# Running with old concatenated trajectory for variational network, and b exploration bias.
python cluster_run.py --name=T143_latentz_gtsub_len10_zex0pt3_bex0pt3_subrat0pt1 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T143_latentz_gtsub_len10_zex0pt3_bex0pt3_subrat0pt1 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3'

python cluster_run.py --name=T144_latentz_gtsub_len10_zex0pt3_bex0pt3_subrat0pt5 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T144_latentz_gtsub_len10_zex0pt3_bex0pt3_subrat0pt5 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3'

python cluster_run.py --name=T145_latentz_gtsub_len10_zex0pt3_bex0pt3_subrat0pt05 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T145_latentz_gtsub_len10_zex0pt3_bex0pt3_subrat0pt05 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.05 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3'

python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T146_latentz_gtsub_len10_zex0pt3_bex0pt3_subrat0pt1 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3

python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T147_latentz_gtsub_len10_zex0pt3_bex0pt3_subrat0pt5 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T148_latentz_gtsub_len10_zex0pt3_bex0pt3_subrat0pt05 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.05 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3
# Eval: 
CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T148_latentz_gtsub_len10_zex0pt3_bex0pt3_subrat0pt05_eval --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.05 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3 --model=Experiment_Logs/T148_latentz_gtsub_len10_zex0pt3_bex0pt3_subrat0pt05/saved_models/Model_epoch11

python cluster_run.py --name=T49_latentz_gtsub_len10_zex0pt3_bex0pt3_subrat0pt1 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T149_latentz_gtsub_len10_zex0pt3_bex0pt3_subrat0pt1 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=Continuous --z_ex_bias=0.3'

python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T150_latentz_gtsub_len10_zex0pt3_bex0pt3_subrat0pt1 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=Continuous --z_ex_bias=0.3

# Running with original latent policy input representation to test what happens.
python cluster_run.py --name=T151_latentz_gtsub_len10_ex0pt3_subrat0pt1_origrep --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T151_latentz_gtsub_len10_ex0pt3_subrat0pt1_origrep --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3'

python cluster_run.py --name=T152_latentz_gtsub_len10_ex0pt3_subrat0pt1_origrep --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T152_latentz_gtsub_len10_ex0pt3_subrat0pt1_origrep --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=Continuous --z_ex_bias=0.3'

python Master.py --train=1 --just_actions=0 --traj_length=10 --likelihood_penalty=5 --name=T84_latentz_gtsub_varent_len10_bexbias0pt3_llp5 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3

python Master.py --train=1 --just_actions=0 --traj_length=10 --likelihood_penalty=5 --name=T153_latentz_gtsub_varent_len10_bexbias0pt3_llp5 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.3

python cluster_run.py --name=T154_latentz_gtsub_len10_ex0pt3_subrat0pt1_origrep --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T154_latentz_gtsub_len10_ex0pt3_subrat0pt1_origrep --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3'
# Eval
python Master.py --train=0 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T154_latentz_gtsub_len10_ex0pt3_subrat0pt1_origrep_eval --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3 --model=Experiment_Logs/T154_latentz_gtsub_len10_ex0pt3_subrat0pt1_origrep/saved_models/Model_epoch12

python cluster_run.py --name=T155_latentz_gtsub_len10_ex0pt3_subrat0pt1_origrep --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T155_latentz_gtsub_len10_ex0pt3_subrat0pt1_origrep --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=Continuous --z_ex_bias=0.3'

# Epsilon 0.5 to 0.25 decay.
python cluster_run.py --name=T156_latentz_gtsub_len10_ex0pt3_subrat0pt1 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T156_latentz_gtsub_len10_ex0pt3_subrat0pt1 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3'

python cluster_run.py --name=T157_latentz_gtsub_len10_ex0pt3_subrat0pt1 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T157_latentz_gtsub_len10_ex0pt3_subrat0pt1 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=Continuous --z_ex_bias=0.3'

python Master.py --train=1 --just_actions=0 --traj_length=10 --likelihood_penalty=5 --name=T84_rerun2 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3


#####
# Trying to see if old concatenated trajectory representation with correct latent_z and latent_b concatenated.
python cluster_run.py --name=T158_latz_gtsub_len10_ex0pt3_subrat0pt1_corrlatz --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T158_latz_gtsub_len10_ex0pt3_subrat0pt1_corrlatz --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=Continuous --z_ex_bias=0.3'

python cluster_run.py --name=T159_latz_gtsub_len10_ex0pt3_subrat0pt1_corrlatz_contdir --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T159_latz_gtsub_len10_ex0pt3_subrat0pt1_corrlatz_contdir --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3'

# Correct input and latent_z latent_b representation.
python cluster_run.py --name=T160_latz_gtsub_len10_ex0pt3_subrat0pt1_corrip --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T160_latz_gtsub_len10_ex0pt3_subrat0pt1_corrip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=Continuous --z_ex_bias=0.3'

python cluster_run.py --name=T161_latz_gtsub_len10_ex0pt3_subrat0pt1_corrip_contdir --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T161_latz_gtsub_len10_ex0pt3_subrat0pt1_corrip_contdir --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3'

# Trying different epsilon greedy strategies with correct input representation. 
python cluster_run.py --name=T162_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt05ov10 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T162_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt05ov10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=Continuous --z_ex_bias=0.1 --data=Continuous --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10'
# Eval
python Master.py --train=0 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T162_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt05ov10_eval --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=Continuous --z_ex_bias=0.1 --data=Continuous --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --model=Experiment_Logs/T162_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt05ov10/saved_models/Model_epoch12

python cluster_run.py --name=T163_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt05ov20 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T163_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt05ov20 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=Continuous --z_ex_bias=0.1 --data=Continuous --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=20'

python cluster_run.py --name=T164_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt1ov20 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T164_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt1ov20 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=Continuous --z_ex_bias=0.1 --data=Continuous --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20'

python cluster_run.py --name=T165_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt2ov20 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T165_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt2ov20 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=Continuous --z_ex_bias=0.1 --data=Continuous --epsilon_from=0.5 --epsilon_to=0.2 --epsilon_over=20'

python cluster_run.py --name=T166_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt3ov20 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T166_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt3ov20 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=Continuous --z_ex_bias=0.1 --data=Continuous --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20'

python cluster_run.py --name=T167_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt4ov20 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T167_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt4ov20 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=Continuous --z_ex_bias=0.1 --data=Continuous --epsilon_from=0.5 --epsilon_to=0.4 --epsilon_over=20'

# Now running on continuous data with full length.
python cluster_run.py --name=T168_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt05ov10 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T168_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt05ov10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=Continuous --z_ex_bias=0.1 --data=Continuous --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10'
# Eval 
python Master.py --train=0 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T168_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt05ov10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=Continuous --z_ex_bias=0.1 --data=Continuous --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --model=Experiment_Logs/T168_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt05ov10/saved_models/Model_epoch21
# Eval 2
python Master.py --train=0 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T168_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt05ov10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=Continuous --z_ex_bias=0.1 --data=Continuous --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --model=Experiment_Logs/T168_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt05ov10/saved_models/Model_epoch35

python cluster_run.py --name=T169_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt05ov20 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T169_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt05ov20 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=Continuous --z_ex_bias=0.1 --data=Continuous --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=20'

python cluster_run.py --name=T170_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt1ov20 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T170_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt1ov20 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=Continuous --z_ex_bias=0.1 --data=Continuous --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20'

python cluster_run.py --name=T171_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt2ov20 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T171_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt2ov20 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=Continuous --z_ex_bias=0.1 --data=Continuous --epsilon_from=0.5 --epsilon_to=0.2 --epsilon_over=20'

python cluster_run.py --name=T172_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt3ov20 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T172_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt3ov20 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=Continuous --z_ex_bias=0.1 --data=Continuous --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20'

python cluster_run.py --name=T173_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt4ov20 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T173_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt4ov20 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=Continuous --z_ex_bias=0.1 --data=Continuous --epsilon_from=0.5 --epsilon_to=0.4 --epsilon_over=20'

# RERUN, because these runs had extra data=continuous flag. 
# Running on continuous directed data with length 10.
python cluster_run.py --name=T174_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt05ov10 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T174_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt05ov10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=ContinuousDir --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10'
python Master.py --train=0 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T174_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt05ov10_eval --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=ContinuousDir --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --model=Experiment_Logs/T174_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt05ov10/saved_models/Model_epoch24

python cluster_run.py --name=T175_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt05ov20 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T175_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt05ov20 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=ContinuousDir --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=20'
# Eval 
python Master.py --train=0 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T175_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt05ov20_eval --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=ContinuousDir --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=20 --model=Experiment_Logs/T175_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt05ov20/saved_models/Model_epoch20

python cluster_run.py --name=T176_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt1ov20 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T176_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt1ov20 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=ContinuousDir --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20'

python cluster_run.py --name=T177_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt2ov20 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T177_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt2ov20 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=ContinuousDir --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.2 --epsilon_over=20'

python cluster_run.py --name=T178_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt3ov20 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T178_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt3ov20 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=ContinuousDir --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20'

python cluster_run.py --name=T179_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt4ov20 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T179_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt4ov20 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=ContinuousDir --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.4 --epsilon_over=20'

# Running on continuous directed data with full length.
python cluster_run.py --name=T180_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt05ov10 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T180_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt05ov10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=ContinuousDir --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10'

python cluster_run.py --name=T181_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt05ov20 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T181_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt05ov20 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=ContinuousDir --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=20'

python cluster_run.py --name=T182_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt1ov20 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T182_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt1ov20 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=ContinuousDir --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20'

python cluster_run.py --name=T183_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt2ov20 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T183_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt2ov20 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=ContinuousDir --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.2 --epsilon_over=20'

python cluster_run.py --name=T184_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt3ov20 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T184_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt3ov20 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=ContinuousDir --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20'

python cluster_run.py --name=T185_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt4ov20 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T185_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt4ov20 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=ContinuousDir --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.4 --epsilon_over=20'

#####################
# Must run with correct assembled trajectory representation. 
# Across: 5, 10, full, Continuous and Directed demonstrations. 
python cluster_run.py --name=T200_latz_gtsub_len5_ex0pt3_corrip_eps0pt5to0pt05ov10 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=5 --likelihood_penalty=5 --name=T200_latz_gtsub_len5_ex0pt3_corrip_eps0pt5to0pt05ov10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=ContinuousDir --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10'

python cluster_run.py --name=T201_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt05ov10 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T201_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt05ov10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=ContinuousDir --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10'

python cluster_run.py --name=T202_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt05ov10 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T202_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt05ov10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=ContinuousDir --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10'
# Eval (though it probably doesn't work)
python Master.py --train=0 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T202_Eval --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=ContinuousDir --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --model=Experiment_Logs/T202_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt05ov10/saved_models/Model_epoch36

python cluster_run.py --name=T203_latz_gtsub_len5_ex0pt3_corrip_eps0pt5to0pt05ov10 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=5 --likelihood_penalty=5 --name=T203_latz_gtsub_len5_ex0pt3_corrip_eps0pt5to0pt05ov10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=Continuous --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10'

python cluster_run.py --name=T204_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt05ov10 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T204_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt05ov10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=Continuous --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10'

python cluster_run.py --name=T205_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt05ov10 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T205_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt05ov10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=Continuous --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10'

python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T206_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt05ov10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=ContinuousDir --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10

python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T207_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt05ov10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=Continuous --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10
# Eval
python Master.py --train=0 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T207_eval --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=Continuous --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --model=Experiment_Logs/T207_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt05ov10/saved_models/Model_epoch49

# 5 len and 10 len works. Now trying different epsilon strategies / exploration biases for full len. 
python cluster_run.py --name=T209_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt1ov10 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T209_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt1ov10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=Continuous --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=10'

python cluster_run.py --name=T210_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt3ov10 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T210_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt3ov10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=Continuous --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=10'

python cluster_run.py --name=T211_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt5ov10 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T211_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt5ov10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=Continuous --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.5 --epsilon_over=10'

python cluster_run.py --name=T212_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt1ov10 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T212_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt1ov10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=ContinuousDir --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=10'

python cluster_run.py --name=T213_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt3ov10 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T213_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt3ov10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=ContinuousDir --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=10'

python cluster_run.py --name=T214_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt5ov10 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=10 --likelihood_penalty=5 --name=T214_latz_gtsub_len10_ex0pt3_corrip_eps0pt5to0pt5ov10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=ContinuousDir --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.5 --epsilon_over=10'

# More trials with continuous directed data with full len. 
python cluster_run.py --name=T215_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt05ov10 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T215_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt05ov10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10'

python cluster_run.py --name=T216_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt1ov10 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T216_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt1ov10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=10'

python cluster_run.py --name=T217_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt1ov20 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T217_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt1ov20 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20'

python cluster_run.py --name=T218_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt3ov20 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T218_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt3ov20 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20'

python cluster_run.py --name=T219_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt5ov20 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T219_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt5ov20 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.5 --epsilon_over=20'

# Debug runs. 
python cluster_run.py --name=T220_latz_gtsub_lenfull_ex0pt1_corrip_eps0pt5to0pt05ov10 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T220_latz_gtsub_lenfull_ex0pt1_corrip_eps0pt5to0pt05ov10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=ContinuousDir --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10'

python cluster_run.py --name=T221_latz_gtsub_lenfull_ex0pt1_wrongip_eps0pt5to0pt05ov10 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T221_latz_gtsub_lenfull_ex0pt1_wrongip_eps0pt5to0pt05ov10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=ContinuousDir --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10'

# Runs with more exploration. 
python cluster_run.py --name=T222_latz_gtsub_lenfull_ex0pt1_corrip_eps0pt5to0pt05ov10 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T222_latz_gtsub_lenfull_ex0pt1_corrip_eps0pt5to0pt05ov10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=ContinuousDir --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10'

python cluster_run.py --name=T223_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt05ov10 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T223_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt05ov10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10'

python cluster_run.py --name=T224_latz_gtsub_lenfull_ex0pt5_corrip_eps0pt5to0pt05ov10 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T224_latz_gtsub_lenfull_ex0pt5_corrip_eps0pt5to0pt05ov10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=ContinuousDir --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10'
# Above was increasing ex bias. Now increase epsilon. 
python cluster_run.py --name=T225_latz_gtsub_lenfull_ex0pt1_corrip_eps0pt5to0pt3ov10 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T225_latz_gtsub_lenfull_ex0pt1_corrip_eps0pt5to0pt3ov10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=ContinuousDir --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=10'

python cluster_run.py --name=T226_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt3ov10 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T226_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt3ov10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=10'

python cluster_run.py --name=T227_latz_gtsub_lenfull_ex0pt5_corrip_eps0pt5to0pt3ov10 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T227_latz_gtsub_lenfull_ex0pt5_corrip_eps0pt5to0pt3ov10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=ContinuousDir --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=10'

# Running T220-223 with epsilon decay over 20 epochs. 
python cluster_run.py --name=T228_latz_gtsub_lenfull_ex0pt1_corrip_eps0pt5to0pt05ov20 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T228_latz_gtsub_lenfull_ex0pt1_corrip_eps0pt5to0pt05ov20 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=ContinuousDir --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=20'

python cluster_run.py --name=T229_latz_gtsub_lenfull_ex0pt1_wrongip_eps0pt5to0pt05ov20 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T229_latz_gtsub_lenfull_ex0pt1_wrongip_eps0pt5to0pt05ov20 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=ContinuousDir --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=20'

python cluster_run.py --name=T230_latz_gtsub_lenfull_ex0pt1_corrip_eps0pt5to0pt05ov20 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T230_latz_gtsub_lenfull_ex0pt1_corrip_eps0pt5to0pt05ov20 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=ContinuousDir --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=20'

python cluster_run.py --name=T231_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt05ov20 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T231_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt05ov20 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=20'


#####################
# Now trying to run with learnt discrete subpolicies.
# REMEMBER! WITHOUT GROUND TRUTH SUBPOLICIES, the subpolicy ratio can be set to 1. Because subpolicy loglikelihood is not of larger magnitude.  
# THis isn't strictly true, keep it at 0.1
python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T300_Learnsub_1 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=Continuous --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 

# Trying with pretrained subpolicies. 
# Run with loading S2. 
python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T301_Learnsub_pretrainS2 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=Continuous --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --subpolicy_model=Experiment_Logs/S2_PretrainTrial_len5/saved_models/Model_epoch49
# Load S3. 
python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T302_Learnsub_pretrainS3 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=Continuous --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --subpolicy_model=Experiment_Logs/S3_PretrainTrial_len5_b01/saved_models/Model_epoch47

#####################
# Run 1
python Master.py --train=1 --setting=pretrain_sub --name=S1_PretrainTrial --entropy=0 --data=Continuous

# Run 2
python Master.py --train=1 --setting=pretrain_sub --name=S2_PretrainTrial_len5 --entropy=0 --data=Continuous
# Eval:
python Master.py --train=0 --setting=pretrain_sub --name=S2_PretrainTrial_len5 --entropy=0 --data=Continuous --model=Experiment_Logs/S2_PretrainTrial_len5/saved_models/Model_epoch49

# Run with latent b's set to 00001
python Master.py --train=1 --setting=pretrain_sub --name=S3_PretrainTrial_len5_b01 --entropy=0 --data=Continuous
