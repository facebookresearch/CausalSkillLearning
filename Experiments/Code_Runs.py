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
python Master.py --train=0 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T220_eval --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=ContinuousDir --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --model=Experiment_Logs/T220_latz_gtsub_lenfull_ex0pt1_corrip_eps0pt5to0pt05ov10/saved_models/Model_epoch36

python cluster_run.py --name=T221_latz_gtsub_lenfull_ex0pt1_wrongip_eps0pt5to0pt05ov10 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T221_latz_gtsub_lenfull_ex0pt1_wrongip_eps0pt5to0pt05ov10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=ContinuousDir --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10'

# Runs with more exploration. 
python cluster_run.py --name=T222_latz_gtsub_lenfull_ex0pt1_corrip_eps0pt5to0pt05ov10 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T222_latz_gtsub_lenfull_ex0pt1_corrip_eps0pt5to0pt05ov10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=ContinuousDir --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10'

python cluster_run.py --name=T223_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt05ov10 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T223_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt05ov10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10'

python cluster_run.py --name=T224_latz_gtsub_lenfull_ex0pt5_corrip_eps0pt5to0pt05ov10 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T224_latz_gtsub_lenfull_ex0pt5_corrip_eps0pt5to0pt05ov10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=ContinuousDir --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10'
# Above was increasing ex bias. Now increase epsilon. 
python cluster_run.py --name=T225_latz_gtsub_lenfull_ex0pt1_corrip_eps0pt5to0pt3ov10 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T225_latz_gtsub_lenfull_ex0pt1_corrip_eps0pt5to0pt3ov10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=ContinuousDir --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=10'

python cluster_run.py --name=T226_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt3ov10 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T226_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt3ov10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=10'

python cluster_run.py --name=T227_latz_gtsub_lenfull_ex0pt5_corrip_eps0pt5to0pt3ov10 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T227_latz_gtsub_lenfull_ex0pt5_corrip_eps0pt5to0pt3ov10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=ContinuousDir --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=10'
# Eval
python Master.py --train=0 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T227_eval --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=ContinuousDir --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=10 --model=Experiment_Logs/T227_latz_gtsub_lenfull_ex0pt5_corrip_eps0pt5to0pt3ov10/saved_models/Model_epoch36

# Running T220-223 with epsilon decay over 20 epochs. 
python cluster_run.py --name=T228_latz_gtsub_lenfull_ex0pt1_corrip_eps0pt5to0pt05ov20 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T228_latz_gtsub_lenfull_ex0pt1_corrip_eps0pt5to0pt05ov20 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=ContinuousDir --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=20'

python cluster_run.py --name=T229_latz_gtsub_lenfull_ex0pt1_wrongip_eps0pt5to0pt05ov20 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T229_latz_gtsub_lenfull_ex0pt1_wrongip_eps0pt5to0pt05ov20 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=ContinuousDir --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=20'

python cluster_run.py --name=T230_latz_gtsub_lenfull_ex0pt1_corrip_eps0pt5to0pt05ov20 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T230_latz_gtsub_lenfull_ex0pt1_corrip_eps0pt5to0pt05ov20 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=ContinuousDir --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=20'

python cluster_run.py --name=T231_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt05ov20 --cmd='python Master.py --train=1 --setting=gtsub --traj_length=-1 --likelihood_penalty=5 --name=T231_latz_gtsub_lenfull_ex0pt3_corrip_eps0pt5to0pt05ov20 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=20'

###############################################################
##################### LEARNT SUBPOLICIIES #####################
###############################################################
# Now trying to run with learnt discrete subpolicies.
# REMEMBER! WITHOUT GROUND TRUTH SUBPOLICIES, the subpolicy ratio can be set to 1. Because subpolicy loglikelihood is not of larger magnitude.  
# THis isn't strictly true, keep it at 0.1
python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T300_Learnsub_1 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=Continuous --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 

# Trying with pretrained subpolicies. 
# Run with loading S2. 
python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T301_Learnsub_pretrainS2 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=Continuous --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --subpolicy_model=Experiment_Logs/S2_PretrainTrial_len5/saved_models/Model_epoch49
# Load S3. 
python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T302_Learnsub_pretrainS3 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=Continuous --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --subpolicy_model=Experiment_Logs/S3_PretrainTrial_len5_b01/saved_models/Model_epoch47

# Most of the pretrained subpolicies were trash until now. 
# S11 is the best model so far. Running trial of learnt discrete subpolicies with S11. 
python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T303_Learnsub_pretrainS11 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=Continuous --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch24

# Run with fixed subpolicies. 
python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T304_Learnsub_pretrainS11_fixsubpol --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=Continuous --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1

# Running T303 and T304 with increased exploration bias. 
python cluster_run.py --name=T305_Learnsub_pretrainS11_ex0pt3 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T305_Learnsub_pretrainS11_ex0pt3 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=Continuous --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49'

python cluster_run.py --name=T306_Learnsub_pretrainS11_fixsubpol_ex0pt3 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T306_Learnsub_pretrainS11_fixsubpol_ex0pt3 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=Continuous --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1'

# Run T303 and 304 as 307 and 308 with correct logging of subpolicy likelihood. 
python cluster_run.py --name=T307_Learnsub_pretrainS11_ex0pt3 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T307_Learnsub_pretrainS11_ex0pt3 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=Continuous --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49'

python cluster_run.py --name=T308_Learnsub_pretrainS11_fixsubpol_ex0pt3 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T308_Learnsub_pretrainS11_fixsubpol_ex0pt3 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=Continuous --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1'

python cluster_run.py --name=T308_Learnsub_pretrainS11_fixsubpol_ex0pt3 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T308_Learnsub_pretrainS11_fixsubpol_ex0pt3 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=Continuous --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1'
# Eval
python Master.py --train=0 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T308_Eval --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=Continuous --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --fix_subpolicy=1 --model=Experiment_Logs/T308_Learnsub_pretrainS11_fixsubpol_ex0pt3/saved_models/Model_epoch19

# Rerunning T307 and T308 with right summation of subpolicy likelihood. 
python cluster_run.py --name=T309_Learnsub_pretrainS11_ex0pt3 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T309_Learnsub_pretrainS11_ex0pt3 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=Continuous --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49'

python cluster_run.py --name=T310_Learnsub_pretrainS11_fixsubpol_ex0pt3 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T310_Learnsub_pretrainS11_fixsubpol_ex0pt3 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=Continuous --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1'

python cluster_run.py --name=T311_Learnsub_pretrainS11_fixsubpol_ex0pt3 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T311_Learnsub_pretrainS11_fixsubpol_ex0pt3 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=Continuous --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1'
# Eval
python Master.py --train=0 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T311_eval --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=Continuous --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --model=Experiment_Logs/T311_Learnsub_pretrainS11_fixsubpol_ex0pt3/saved_models/Model_epoch5 --fix_subpolicy=1 
# Eval at 13 epochs.
python Master.py --train=0 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T311_eval --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=Continuous --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --model=Experiment_Logs/T311_Learnsub_pretrainS11_fixsubpol_ex0pt3/saved_models/Model_epoch13 --fix_subpolicy=1 
# Eval at 19 epochs.
python Master.py --train=0 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T311_eval --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=Continuous --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --model=Experiment_Logs/T311_Learnsub_pretrainS11_fixsubpol_ex0pt3/saved_models/Model_epoch19 --fix_subpolicy=1 


# Random debug
python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=TLearnsub_Trial --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=Continuous --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1

# Now that we see it able to learn a variational network in the learnt subpolicy case, start switching to a longer trajectory length. 
python cluster_run.py --name=T312_Learnsub_pretrainS11_fixsubpol_ex0pt3_len10 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --likelihood_penalty=5 --name=T312_Learnsub_pretrainS11_fixsubpol_ex0pt3_len10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=Continuous --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1'

# Run T311 with image visualizations.
python cluster_run.py --name=T313_Learnsub_pretrainS11_fixsubpol_ex0pt3 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T313_Learnsub_pretrainS11_fixsubpol_ex0pt3 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=Continuous --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1'

# T313 seemed to collapse to a single mode of the variational network. So maybe increase the exploration bias to 0.5 for z's.
python cluster_run.py --name=T314_Learnsub_pretrainS11_fixsubpol_ex0pt5 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T314_Learnsub_pretrainS11_fixsubpol_ex0pt5 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1'

# Running 314 locally to debug OOM.
python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T315_Learnsub_pretrainS11_fixsubpol_ex0pt5 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1

# Exploration trials to see if latent and variational networks still collapse initially. 
python cluster_run.py --name=T316_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt05ov10 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T316_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt05ov10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1'
# Eval at 11
python Master.py --train=0 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T316_Eval --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --model=Experiment_Logs/T316_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt05ov10/saved_models/Model_epoch11
# Eval at 18
python Master.py --train=0 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T316_Eval --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --model=Experiment_Logs/T316_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt05ov10/saved_models/Model_epoch18

python cluster_run.py --name=T317_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt1ov10 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T317_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt1ov10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=10 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1'

python cluster_run.py --name=T318_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt1ov20 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T318_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt1ov20 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1'

python cluster_run.py --name=T319_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt3ov20 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T319_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt3ov20 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1'

python cluster_run.py --name=T320_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt5ov20 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T320_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt5ov20 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.5 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1'

# Running with entropy reguarlization on the latent policy, just so that we see if it doesn't collapse.
python cluster_run.py --name=T321_Learnsub_pretrainS11_fixsubpol_ex0pt5_entreg --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T321_Learnsub_pretrainS11_fixsubpol_ex0pt5_entreg --ent_weight=0.1 --entropy=1 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1'
# Eval at 18
python Master.py --train=0 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T321_Eval --ent_weight=0.1 --entropy=1 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --model=Experiment_Logs/T321_Learnsub_pretrainS11_fixsubpol_ex0pt5_entreg/saved_models/Model_epoch18

# Exploration trials with length = 10 to see if exploration can solve it. 
python cluster_run.py --name=T322_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt05ov10_len10 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --likelihood_penalty=5 --name=T322_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt05ov10_len10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1'

python cluster_run.py --name=T323_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt1ov10_len10 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --likelihood_penalty=5 --name=T323_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt1ov10_len10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=10 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1'

python cluster_run.py --name=T324_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt1ov20_len10 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --likelihood_penalty=5 --name=T324_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt1ov20_len10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1'

python cluster_run.py --name=T325_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt3ov20_len10 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --likelihood_penalty=5 --name=T325_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt3ov20_len10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1'

python cluster_run.py --name=T326_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt5ov20_len10 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --likelihood_penalty=5 --name=T326_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt5ov20_len10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.5 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1'

# Profiling debugging:
python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=TProfile --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1
# With faster disp freq
python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=TProfile --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1 --display_freq=4

# Train with entropy regularization wiht higher entropy weight.
python cluster_run.py --name=T327_Learnsub_pretrainS11_fixsubpol_ex0pt5_entreg --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T327_Learnsub_pretrainS11_fixsubpol_ex0pt5_entreg --ent_weight=1. --entropy=1 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1'

# Train with new z selection method that reruns T316-320.
python cluster_run.py --name=T328_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt05ov10 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T328_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt05ov10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1 --new_z_selection=1'
#Eval
python Master.py --train=0 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T328_Eval --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --fix_subpolicy=1 --new_z_selection=1 --model=Experiment_Logs/T328_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt05ov10/saved_models/Model_epoch19

python cluster_run.py --name=T329_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt1ov10 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T329_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt1ov10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=10 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T330_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt1ov20 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T330_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt1ov20 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1 --new_z_selection=1'
# Eval at model 11.
python Master.py --train=0 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T330_Eval --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --model=Experiment_Logs/T330_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt1ov20/saved_models/Model_epoch11 

python cluster_run.py --name=T331_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt3ov20 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T331_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt3ov20 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T332_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt5ov20 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T332_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt5ov20 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.5 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1'

# Running T316-320 with lowered learning rate for the latent policy. 
python cluster_run.py --name=T333_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt05ov10_lowerlatentLR --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T333_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt05ov10_lowerlatentLR --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1'

python cluster_run.py --name=T334_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt1ov10 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T334_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt1ov10_lowerlatentLR --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=10 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1'

python cluster_run.py --name=T335_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt1ov20 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T335_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt1ov20_lowerlatentLR --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1'

python cluster_run.py --name=T336_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt3ov20 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T336_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt3ov20_lowerlatentLR --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1'

python cluster_run.py --name=T337_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt5ov20 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T337_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt5ov20_lowerlatentLR --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.5 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1'

# Run with b exploration bias only being done for the continuation, rather than both? 
python cluster_run.py --name=T338_Learnsub_pretrainS11_fixsubpol_contex0pt5 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T338_Learnsub_pretrainS11_fixsubpol_contex0pt5 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T339_Learnsub_pretrainS11_fixsubpol_contex0pt5 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T339_Learnsub_pretrainS11_fixsubpol_contex0pt5_epsto0pt3 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T340_Learnsub_pretrainS11_fixsubpol_contex0pt5 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T340_Learnsub_pretrainS11_fixsubpol_contex0pt5_epsto0pt5 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.5 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1 --new_z_selection=1'

# Debugging new z selection
python Master.py --train=1 --setting=learntsub --traj_length=10 --likelihood_penalty=5 --name=TDebug_Zsel --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.5 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1 --new_z_selection=1

# Running with new z selection and new diff_val computation.
python cluster_run.py --name=T341_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt05ov10 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T341_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt05ov10_newztryb --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=10 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T342_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt1ov20 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T342_Learnsub_pretrainS11_fixsubpol_ex0pt5_eps0pt5to0pt1ov20_newztryb --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1 --new_z_selection=1'

# The T338-342 runs were pretty much being implemented with the b exploration bias being added to the first timestep, not all timesteps. This is corrected below: 
python cluster_run.py --name=T343_Learnsub_pretrainS11_fixsubpol_contex0pt5 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T343_Learnsub_pretrainS11_fixsubpol_contex0pt5 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T344_Learnsub_pretrainS11_fixsubpol_contex0pt5 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T344_Learnsub_pretrainS11_fixsubpol_contex0pt5_epsto0pt3 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T345_Learnsub_pretrainS11_fixsubpol_contex0pt5 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T345_Learnsub_pretrainS11_fixsubpol_contex0pt5_epsto0pt5 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.5 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1 --new_z_selection=1'

# Trying with different bex bias values.
python cluster_run.py --name=T346_Learnsub_pretrainS11_fixsubpol_contex1 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T346_Learnsub_pretrainS11_fixsubpol_contex1 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=1. --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T347_Learnsub_pretrainS11_fixsubpol_contex1 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T347_Learnsub_pretrainS11_fixsubpol_contex1_epsto0pt3 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=1. --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T348_Learnsub_pretrainS11_fixsubpol_contex1 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T348_Learnsub_pretrainS11_fixsubpol_contex1_epsto0pt5 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=1. --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.5 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T349_Learnsub_pretrainS11_fixsubpol_contex2 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T349_Learnsub_pretrainS11_fixsubpol_contex2 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=2. --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T350_Learnsub_pretrainS11_fixsubpol_contex2 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T350_Learnsub_pretrainS11_fixsubpol_contex2_epsto0pt3 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=2. --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T351_Learnsub_pretrainS11_fixsubpol_contex2 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T351_Learnsub_pretrainS11_fixsubpol_contex2_epsto0pt5 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=2. --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.5 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1 --new_z_selection=1'

# Rerunning T343-T345 with subpolicy ratio increased to 1. Also will try with ratio = 0.5
python cluster_run.py --name=T352_Learnsub_pretrainS11_fixsubpol_contex0pt5_subrat0pt5 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T352_Learnsub_pretrainS11_fixsubpol_contex0pt5_subrat0pt5 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1 --new_z_selection=1'
# Eval at 14.
python Master.py --train=0 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T352_Eval --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --model=Experiment_Logs/T352_Learnsub_pretrainS11_fixsubpol_contex0pt5_subrat0pt5/saved_models/Model_epoch14
# Eval at 19.
python Master.py --train=0 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T352_Eval --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --model=Experiment_Logs/T352_Learnsub_pretrainS11_fixsubpol_contex0pt5_subrat0pt5/saved_models/Model_epoch19

python cluster_run.py --name=T353_Learnsub_pretrainS11_fixsubpol_contex0pt5_subrat0pt5 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T353_Learnsub_pretrainS11_fixsubpol_contex0pt5_subrat0pt5 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T354_Learnsub_pretrainS11_fixsubpol_contex0pt5_subrat0pt5 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T354_Learnsub_pretrainS11_fixsubpol_contex0pt5_subrat0pt5 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.5 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T355_Learnsub_pretrainS11_fixsubpol_contex0pt5_subrat1pt0 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T355_Learnsub_pretrainS11_fixsubpol_contex0pt5_subrat1pt0 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T356_Learnsub_pretrainS11_fixsubpol_contex0pt5_subrat1pt0 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T356_Learnsub_pretrainS11_fixsubpol_contex0pt5_subrat1pt0 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T357_Learnsub_pretrainS11_fixsubpol_contex0pt5_subrat1pt0 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=T357_Learnsub_pretrainS11_fixsubpol_contex0pt5_subrat1pt0 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.5 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1 --new_z_selection=1'

# Debug b.
python cluster_run.py --name=TDebug_b --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=TDebug_b --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1 --new_z_selection=1'

python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=TDebug_b2 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1 --new_z_selection=1

python cluster_run.py --name=TDebug_b_subrat0pt5 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=TDebug_b_subrat0pt5 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=TDebug_b_subrat1pt0 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --likelihood_penalty=5 --name=TDebug_b_subrat1pt0 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1.0 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1 --new_z_selection=1'

# Running with old pretrained model S11 with length 10.
python cluster_run.py --name=T358_Learnsub_pretrainS11_fixsubpol_contex0pt5_subrat0pt5 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --likelihood_penalty=5 --name=T358_Learnsub_pretrainS11_fixsubpol_contex0pt5_subrat0pt5_len10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T359_Learnsub_pretrainS11_fixsubpol_contex0pt5_subrat0pt5 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --likelihood_penalty=5 --name=T359_Learnsub_pretrainS11_fixsubpol_contex0pt5_subrat0pt5_len10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.5 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T360_Learnsub_pretrainS11_fixsubpol_contex0pt5_subrat1pt0 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --likelihood_penalty=5 --name=T360_Learnsub_pretrainS11_fixsubpol_contex0pt5_subrat1pt0_len10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T361_Learnsub_pretrainS11_fixsubpol_contex0pt5_subrat1pt0 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --likelihood_penalty=5 --name=T361_Learnsub_pretrainS11_fixsubpol_contex0pt5_subrat1pt0_len10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T362_Learnsub_pretrainS11_fixsubpol_contex0pt5_subrat1pt0 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --likelihood_penalty=5 --name=T362_Learnsub_pretrainS11_fixsubpol_contex0pt5_subrat1pt0_len10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.5 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49 --fix_subpolicy=1 --new_z_selection=1'

# Running with S32 instead of S11 as the pretrained model. Trying to see if we can replicate T352 success. 
python cluster_run.py --name=T363_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --name=T363_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T364_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --name=T364_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T365_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --name=T365_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.5 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T366_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat1pt0_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --name=T366_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat1pt0_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T367_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat1pt0_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --name=T367_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat1pt0_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T368_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat1pt0_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --name=T368_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat1pt0_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.5 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1'

# Running T363-T368 with len 10 instead of len 5. 
python cluster_run.py --name=T369_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip_len10 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T369_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip_len10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T370_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip_len10 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T370_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip_len10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T371_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip_len10 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T371_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip_len10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.5 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T372_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat1pt0_nolatbip_len10 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T372_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat1pt0_nolatbip_len10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T373_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat1pt0_nolatbip_len10 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T373_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat1pt0_nolatbip_len10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T374_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat1pt0_nolatbip_len10 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T374_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat1pt0_nolatbip_len10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.5 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1'

# The 363-374 trials were mistakenly run with latent_b's set to 10000.
# Re-running with latent_b's being predicted. 
python cluster_run.py --name=T375_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --name=T375_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T376_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --name=T376_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T377_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --name=T377_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.5 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T378_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat1pt0_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --name=T378_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat1pt0_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T379_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat1pt0_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --name=T379_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat1pt0_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T380_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat1pt0_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --name=T380_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat1pt0_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.5 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1'

# Run with length 10 instead of length 5.
python cluster_run.py --name=T381_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip_len10 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T381_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip_len10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1'

# Debug run:
python Master.py --train=1 --setting=learntsub --traj_length=10 --name=TDebug_jointtraining --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.3 --data=Continuous --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1

python cluster_run.py --name=T382_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip_len10 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T382_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip_len10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T383_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip_len10 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T383_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip_len10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.5 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T384_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat1pt0_nolatbip_len10 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T384_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat1pt0_nolatbip_len10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T385_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat1pt0_nolatbip_len10 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T385_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat1pt0_nolatbip_len10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T386_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat1pt0_nolatbip_len10 --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T386_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat1pt0_nolatbip_len10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.5 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1'

# Running with multiplying the preprobabiltiies with 0.1 and then adding bias. Also trying with bias = 0.3, since 0.5 seems like a bit much. 
python cluster_run.py --name=T387_learnsub_pretrainS32_fixsubpol_contex0pt3_subrat0pt5_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --name=T387_learnsub_pretrainS32_fixsubpol_contex0pt3_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.3 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T388_learnsub_pretrainS32_fixsubpol_contex0pt4_subrat0pt5_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --name=T388_learnsub_pretrainS32_fixsubpol_contex0pt4_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.4 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T389_learnsub_pretrainS32_fixsubpol_contex0pt4_subrat0pt5_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --name=T389_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1'

# Running with multiplying both z and b preprobabiltiies with 0.1 and then adding bias. Also trying with bias = 0.3, since 0.5 seems like a bit much. 
python cluster_run.py --name=T390_learnsub_pretrainS32_fixsubpol_contex0pt3_subrat0pt5_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --name=T390_learnsub_pretrainS32_fixsubpol_contex0pt3_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.3 --data=Continuous --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T391_learnsub_pretrainS32_fixsubpol_contex0pt4_subrat0pt5_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --name=T391_learnsub_pretrainS32_fixsubpol_contex0pt4_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.4 --data=Continuous --z_ex_bias=0.4 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T392_learnsub_pretrainS32_fixsubpol_contex0pt4_subrat0pt5_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --name=T392_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T393_learnsub_pretrainS32_fixsubpol_contex0pt3_subrat1pt0_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --name=T393_learnsub_pretrainS32_fixsubpol_contex0pt3_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.3 --data=Continuous --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1'
# Eval at 2
python Master.py --train=0 --setting=learntsub --traj_length=5 --name=T393_learnsub_pretrainS32_fixsubpol_contex0pt3_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.3 --data=Continuous --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --model=Experiment_Logs/T393_learnsub_pretrainS32_fixsubpol_contex0pt3_subrat0pt5_nolatbip/saved_models/Model_epoch2

python cluster_run.py --name=T394_learnsub_pretrainS32_fixsubpol_contex0pt4_subrat1pt0_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --name=T394_learnsub_pretrainS32_fixsubpol_contex0pt4_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.4 --data=Continuous --z_ex_bias=0.4 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T395_learnsub_pretrainS32_fixsubpol_contex0pt4_subrat1pt0_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --name=T395_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1'

# Run T390-T395 with len 10, since it looks like this explores to a much better extent.
python cluster_run.py --name=T396_learnsub_pretrainS32_fixsubpol_contex0pt3_subrat0pt5_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T396_learnsub_pretrainS32_fixsubpol_contex0pt3_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.3 --data=Continuous --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T397_learnsub_pretrainS32_fixsubpol_contex0pt4_subrat0pt5_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T397_learnsub_pretrainS32_fixsubpol_contex0pt4_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.4 --data=Continuous --z_ex_bias=0.4 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T398_learnsub_pretrainS32_fixsubpol_contex0pt4_subrat0pt5_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T398_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T399_learnsub_pretrainS32_fixsubpol_contex0pt3_subrat1pt0_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T399_learnsub_pretrainS32_fixsubpol_contex0pt3_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.3 --data=Continuous --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T400_learnsub_pretrainS32_fixsubpol_contex0pt4_subrat1pt0_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T400_learnsub_pretrainS32_fixsubpol_contex0pt4_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.4 --data=Continuous --z_ex_bias=0.4 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1'

python cluster_run.py --name=T401_learnsub_pretrainS32_fixsubpol_contex0pt4_subrat1pt0_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T401_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1'

# Reducing probability factor to 0.01
# Running with multiplying both z and b preprobabiltiies with 0.1 and then adding bias. Also trying with bias = 0.3, since 0.5 seems like a bit much. 
python cluster_run.py --name=T402_learnsub_pretrainS32_fixsubpol_contex0pt3_subrat0pt5_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --name=T402_learnsub_pretrainS32_fixsubpol_contex0pt3_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.3 --data=Continuous --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1 --z_probability_factor=0.05 --b_probability_factor=0.01'

python cluster_run.py --name=T403_learnsub_pretrainS32_fixsubpol_contex0pt4_subrat0pt5_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --name=T403_learnsub_pretrainS32_fixsubpol_contex0pt4_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.4 --data=Continuous --z_ex_bias=0.4 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1 --z_probability_factor=0.05 --b_probability_factor=0.01'

python cluster_run.py --name=T404_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --name=T404_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1 --z_probability_factor=0.05 --b_probability_factor=0.01'

python cluster_run.py --name=T405_learnsub_pretrainS32_fixsubpol_contex0pt3_subrat1pt0_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --name=T405_learnsub_pretrainS32_fixsubpol_contex0pt3_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.3 --data=Continuous --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1 --z_probability_factor=0.05 --b_probability_factor=0.01'

python cluster_run.py --name=T406_learnsub_pretrainS32_fixsubpol_contex0pt4_subrat1pt0_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --name=T406_learnsub_pretrainS32_fixsubpol_contex0pt4_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.4 --data=Continuous --z_ex_bias=0.4 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1 --z_probability_factor=0.05 --b_probability_factor=0.01'

python cluster_run.py --name=T407_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat1pt0_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=5 --name=T407_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1 --z_probability_factor=0.05 --b_probability_factor=0.01'
# Eval at 11.
python Master.py --train=0 --setting=learntsub --traj_length=5 --name=T407_Eval --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --z_probability_factor=0.05 --b_probability_factor=0.01 --model=Experiment_Logs/T407_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip/saved_models/Model_epoch11

# Running T402-T407 with len 10 instead of 5. 
# Reducing probability factor to 0.01
# Running with multiplying both z and b preprobabiltiies with 0.1 and then adding bias. Also trying with bias = 0.3, since 0.5 seems like a bit much. 
python cluster_run.py --name=T408_learnsub_pretrainS32_fixsubpol_contex0pt3_subrat0pt5_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T408_learnsub_pretrainS32_fixsubpol_contex0pt3_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.3 --data=Continuous --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1 --z_probability_factor=0.05 --b_probability_factor=0.01'

python cluster_run.py --name=T409_learnsub_pretrainS32_fixsubpol_contex0pt4_subrat0pt5_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T409_learnsub_pretrainS32_fixsubpol_contex0pt4_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.4 --data=Continuous --z_ex_bias=0.4 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1 --z_probability_factor=0.05 --b_probability_factor=0.01'

python cluster_run.py --name=T410_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T410_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1 --z_probability_factor=0.05 --b_probability_factor=0.01'
# Eval at 10 to debug.
python Master.py --train=0 --setting=learntsub --traj_length=10 --name=T410_Eval --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --z_probability_factor=0.05 --b_probability_factor=0.01 --model=Experiment_Logs/T410_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip/saved_models/Model_epoch10
# Train debug. 
python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T410_Debug --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1 --z_probability_factor=0.05 --b_probability_factor=0.01
#
python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T410_Debug --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1 --z_probability_factor=0.01 --b_probability_factor=0.01

python cluster_run.py --name=T411_learnsub_pretrainS32_fixsubpol_contex0pt3_subrat1pt0_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T411_learnsub_pretrainS32_fixsubpol_contex0pt3_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.3 --data=Continuous --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1 --z_probability_factor=0.05 --b_probability_factor=0.01'

python cluster_run.py --name=T412_learnsub_pretrainS32_fixsubpol_contex0pt4_subrat1pt0_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T412_learnsub_pretrainS32_fixsubpol_contex0pt4_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.4 --data=Continuous --z_ex_bias=0.4 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1 --z_probability_factor=0.05 --b_probability_factor=0.01'

python cluster_run.py --name=T413_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat1pt0_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T413_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1 --z_probability_factor=0.05 --b_probability_factor=0.01'

# Rerun T408-T413 with z_prob_factor reduced to 0.01., and epsilon _to set to 0.5
python cluster_run.py --name=T414_learnsub_pretrainS32_fixsubpol_contex0pt3_subrat0pt5_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T414_learnsub_pretrainS32_fixsubpol_contex0pt3_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.3 --data=Continuous --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1 --z_probability_factor=0.01 --b_probability_factor=0.01'
# Eval at 10 to debug.
python Master.py --train=0 --setting=learntsub --traj_length=10 --name=T414_learnsub_pretrainS32_fixsubpol_contex0pt3_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.3 --data=Continuous --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --z_probability_factor=0.01 --b_probability_factor=0.01 --model=Experiment_Logs/T414_learnsub_pretrainS32_fixsubpol_contex0pt3_subrat0pt5_nolatbip/saved_models/Model_epoch10

python cluster_run.py --name=T415_learnsub_pretrainS32_fixsubpol_contex0pt4_subrat0pt5_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T415_learnsub_pretrainS32_fixsubpol_contex0pt4_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.4 --data=Continuous --z_ex_bias=0.4 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1 --z_probability_factor=0.01 --b_probability_factor=0.01'

python cluster_run.py --name=T416_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T416_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1 --z_probability_factor=0.01 --b_probability_factor=0.01'

python cluster_run.py --name=T417_learnsub_pretrainS32_fixsubpol_contex0pt3_subrat1pt0_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T417_learnsub_pretrainS32_fixsubpol_contex0pt3_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.3 --data=Continuous --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1 --z_probability_factor=0.01 --b_probability_factor=0.01'

python cluster_run.py --name=T418_learnsub_pretrainS32_fixsubpol_contex0pt4_subrat1pt0_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T418_learnsub_pretrainS32_fixsubpol_contex0pt4_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.4 --data=Continuous --z_ex_bias=0.4 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1 --z_probability_factor=0.01 --b_probability_factor=0.01'

python cluster_run.py --name=T419_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat1pt0_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T419_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1 --z_probability_factor=0.01 --b_probability_factor=0.01'

# Running with higher epsilon and higher z_ex_bias. This just slows down how quickly variational net commits to z's. 
python cluster_run.py --name=T420_learnsub_pretrainS32_fixsubpol_contex0pt3_subrat0pt5_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T420_learnsub_pretrainS32_fixsubpol_contex0pt3_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.3 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.7 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1 --z_probability_factor=0.01 --b_probability_factor=0.01'

python cluster_run.py --name=T421_learnsub_pretrainS32_fixsubpol_contex0pt4_subrat0pt5_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T421_learnsub_pretrainS32_fixsubpol_contex0pt4_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.4 --data=Continuous --z_ex_bias=1. --epsilon_from=0.7 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1 --z_probability_factor=0.01 --b_probability_factor=0.01'

python cluster_run.py --name=T422_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T422_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=2. --epsilon_from=0.7 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1 --z_probability_factor=0.01 --b_probability_factor=0.01'

python cluster_run.py --name=T423_learnsub_pretrainS32_fixsubpol_contex0pt3_subrat1pt0_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T423_learnsub_pretrainS32_fixsubpol_contex0pt3_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.3 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.7 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1 --z_probability_factor=0.01 --b_probability_factor=0.01'

python cluster_run.py --name=T424_learnsub_pretrainS32_fixsubpol_contex0pt4_subrat1pt0_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T424_learnsub_pretrainS32_fixsubpol_contex0pt4_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.4 --data=Continuous --z_ex_bias=1. --epsilon_from=0.7 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1 --z_probability_factor=0.01 --b_probability_factor=0.01'

python cluster_run.py --name=T425_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat1pt0_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T425_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.5 --data=Continuous --z_ex_bias=2. --epsilon_from=0.7 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1 --z_probability_factor=0.01 --b_probability_factor=0.01'

# Running with fake batch size = 3 or 5 or something. 
python cluster_run.py --name=T426_learnsub_pretrainS32_fixsubpol_contex0pt3_subrat0pt5_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T426_learnsub_pretrainS32_fixsubpol_contex0pt3_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.3 --data=Continuous --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1 --z_probability_factor=0.01 --b_probability_factor=0.01 --fake_batch_size=5'

python cluster_run.py --name=T427_learnsub_pretrainS32_fixsubpol_contex0pt4_subrat0pt5_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T427_learnsub_pretrainS32_fixsubpol_contex0pt4_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.4 --data=Continuous --z_ex_bias=0.4 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1 --z_probability_factor=0.01 --b_probability_factor=0.01 --fake_batch_size=5'

python cluster_run.py --name=T428_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T428_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.5 --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1 --z_probability_factor=0.01 --b_probability_factor=0.01 --fake_batch_size=5'

python cluster_run.py --name=T429_learnsub_pretrainS32_fixsubpol_contex0pt3_subrat1pt0_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T429_learnsub_pretrainS32_fixsubpol_contex0pt3_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.3 --data=Continuous --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1 --z_probability_factor=0.01 --b_probability_factor=0.01 --fake_batch_size=5'

python cluster_run.py --name=T430_learnsub_pretrainS32_fixsubpol_contex0pt4_subrat1pt0_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T430_learnsub_pretrainS32_fixsubpol_contex0pt4_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.4 --data=Continuous --z_ex_bias=0.4 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1 --z_probability_factor=0.01 --b_probability_factor=0.01 --fake_batch_size=5'

python cluster_run.py --name=T431_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat1pt0_nolatbip --cmd='python Master.py --train=1 --setting=learntsub --traj_length=10 --name=T431_learnsub_pretrainS32_fixsubpol_contex0pt5_subrat0pt5_nolatbip --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=1. --b_ex_bias=0.5 --data=Continuous --z_ex_bias=0.5 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --subpolicy_model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29 --fix_subpolicy=1 --new_z_selection=1 --z_probability_factor=0.01 --b_probability_factor=0.01 --fake_batch_size=5'



###############################################################
################ PRETRAINING SUBPOLICY TRIALS. ################
###############################################################
# Run 1
python Master.py --train=1 --setting=pretrain_sub --name=S1_PretrainTrial --entropy=0 --data=Continuous

# Run 2
python Master.py --train=1 --setting=pretrain_sub --name=S2_PretrainTrial_len5 --entropy=0 --data=Continuous
# Eval:
python Master.py --train=0 --setting=pretrain_sub --name=S2_PretrainTrial_len5 --entropy=0 --data=Continuous --model=Experiment_Logs/S2_PretrainTrial_len5/saved_models/Model_epoch49

# Run with latent b's set to 00001
python Master.py --train=1 --setting=pretrain_sub --name=S3_PretrainTrial_len5_b01 --entropy=0 --data=Continuous

# Run with entropy / KL terms of encoder added. 
python Master.py --train=1 --setting=pretrain_sub --name=S4_Pretrain_len5_KLD --entropy=0 --data=Continuous

# Potentially debugging losses. 
python Master.py --train=1 --setting=pretrain_sub --name=S5_Debug --entropy=0 --data=Continuous 

# Summing losses and actually stepping with subpolicy opt. 
python Master.py --train=1 --setting=pretrain_sub --name=S6_Pretrain_len5_KLD_sumloss --entropy=0 --data=Continuous 

# Fix reinforce / KL for encoder and summing KL. 
python Master.py --train=1 --setting=pretrain_sub --name=S7_Pretrain_len5_rightlossv1 --entropy=0 --data=Continuous
#
python Master.py --train=1 --setting=pretrain_sub --name=S8_Pretrain_len5_rightlossv1 --entropy=0 --data=Continuous

# Without retain graph.
python Master.py --train=1 --setting=pretrain_sub --name=S9_Pretrain_len5_rightlossv1 --entropy=0 --data=Continuous
# Eval 
python Master.py --train=0 --setting=pretrain_sub --name=S9_Pretrain_len5_rightlossv1 --entropy=0 --data=Continuous --model=Experiment_Logs/S9_Pretrain_len5_rightlossv1/saved_models/Model_epoch7
# Eval 2 
python Master.py --train=0 --setting=pretrain_sub --name=S9_Pretrain_len5_rightlossv1 --entropy=0 --data=Continuous --model=Experiment_Logs/S9_Pretrain_len5_rightlossv1/saved_models/Model_epoch19


# Running Pretraining with full logging. 
python Master.py --train=1 --setting=pretrain_sub --name=S10_Pretrain_len5_rightlossv1_logging --entropy=0 --data=Continuous
# Eval 
python Master.py --train=0 --setting=pretrain_sub --name=S10_eval --entropy=0 --data=Continuous --model=Experiment_Logs/S10_Pretrain_len5_rightlossv1_logging/saved_models/Model_epoch16

# Decreasing KL weight. 
python Master.py --train=1 --setting=pretrain_sub --name=S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1 --entropy=0 --data=Continuous --kl_weight=0.1
# Eval
python Master.py --train=0 --setting=pretrain_sub --name=S11_eval --entropy=0 --data=Continuous --kl_weight=0.1 --model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch24
# Eval at 49
python Master.py --train=0 --setting=pretrain_sub --name=S11_eval --entropy=0 --data=Continuous --kl_weight=0.1 --model=Experiment_Logs/S11_Pretrain_len5_rightlossv1_logging_kldwt0pt1/saved_models/Model_epoch49

# Run continuous latent z.
python Master.py --train=1 --setting=pretrain_sub --name=S12_Pretrain_ContZ --entropy=0 --data=Continuous --kl_weight=0.1 --discrete_z=0
# Continuous latent z with lowered kl / entropy term weight, since entropy magnitude is a lot lower than kl. 
python cluster_run.py --name=S13_Pretrain_ContZ_klwt0pt01 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S13_Pretrain_ContZ_klwt0pt01 --entropy=0 --data=Continuous --kl_weight=0.01 --discrete_z=0'
# Eval
python Master.py --train=0 --setting=pretrain_sub --name=S13_Pretrain_ContZ_klwt0pt01 --entropy=0 --data=Continuous --kl_weight=0.01 --discrete_z=0 --model=Experiment_Logs/S13_Pretrain_ContZ_klwt0pt01/saved_models/Model_epoch48

# Running S11 (discrete subpolicies / latent_z) with more subpolicies. 
python cluster_run.py --name=S14_Pretrain_len5_rightlossv1_nump4_kldwt0pt1 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S14_Pretrain_len5_rightlossv1_nump4_kldwt0pt1 --entropy=0 --data=Continuous --kl_weight=0.1 --number_policies=4'

python cluster_run.py --name=S15_Pretrain_len5_rightlossv1_nump6_kldwt0pt1 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S15_Pretrain_len5_rightlossv1_nump6_kldwt0pt1 --entropy=0 --data=Continuous --kl_weight=0.1 --number_policies=6'

python cluster_run.py --name=S16_Pretrain_len5_rightlossv1_nump8_kldwt0pt1 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S16_Pretrain_len5_rightlossv1_nump8_kldwt0pt1 --entropy=0 --data=Continuous --kl_weight=0.1 --number_policies=8'

python cluster_run.py --name=S17_Pretrain_len5_rightlossv1_nump10_kldwt0pt1 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S17_Pretrain_len5_rightlossv1_nump10_kldwt0pt1 --entropy=0 --data=Continuous --kl_weight=0.1 --number_policies=10'

# # # # # # #
# Running with continuous latent z.
python cluster_run.py --name=S20_Pretrain_ContZ_NewKL_klwt0pt01 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S20_Pretrain_ContZ_NewKL_klwt0pt01 --entropy=0 --data=Continuous --kl_weight=0.01 --discrete_z=0'

python cluster_run.py --name=S21_Pretrain_ContZ_NewKL_klwt0pt1 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S21_Pretrain_ContZ_NewKL_klwt0pt1 --entropy=0 --data=Continuous --kl_weight=0.1 --discrete_z=0'

python cluster_run.py --name=S22_Pretrain_ContZ_NewKL_klwt0pt5 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S22_Pretrain_ContZ_NewKL_klwt0pt5 --entropy=0 --data=Continuous --kl_weight=0.5 --discrete_z=0'

python cluster_run.py --name=S23_Pretrain_ContZ_NewKL_klwt1pt0 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S23_Pretrain_ContZ_NewKL_klwt1pt0 --entropy=0 --data=Continuous --kl_weight=1. --discrete_z=0'

# Rerunning subpolicy pretraining independent of b. 
python cluster_run.py --name=S24_Pretrain_len5_rightlossv1_nump4_kldwt0pt1 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S24_Pretrain_len5_rightlossv1_nump4_kldwt0pt1 --entropy=0 --data=Continuous --kl_weight=0.1 --number_policies=4'
# Eval
python Master.py --train=0 --setting=pretrain_sub --name=S24_Pretrain_len5_rightlossv1_nump4_kldwt0pt1 --entropy=0 --data=Continuous --kl_weight=0.1 --number_policies=4 --model=Experiment_Logs/S24_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch18
# Eval at 20
python Master.py --train=0 --setting=pretrain_sub --name=S24_Pretrain_len5_rightlossv1_nump4_kldwt0pt1 --entropy=0 --data=Continuous --kl_weight=0.1 --number_policies=4 --model=Experiment_Logs/S24_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch20
# Eval at 48
python Master.py --train=0 --setting=pretrain_sub --name=S24_Pretrain_len5_rightlossv1_nump4_kldwt0pt1 --entropy=0 --data=Continuous --kl_weight=0.1 --number_policies=4 --model=Experiment_Logs/S24_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch48

python cluster_run.py --name=S25_Pretrain_len5_rightlossv1_nump6_kldwt0pt1 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S25_Pretrain_len5_rightlossv1_nump6_kldwt0pt1 --entropy=0 --data=Continuous --kl_weight=0.1 --number_policies=6'

python cluster_run.py --name=S26_Pretrain_len5_rightlossv1_nump8_kldwt0pt1 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S26_Pretrain_len5_rightlossv1_nump8_kldwt0pt1 --entropy=0 --data=Continuous --kl_weight=0.1 --number_policies=8'

python cluster_run.py --name=S27_Pretrain_len5_rightlossv1_nump10_kldwt0pt1 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S27_Pretrain_len5_rightlossv1_nump10_kldwt0pt1 --entropy=0 --data=Continuous --kl_weight=0.1 --number_policies=10'

# Continuous z rerun independent of b.
python cluster_run.py --name=S28_Pretrain_ContZ_NewKL_klwt0pt01 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S28_Pretrain_ContZ_NewKL_klwt0pt01 --entropy=0 --data=Continuous --kl_weight=0.01 --discrete_z=0'
# Eval
python Master.py --train=0 --setting=pretrain_sub --name=S28_Eval49 --entropy=0 --data=Continuous --kl_weight=0.01 --discrete_z=0 --model=Experiment_Logs/S28_Pretrain_ContZ_NewKL_klwt0pt01/saved_models/Model_epoch49

python cluster_run.py --name=S29_Pretrain_ContZ_NewKL_klwt0pt1 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S29_Pretrain_ContZ_NewKL_klwt0pt1 --entropy=0 --data=Continuous --kl_weight=0.1 --discrete_z=0'
# Eval
python Master.py --train=0 --setting=pretrain_sub --name=S29_Eval --entropy=0 --data=Continuous --kl_weight=0.1 --discrete_z=0 --model=Experiment_Logs/S29_Pretrain_ContZ_NewKL_klwt0pt1/saved_models/Model_epoch23

python cluster_run.py --name=S30_Pretrain_ContZ_NewKL_klwt0pt5 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S30_Pretrain_ContZ_NewKL_klwt0pt5 --entropy=0 --data=Continuous --kl_weight=0.5 --discrete_z=0'
# Eval
python Master.py --train=0 --setting=pretrain_sub --name=S30_Eval --entropy=0 --data=Continuous --kl_weight=0.5 --discrete_z=0 --model=Experiment_Logs/S30_Pretrain_ContZ_NewKL_klwt0pt5/saved_models/Model_epoch44

python cluster_run.py --name=S31_Pretrain_ContZ_NewKL_klwt1pt0 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S31_Pretrain_ContZ_NewKL_klwt1pt0 --entropy=0 --data=Continuous --kl_weight=1. --discrete_z=0'
# Eval
python Master.py --train=0 --setting=pretrain_sub --name=S31_Eval --entropy=0 --data=Continuous --kl_weight=1. --discrete_z=0 --model=Experiment_Logs/S31_Pretrain_ContZ_NewKL_klwt1pt0/saved_models/Model_epoch25


# Run S24 again. 
python cluster_run.py --name=S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1 --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1 --entropy=0 --data=Continuous --kl_weight=0.1 --number_policies=4'
# Eval at 29
python Master.py --train=0 --setting=pretrain_sub --name=S32_Eval --entropy=0 --data=Continuous --kl_weight=0.1 --number_policies=4 --model=Experiment_Logs/S32_Pretrain_len5_rightlossv1_nump4_kldwt0pt1/saved_models/Model_epoch29

# Rerun continuous with right inputs being provided. 
# Discrete: 
python cluster_run.py --name=S33_Pretrain_len5_rightlossv1_nump4_kldwt0pt1_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S33_Pretrain_len5_rightlossv1_nump4_kldwt0pt1_rightip --entropy=0 --data=Continuous --kl_weight=0.1 --number_policies=4'
# Eval at 4
python Master.py --train=0 --setting=pretrain_sub --name=S33_Pretrain_len5_rightlossv1_nump4_kldwt0pt1_rightip --entropy=0 --data=Continuous --kl_weight=0.1 --number_policies=4 --model=Experiment_Logs/S33_Pretrain_len5_rightlossv1_nump4_kldwt0pt1_rightip/saved_models/Model_epoch36

python cluster_run.py --name=S34_Pretrain_len5_rightlossv1_nump6_kldwt0pt1_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S34_Pretrain_len5_rightlossv1_nump6_kldwt0pt1_rightip --entropy=0 --data=Continuous --kl_weight=0.1 --number_policies=6'

python cluster_run.py --name=S35_Pretrain_len5_rightlossv1_nump8_kldwt0pt1_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S35_Pretrain_len5_rightlossv1_nump8_kldwt0pt1_rightip --entropy=0 --data=Continuous --kl_weight=0.1 --number_policies=8'

python cluster_run.py --name=S36_Pretrain_len5_rightlossv1_nump10_kldwt0pt1_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S36_Pretrain_len5_rightlossv1_nump10_kldwt0pt1_rightip --entropy=0 --data=Continuous --kl_weight=0.1 --number_policies=10'

# Continuous
python cluster_run.py --name=S37_Pretrain_ContZ_NewKL_klwt0pt01_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S37_Pretrain_ContZ_NewKL_klwt0pt01_rightip --entropy=0 --data=Continuous --kl_weight=0.01 --discrete_z=0'
# Eval 
python Master.py --train=0 --setting=pretrain_sub --name=S37_Pretrain_ContZ_NewKL_klwt0pt01_rightip --entropy=0 --data=Continuous --kl_weight=0.01 --discrete_z=0 --model=Experiment_Logs/S37_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch17

python cluster_run.py --name=S38_Pretrain_ContZ_NewKL_klwt0pt1_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S38_Pretrain_ContZ_NewKL_klwt0pt1_rightip --entropy=0 --data=Continuous --kl_weight=0.1 --discrete_z=0'
# Eval
python Master.py --train=0 --setting=pretrain_sub --name=S38_Pretrain_ContZ_NewKL_klwt0pt1_rightip --entropy=0 --data=Continuous --kl_weight=0.1 --discrete_z=0 --model=Experiment_Logs/S38_Pretrain_ContZ_NewKL_klwt0pt1_rightip/saved_models/Model_epoch21

python cluster_run.py --name=S39_Pretrain_ContZ_NewKL_klwt0pt5_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S39_Pretrain_ContZ_NewKL_klwt0pt5_rightip --entropy=0 --data=Continuous --kl_weight=0.5 --discrete_z=0'
python Master.py --train=0 --setting=pretrain_sub --name=S39_Pretrain_ContZ_NewKL_klwt0pt5_rightip --entropy=0 --data=Continuous --kl_weight=0.5 --discrete_z=0 --model=Experiment_Logs/S39_Pretrain_ContZ_NewKL_klwt0pt5_rightip/saved_models/Model_epoch21

python cluster_run.py --name=S40_Pretrain_ContZ_NewKL_klwt1pt0_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S40_Pretrain_ContZ_NewKL_klwt1pt0_rightip --entropy=0 --data=Continuous --kl_weight=1. --discrete_z=0'
python Master.py --train=0 --setting=pretrain_sub --name=S40_Pretrain_ContZ_NewKL_klwt1pt0_rightip --entropy=0 --data=Continuous --kl_weight=1. --discrete_z=0 --model=Experiment_Logs/S40_Pretrain_ContZ_NewKL_klwt1pt0_rightip/saved_models/Model_epoch21

python Master.py --train=1 --setting=pretrain_sub --name=S41_debug --entropy=0 --data=Continuous --kl_weight=1. --discrete_z=0

###### Rerunning with correct padded action sequence. 
# Discrete: 
python cluster_run.py --name=S42_Pretrain_len5_rightlossv1_nump4_kldwt0pt1_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S42_Pretrain_len5_rightlossv1_nump4_kldwt0pt1_rightip --entropy=0 --data=Continuous --kl_weight=0.1 --number_policies=4'
# Eval at 21
python Master.py --train=0 --setting=pretrain_sub --name=S42_Pretrain_len5_rightlossv1_nump4_kldwt0pt1_rightip --entropy=0 --data=Continuous --kl_weight=0.1 --number_policies=4 --model=Experiment_Logs/S42_Pretrain_len5_rightlossv1_nump4_kldwt0pt1_rightip/saved_models/Model_epoch21

python cluster_run.py --name=S43_Pretrain_len5_rightlossv1_nump6_kldwt0pt1_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S43_Pretrain_len5_rightlossv1_nump6_kldwt0pt1_rightip --entropy=0 --data=Continuous --kl_weight=0.1 --number_policies=6'

python cluster_run.py --name=S44_Pretrain_len5_rightlossv1_nump8_kldwt0pt1_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S44_Pretrain_len5_rightlossv1_nump8_kldwt0pt1_rightip --entropy=0 --data=Continuous --kl_weight=0.1 --number_policies=8'

python cluster_run.py --name=S45_Pretrain_len5_rightlossv1_nump10_kldwt0pt1_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S45_Pretrain_len5_rightlossv1_nump10_kldwt0pt1_rightip --entropy=0 --data=Continuous --kl_weight=0.1 --number_policies=10'

# Continuous
python cluster_run.py --name=S46_Pretrain_ContZ_NewKL_klwt0pt01_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S46_Pretrain_ContZ_NewKL_klwt0pt01_rightip --entropy=0 --data=Continuous --kl_weight=0.01 --discrete_z=0'

python cluster_run.py --name=S47_Pretrain_ContZ_NewKL_klwt0pt1_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S47_Pretrain_ContZ_NewKL_klwt0pt1_rightip --entropy=0 --data=Continuous --kl_weight=0.1 --discrete_z=0'

python cluster_run.py --name=S48_Pretrain_ContZ_NewKL_klwt0pt5_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S48_Pretrain_ContZ_NewKL_klwt0pt5_rightip --entropy=0 --data=Continuous --kl_weight=0.5 --discrete_z=0'

python cluster_run.py --name=S49_Pretrain_ContZ_NewKL_klwt1pt0_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S49_Pretrain_ContZ_NewKL_klwt1pt0_rightip --entropy=0 --data=Continuous --kl_weight=1. --discrete_z=0'
# Eval
python Master.py --train=0 --setting=pretrain_sub --name=S49_Pretrain_ContZ_NewKL_klwt1pt0_rightip --entropy=0 --data=Continuous --kl_weight=1. --discrete_z=0 --model=Experiment_Logs/S49_Pretrain_ContZ_NewKL_klwt1pt0_rightip/saved_models/Model_epoch13

# Evaluating correct likelihoods and feeding correct padded action sequence. 
# Discrete: 
python cluster_run.py --name=S50_Pretrain_len5_rightlossv1_nump4_kldwt0pt1_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S50_Pretrain_len5_rightlossv1_nump4_kldwt0pt1_rightip --entropy=0 --data=Continuous --kl_weight=0.1 --number_policies=4'
# Eval
python Master.py --train=0 --setting=pretrain_sub --name=S50_Pretrain_len5_rightlossv1_nump4_kldwt0pt1_rightip --entropy=0 --data=Continuous --kl_weight=0.1 --number_policies=4 --model=Experiment_Logs/S50_Pretrain_len5_rightlossv1_nump4_kldwt0pt1_rightip/saved_models/Model_epoch12 --discrete_z=1

python cluster_run.py --name=S51_Pretrain_len5_rightlossv1_nump6_kldwt0pt1_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S51_Pretrain_len5_rightlossv1_nump6_kldwt0pt1_rightip --entropy=0 --data=Continuous --kl_weight=0.1 --number_policies=6'

python cluster_run.py --name=S52_Pretrain_len5_rightlossv1_nump8_kldwt0pt1_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S52_Pretrain_len5_rightlossv1_nump8_kldwt0pt1_rightip --entropy=0 --data=Continuous --kl_weight=0.1 --number_policies=8'

python cluster_run.py --name=S53_Pretrain_len5_rightlossv1_nump10_kldwt0pt1_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S53_Pretrain_len5_rightlossv1_nump10_kldwt0pt1_rightip --entropy=0 --data=Continuous --kl_weight=0.1 --number_policies=10'

python cluster_run.py --name=S54_Pretrain_ContZ_NewKL_klwt0pt01_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S54_Pretrain_ContZ_NewKL_klwt0pt01_rightip --entropy=0 --data=Continuous --kl_weight=0.01 --discrete_z=0'
# Eval 
python Master.py --train=0 --setting=pretrain_sub --name=S54_Pretrain_ContZ_NewKL_klwt0pt01_rightip --entropy=0 --data=Continuous --kl_weight=0.01 --discrete_z=0  --model=Experiment_Logs/S54_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch18
python Master.py --train=0 --setting=pretrain_sub --name=S54_Pretrain_ContZ_NewKL_klwt0pt01_rightip --entropy=0 --data=Continuous --kl_weight=0.01 --discrete_z=0  --model=Experiment_Logs/S54_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch20
python Master.py --train=0 --setting=pretrain_sub --name=S54_Pretrain_ContZ_NewKL_klwt0pt01_rightip --entropy=0 --data=Continuous --kl_weight=0.01 --discrete_z=0  --model=Experiment_Logs/S54_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch22

python cluster_run.py --name=S55_Pretrain_ContZ_NewKL_klwt0pt1_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S55_Pretrain_ContZ_NewKL_klwt0pt1_rightip --entropy=0 --data=Continuous --kl_weight=0.1 --discrete_z=0'
python Master.py --train=0 --setting=pretrain_sub --name=S55_Pretrain_ContZ_NewKL_klwt0pt1_rightip --entropy=0 --data=Continuous --kl_weight=0.1 --discrete_z=0 --model=Experiment_Logs/S55_Pretrain_ContZ_NewKL_klwt0pt1_rightip/saved_models/Model_epoch38
python Master.py --train=0 --setting=pretrain_sub --name=S55_Pretrain_ContZ_NewKL_klwt0pt1_rightip --entropy=0 --data=Continuous --kl_weight=0.1 --discrete_z=0 --model=Experiment_Logs/S55_Pretrain_ContZ_NewKL_klwt0pt1_rightip/saved_models/Model_epoch49
python Master.py --train=0 --setting=pretrain_sub --name=S55_Pretrain_ContZ_NewKL_klwt0pt1_rightip --entropy=0 --data=Continuous --kl_weight=0.1 --discrete_z=0 --model=Experiment_Logs/S55_Pretrain_ContZ_NewKL_klwt0pt1_rightip/saved_models/Model_epoch49 --debug=1

python cluster_run.py --name=S56_Pretrain_ContZ_NewKL_klwt0pt5_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S56_Pretrain_ContZ_NewKL_klwt0pt5_rightip --entropy=0 --data=Continuous --kl_weight=0.5 --discrete_z=0'

python cluster_run.py --name=S57_Pretrain_ContZ_NewKL_klwt1pt0_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S57_Pretrain_ContZ_NewKL_klwt1pt0_rightip --entropy=0 --data=Continuous --kl_weight=1. --discrete_z=0'
# Eval at 3
python Master.py --train=0 --setting=pretrain_sub --name=S57_Pretrain_ContZ_NewKL_klwt1pt0_rightip --entropy=0 --data=Continuous --kl_weight=1. --discrete_z=0 --model=Experiment_Logs/S57_Pretrain_ContZ_NewKL_klwt1pt0_rightip/saved_models/Model_epoch16

# Run S50 again because of preemption
python cluster_run.py --name=S58_Pretrain_len5_rightlossv1_nump4_kldwt0pt1_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S58_Pretrain_len5_rightlossv1_nump4_kldwt0pt1_rightip --entropy=0 --data=Continuous --kl_weight=0.1 --number_policies=4 --model=Experiment_Logs/S50_Pretrain_len5_rightlossv1_nump4_kldwt0pt1_rightip/saved_models/Model_epoch12'
# Eval
python Master.py --train=0 --setting=pretrain_sub --name=S58_Pretrain_len5_rightlossv1_nump4_kldwt0pt1_rightip --entropy=0 --data=Continuous --kl_weight=0.1 --number_policies=4 --model=Experiment_Logs/S58_Pretrain_len5_rightlossv1_nump4_kldwt0pt1_rightip/saved_models/Model_epoch24

python cluster_run.py --name=S59_Pretrain_len5_rightlossv1_nump4_kldwt0pt1_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S59_Pretrain_len5_rightlossv1_nump4_kldwt0pt1_rightip --entropy=0 --data=Continuous --kl_weight=0.1 --number_policies=4'

# Run contz loading S29.
python cluster_run.py --name=S60_Pretrain_ContZ_NewKL_klwt0pt01_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S60_Pretrain_ContZ_NewKL_klwt0pt01_rightip --entropy=0 --data=Continuous --kl_weight=0.01 --discrete_z=0 --model=Experiment_Logs/S29_Pretrain_ContZ_NewKL_klwt0pt1/saved_models/Model_epoch15'
python Master.py --train=0 --setting=pretrain_sub --name=S60_Pretrain_ContZ_NewKL_klwt0pt01_rightip --entropy=0 --data=Continuous --kl_weight=0.01 --discrete_z=0 --model=Experiment_Logs/S60_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch30
python Master.py --train=0 --setting=pretrain_sub --name=S60_Pretrain_ContZ_NewKL_klwt0pt01_rightip --entropy=0 --data=Continuous --kl_weight=0.01 --discrete_z=0 --model=Experiment_Logs/S60_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch49
python Master.py --train=0 --setting=pretrain_sub --name=S60_Pretrain_ContZ_NewKL_klwt0pt01_rightip --entropy=0 --data=Continuous --kl_weight=0.01 --discrete_z=0 --model=Experiment_Logs/S60_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch49 --debug=1

python cluster_run.py --name=S61_Pretrain_ContZ_NewKL_klwt0pt1_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S61_Pretrain_ContZ_NewKL_klwt0pt1_rightip --entropy=0 --data=Continuous --kl_weight=0.1 --discrete_z=0 --model=Experiment_Logs/S29_Pretrain_ContZ_NewKL_klwt0pt1/saved_models/Model_epoch15'
# Eval
python Master.py --train=0 --setting=pretrain_sub --name=S61_Pretrain_ContZ_NewKL_klwt0pt1_rightip --entropy=0 --data=Continuous --kl_weight=0.1 --discrete_z=0 --model=Experiment_Logs/S61_Pretrain_ContZ_NewKL_klwt0pt1_rightip/saved_models/Model_epoch32
python Master.py --train=0 --setting=pretrain_sub --name=S61_Pretrain_ContZ_NewKL_klwt0pt1_rightip --entropy=0 --data=Continuous --kl_weight=0.1 --discrete_z=0 --model=Experiment_Logs/S61_Pretrain_ContZ_NewKL_klwt0pt1_rightip/saved_models/Model_epoch49
python Master.py --train=0 --setting=pretrain_sub --name=S61_Pretrain_ContZ_NewKL_klwt0pt1_rightip --entropy=0 --data=Continuous --kl_weight=0.1 --discrete_z=0 --model=Experiment_Logs/S61_Pretrain_ContZ_NewKL_klwt0pt1_rightip/saved_models/Model_epoch49 --debug=1

python cluster_run.py --name=S62_Pretrain_ContZ_NewKL_klwt0pt5_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S62_Pretrain_ContZ_NewKL_klwt0pt5_rightip --entropy=0 --data=Continuous --kl_weight=0.5 --discrete_z=0 --model=Experiment_Logs/S29_Pretrain_ContZ_NewKL_klwt0pt1/saved_models/Model_epoch15'

python cluster_run.py --name=S63_Pretrain_ContZ_NewKL_klwt1pt0_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S63_Pretrain_ContZ_NewKL_klwt1pt0_rightip --entropy=0 --data=Continuous --kl_weight=1. --discrete_z=0 --model=Experiment_Logs/S29_Pretrain_ContZ_NewKL_klwt0pt1/saved_models/Model_epoch15'
# Eval
python Master.py --train=0 --setting=pretrain_sub --name=S63_Pretrain_ContZ_NewKL_klwt1pt0_rightip --entropy=0 --data=Continuous --kl_weight=1. --discrete_z=0 --model=Experiment_Logs/S63_Pretrain_ContZ_NewKL_klwt1pt0_rightip/saved_models/Model_epoch17

# Run S50 again as S64 because of preemption, now with discrete z.
python cluster_run.py --name=S64_Pretrain_len5_rightlossv1_nump4_kldwt0pt1_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S64_Pretrain_len5_rightlossv1_nump4_kldwt0pt1_rightip --entropy=0 --data=Continuous --kl_weight=0.1 --number_policies=4 --model=Experiment_Logs/S50_Pretrain_len5_rightlossv1_nump4_kldwt0pt1_rightip/saved_models/Model_epoch12 --discrete_z=1'

# # # # # 
# Running without tanh activation being used.
python cluster_run.py --name=S65_Pretrain_ContZ_NewKL_klwt0pt01_notanh --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S65_Pretrain_ContZ_NewKL_klwt0pt01_notanh --entropy=0 --data=Continuous --kl_weight=0.01 --discrete_z=0 --mean_nonlinearity=0'

python cluster_run.py --name=S66_Pretrain_ContZ_NewKL_klwt0pt1_notanh --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S66_Pretrain_ContZ_NewKL_klwt0pt1_notanh --entropy=0 --data=Continuous --kl_weight=0.1 --discrete_z=0 --mean_nonlinearity=0'
python Master.py --train=0 --setting=pretrain_sub --name=S66_Pretrain_ContZ_NewKL_klwt0pt1_notanh --entropy=0 --data=Continuous --kl_weight=0.1 --discrete_z=0 --mean_nonlinearity=0 --model=Experiment_Logs/S66_Pretrain_ContZ_NewKL_klwt0pt1_notanh/saved_models/Model_epoch15
python Master.py --train=0 --setting=pretrain_sub --name=S66_Pretrain_ContZ_NewKL_klwt0pt1_notanh --entropy=0 --data=Continuous --kl_weight=0.1 --discrete_z=0 --mean_nonlinearity=0 --model=Experiment_Logs/S66_Pretrain_ContZ_NewKL_klwt0pt1_notanh/saved_models/Model_epoch28
python Master.py --train=0 --setting=pretrain_sub --name=S66_Pretrain_ContZ_NewKL_klwt0pt1_notanh --entropy=0 --data=Continuous --kl_weight=0.1 --discrete_z=0 --mean_nonlinearity=0 --model=Experiment_Logs/S66_Pretrain_ContZ_NewKL_klwt0pt1_notanh/saved_models/Model_epoch42

python cluster_run.py --name=S67_Pretrain_ContZ_NewKL_klwt0pt5_notanh --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S67_Pretrain_ContZ_NewKL_klwt0pt5_notanh --entropy=0 --data=Continuous --kl_weight=0.5 --discrete_z=0 --mean_nonlinearity=0'

python cluster_run.py --name=S68_Pretrain_ContZ_NewKL_klwt1pt0_notanh --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S68_Pretrain_ContZ_NewKL_klwt1pt0_notanh --entropy=0 --data=Continuous --kl_weight=1. --discrete_z=0 --mean_nonlinearity=0'

# Running with tanh activation being used - but without states being provided to the encoder. 
python cluster_run.py --name=S69_Pretrain_ContZ_NewKL_klwt0pt01_tanh --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S69_Pretrain_ContZ_NewKL_klwt0pt01_tanh --entropy=0 --data=Continuous --kl_weight=0.01 --discrete_z=0 --mean_nonlinearity=1 --state_dependent_z=0'

python cluster_run.py --name=S70_Pretrain_ContZ_NewKL_klwt0pt1_tanh --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S70_Pretrain_ContZ_NewKL_klwt0pt1_tanh --entropy=0 --data=Continuous --kl_weight=0.1 --discrete_z=0 --mean_nonlinearity=1 --state_dependent_z=0'

python cluster_run.py --name=S71_Pretrain_ContZ_NewKL_klwt0pt5_tanh --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S71_Pretrain_ContZ_NewKL_klwt0pt5_tanh --entropy=0 --data=Continuous --kl_weight=0.5 --discrete_z=0 --mean_nonlinearity=1 --state_dependent_z=0'

python cluster_run.py --name=S72_Pretrain_ContZ_NewKL_klwt1pt0_tanh --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S72_Pretrain_ContZ_NewKL_klwt1pt0_tanh --entropy=0 --data=Continuous --kl_weight=1. --discrete_z=0 --mean_nonlinearity=1 --state_dependent_z=0'

# Running without tanh activation being used, and without states being provided to the encoder.
python cluster_run.py --name=S73_Pretrain_ContZ_NewKL_klwt0pt01_notanh --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S73_Pretrain_ContZ_NewKL_klwt0pt01_notanh --entropy=0 --data=Continuous --kl_weight=0.01 --discrete_z=0 --mean_nonlinearity=0 --state_dependent_z=0'
# Eval 
python Master.py --train=0 --setting=pretrain_sub --name=S73_Pretrain_ContZ_NewKL_klwt0pt01_notanh --entropy=0 --data=Continuous --kl_weight=0.01 --discrete_z=0 --mean_nonlinearity=0 --state_dependent_z=0 --model=Experiment_Logs/S73_Pretrain_ContZ_NewKL_klwt0pt01_notanh/saved_models/Model_epoch18
python Master.py --train=0 --setting=pretrain_sub --name=S73_Pretrain_ContZ_NewKL_klwt0pt01_notanh --entropy=0 --data=Continuous --kl_weight=0.01 --discrete_z=0 --mean_nonlinearity=0 --state_dependent_z=0 --model=Experiment_Logs/S73_Pretrain_ContZ_NewKL_klwt0pt01_notanh/saved_models/Model_epoch20

python cluster_run.py --name=S74_Pretrain_ContZ_NewKL_klwt0pt1_notanh --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S74_Pretrain_ContZ_NewKL_klwt0pt1_notanh --entropy=0 --data=Continuous --kl_weight=0.1 --discrete_z=0 --mean_nonlinearity=0 --state_dependent_z=0'

python cluster_run.py --name=S75_Pretrain_ContZ_NewKL_klwt0pt5_notanh --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S75_Pretrain_ContZ_NewKL_klwt0pt5_notanh --entropy=0 --data=Continuous --kl_weight=0.5 --discrete_z=0 --mean_nonlinearity=0 --state_dependent_z=0'

python cluster_run.py --name=S76_Pretrain_ContZ_NewKL_klwt1pt0_notanh --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S76_Pretrain_ContZ_NewKL_klwt1pt0_notanh --entropy=0 --data=Continuous --kl_weight=1. --discrete_z=0 --mean_nonlinearity=0 --state_dependent_z=0'

# Running on ContinuousNonZero data.
python cluster_run.py --partition=priority --name=S77_Pretrain_ContZ_NewKL_klwt0pt01_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S77_Pretrain_ContZ_NewKL_klwt0pt01_rightip --entropy=0 --data=ContinuousNonZero --kl_weight=0.01 --discrete_z=0'
# Eval 
python Master.py --train=0 --setting=pretrain_sub --name=S77_Pretrain_ContZ_NewKL_klwt0pt01_rightip --entropy=0 --data=ContinuousNonZero --kl_weight=0.01 --discrete_z=0 --model=Experiment_Logs/S77_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch14
python Master.py --train=0 --setting=pretrain_sub --name=S77_Pretrain_ContZ_NewKL_klwt0pt01_rightip --entropy=0 --data=ContinuousNonZero --kl_weight=0.01 --discrete_z=0 --model=Experiment_Logs/S77_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch30
python Master.py --train=0 --setting=pretrain_sub --name=S77_Pretrain_ContZ_NewKL_klwt0pt01_rightip --entropy=0 --data=ContinuousNonZero --kl_weight=0.01 --discrete_z=0 --model=Experiment_Logs/S77_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch33
# Debug
python Master.py --train=1 --setting=pretrain_sub --name=SNonZero_Debug --entropy=0 --data=ContinuousNonZero --kl_weight=0.01 --discrete_z=0 --debug=1

python cluster_run.py --partition=priority --name=S78_Pretrain_ContZ_NewKL_klwt0pt1_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S78_Pretrain_ContZ_NewKL_klwt0pt1_rightip --entropy=0 --data=ContinuousNonZero --kl_weight=0.1 --discrete_z=0'
python Master.py --train=0 --setting=pretrain_sub --name=S78_Eval --entropy=0 --data=ContinuousNonZero --kl_weight=0.1 --discrete_z=0 --model=Experiment_Logs/S78_Pretrain_ContZ_NewKL_klwt0pt1_rightip/saved_models/Model_epoch22
python Master.py --train=0 --setting=pretrain_sub --name=S78_Eval --entropy=0 --data=ContinuousNonZero --kl_weight=0.1 --discrete_z=0 --model=Experiment_Logs/S78_Pretrain_ContZ_NewKL_klwt0pt1_rightip/saved_models/Model_epoch34

python cluster_run.py --partition=priority --name=S79_Pretrain_ContZ_NewKL_klwt0pt5_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S79_Pretrain_ContZ_NewKL_klwt0pt5_rightip --entropy=0 --data=ContinuousNonZero --kl_weight=0.5 --discrete_z=0'

python cluster_run.py --partition=learnfair --name=S80_Pretrain_ContZ_NewKL_klwt0pt01_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S80_Pretrain_ContZ_NewKL_klwt0pt01_rightip --entropy=0 --data=ContinuousNonZero --kl_weight=0.01 --discrete_z=0'
# Eval
python Master.py --train=0 --setting=pretrain_sub --name=S80_Pretrain_ContZ_NewKL_klwt0pt01_rightip --entropy=0 --data=ContinuousNonZero --kl_weight=0.01 --discrete_z=0 --model=Experiment_Logs/S80_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch16
python Master.py --train=0 --setting=pretrain_sub --name=S80_Pretrain_ContZ_NewKL_klwt0pt01_rightip --entropy=0 --data=ContinuousNonZero --kl_weight=0.01 --discrete_z=0 --model=Experiment_Logs/S80_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch20
python Master.py --train=0 --setting=pretrain_sub --name=S80_Pretrain_ContZ_NewKL_klwt0pt01_rightip --entropy=0 --data=ContinuousNonZero --kl_weight=0.01 --discrete_z=0 --model=Experiment_Logs/S80_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch49

python cluster_run.py --partition=learnfair --name=S81_Pretrain_ContZ_NewKL_klwt0pt1_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S81_Pretrain_ContZ_NewKL_klwt0pt1_rightip --entropy=0 --data=ContinuousNonZero --kl_weight=0.1 --discrete_z=0'

python cluster_run.py --partition=learnfair --name=S82_Pretrain_ContZ_NewKL_klwt0pt5_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S82_Pretrain_ContZ_NewKL_klwt0pt5_rightip --entropy=0 --data=ContinuousNonZero --kl_weight=0.5 --discrete_z=0'

python cluster_run.py --partition=learnfair --name=S83_Pretrain_ContZ_NewKL_klwt1pt0_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S83_Pretrain_ContZ_NewKL_klwt1pt0_rightip --entropy=0 --data=ContinuousNonZero --kl_weight=1. --discrete_z=0'

# Load S80 model 22.
python cluster_run.py --partition=learnfair --name=S84_Pretrain_ContZ_NewKL_klwt0pt01_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S84_Pretrain_ContZ_NewKL_klwt0pt01_rightip --entropy=0 --data=ContinuousNonZero --kl_weight=0.01 --discrete_z=0 --model=Experiment_Logs/S80_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch22'

python cluster_run.py --partition=learnfair --name=S85_Pretrain_ContZ_NewKL_klwt0pt05_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S85_Pretrain_ContZ_NewKL_klwt0pt05_rightip --entropy=0 --data=ContinuousNonZero --kl_weight=0.05 --discrete_z=0 --model=Experiment_Logs/S80_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch22'
python Master.py --train=0 --setting=pretrain_sub --name=S85_Pretrain_ContZ_NewKL_klwt0pt05_rightip --entropy=0 --data=ContinuousNonZero --kl_weight=0.05 --discrete_z=0 --model=Experiment_Logs/S85_Pretrain_ContZ_NewKL_klwt0pt05_rightip/saved_models/Model_epoch39

python cluster_run.py --partition=learnfair --name=S86_Pretrain_ContZ_NewKL_klwt0pt1_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S86_Pretrain_ContZ_NewKL_klwt0pt1_rightip --entropy=0 --data=ContinuousNonZero --kl_weight=0.1 --discrete_z=0 --model=Experiment_Logs/S80_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch22'

# Pretrain for much longer
python cluster_run.py --partition=learnfair --name=S87_Pretrain_ContZ_NewKL_klwt0pt01_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S87_Pretrain_ContZ_NewKL_klwt0pt01_rightip --entropy=0 --data=ContinuousNonZero --kl_weight=0.01 --discrete_z=0 --model=Experiment_Logs/S80_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch22'


python cluster_run.py --partition=learnfair --name=S88_Pretrain_ContZ_NewKL_klwt0pt05_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S88_Pretrain_ContZ_NewKL_klwt0pt05_rightip --entropy=0 --data=ContinuousNonZero --kl_weight=0.05 --discrete_z=0 --model=Experiment_Logs/S80_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch22'

python cluster_run.py --partition=learnfair --name=S89_Pretrain_ContZ_NewKL_klwt0pt1_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S89_Pretrain_ContZ_NewKL_klwt0pt1_rightip --entropy=0 --data=ContinuousNonZero --kl_weight=0.1 --discrete_z=0 --model=Experiment_Logs/S80_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch22'

# Load S85
python cluster_run.py --partition=learnfair --name=S90_Pretrain_ContZ_NewKL_klwt0pt01_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S90_Pretrain_ContZ_NewKL_klwt0pt01_rightip --entropy=0 --data=ContinuousNonZero --kl_weight=0.01 --discrete_z=0 --model=Experiment_Logs/S85_Pretrain_ContZ_NewKL_klwt0pt05_rightip/saved_models/Model_epoch39'
# Eval at 10
python Master.py --train=0 --setting=pretrain_sub --name=S90_Pretrain_ContZ_NewKL_klwt0pt01_rightip --entropy=0 --data=ContinuousNonZero --kl_weight=0.01 --discrete_z=0 --model=Experiment_Logs/S90_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch10
python Master.py --train=0 --setting=pretrain_sub --name=S90_Pretrain_ContZ_NewKL_klwt0pt01_rightip --entropy=0 --data=ContinuousNonZero --kl_weight=0.01 --discrete_z=0 --model=Experiment_Logs/S90_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch35

python cluster_run.py --partition=learnfair --name=S91_Pretrain_ContZ_NewKL_klwt0pt05_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S91_Pretrain_ContZ_NewKL_klwt0pt05_rightip --entropy=0 --data=ContinuousNonZero --kl_weight=0.05 --discrete_z=0 --model=Experiment_Logs/S85_Pretrain_ContZ_NewKL_klwt0pt05_rightip/saved_models/Model_epoch39'
python Master.py --train=0 --setting=pretrain_sub --name=S91_Pretrain_ContZ_NewKL_klwt0pt05_rightip --entropy=0 --data=ContinuousNonZero --kl_weight=0.05 --discrete_z=0 --model=Experiment_Logs/S91_Pretrain_ContZ_NewKL_klwt0pt05_rightip/saved_models/Model_epoch30
python Master.py --train=0 --setting=pretrain_sub --name=S91_Pretrain_ContZ_NewKL_klwt0pt05_rightip --entropy=0 --data=ContinuousNonZero --kl_weight=0.05 --discrete_z=0 --model=Experiment_Logs/S91_Pretrain_ContZ_NewKL_klwt0pt05_rightip/saved_models/Model_epoch35

python cluster_run.py --partition=learnfair --name=S92_Pretrain_ContZ_NewKL_klwt0pt1_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S92_Pretrain_ContZ_NewKL_klwt0pt1_rightip --entropy=0 --data=ContinuousNonZero --kl_weight=0.1 --discrete_z=0 --model=Experiment_Logs/S85_Pretrain_ContZ_NewKL_klwt0pt05_rightip/saved_models/Model_epoch39'
python Master.py --train=0 --setting=pretrain_sub --entropy=0 --data=ContinuousNonZero --kl_weight=0.1 --discrete_z=0 --model=Experiment_Logs/S92_Pretrain_ContZ_NewKL_klwt0pt1_rightip/saved_models/Model_epoch5 --name=S92_L5
python Master.py --train=0 --setting=pretrain_sub --entropy=0 --data=ContinuousNonZero --kl_weight=0.1 --discrete_z=0 --model=Experiment_Logs/S92_Pretrain_ContZ_NewKL_klwt0pt1_rightip/saved_models/Model_epoch50 --name=S92_L50

# Trying with small z dimensions.
python cluster_run.py --partition=learnfair --name=S93_Pretrain_ContZ_NewKL_klwt0pt01_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S93_Pretrain_ContZ_NewKL_klwt0pt01_rightip --entropy=0 --data=ContinuousNonZero --kl_weight=0.01 --discrete_z=0 --z_dimensions=1'
# 
python Master.py --train=1 --setting=pretrain_sub --name=S9Debug --entropy=0 --data=ContinuousNonZero --kl_weight=0.01 --discrete_z=0 --z_dimensions=1 --debug=1

python cluster_run.py --partition=learnfair --name=S94_Pretrain_ContZ_NewKL_klwt0pt05_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S94_Pretrain_ContZ_NewKL_klwt0pt05_rightip --entropy=0 --data=ContinuousNonZero --kl_weight=0.05 --discrete_z=0 --z_dimensions=1'
# Eval
python Master.py --train=0 --setting=pretrain_sub --name=S94_Pretrain_ContZ_NewKL_klwt0pt05_rightip --entropy=0 --data=ContinuousNonZero --kl_weight=0.05 --discrete_z=0 --z_dimensions=1 --model=Experiment_Logs/S94_Pretrain_ContZ_NewKL_klwt0pt05_rightip/saved_models/Model_epoch30

python cluster_run.py --partition=learnfair --name=S95_Pretrain_ContZ_NewKL_klwt0pt1_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S95_Pretrain_ContZ_NewKL_klwt0pt1_rightip --entropy=0 --data=ContinuousNonZero --kl_weight=0.1 --discrete_z=0 --z_dimensions=1'
python Master.py --train=0 --setting=pretrain_sub --name=S95_Pretrain_ContZ_NewKL_klwt0pt1_rightip --entropy=0 --data=ContinuousNonZero --kl_weight=0.1 --discrete_z=0 --z_dimensions=1 --model=Experiment_Logs/S95_Pretrain_ContZ_NewKL_klwt0pt1_rightip/saved_models/Model_epoch25

python cluster_run.py --partition=learnfair --name=S96_Pretrain_ContZ_NewKL_klwt0pt01_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S96_Pretrain_ContZ_NewKL_klwt0pt01_rightip --entropy=0 --data=ContinuousNonZero --kl_weight=0.01 --discrete_z=0 --z_dimensions=2'
python Master.py --train=0 --setting=pretrain_sub --name=S96_Pretrain_ContZ_NewKL_klwt0pt01_rightip --entropy=0 --data=ContinuousNonZero --kl_weight=0.01 --discrete_z=0 --z_dimensions=2 --model=Experiment_Logs/S96_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch30

python cluster_run.py --partition=learnfair --name=S97_Pretrain_ContZ_NewKL_klwt0pt05_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S97_Pretrain_ContZ_NewKL_klwt0pt05_rightip --entropy=0 --data=ContinuousNonZero --kl_weight=0.05 --discrete_z=0 --z_dimensions=2'
# Eval 
python Master.py --train=0 --setting=pretrain_sub --name=S97_Pretrain_ContZ_NewKL_klwt0pt05_rightip --entropy=0 --data=ContinuousNonZero --kl_weight=0.05 --discrete_z=0 --z_dimensions=2 --model=Experiment_Logs/S97_Pretrain_ContZ_NewKL_klwt0pt05_rightip/saved_models/Model_epoch25

python cluster_run.py --partition=learnfair --name=S98_Pretrain_ContZ_NewKL_klwt0pt1_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S98_Pretrain_ContZ_NewKL_klwt0pt1_rightip --entropy=0 --data=ContinuousNonZero --kl_weight=0.1 --discrete_z=0 --z_dimensions=2'

python cluster_run.py --partition=learnfair --name=S99_Pretrain_ContZ_NewKL_klwt0pt01_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S99_Pretrain_ContZ_NewKL_klwt0pt01_rightip --entropy=0 --data=ContinuousNonZero --kl_weight=0.01 --discrete_z=0 --z_dimensions=4'

python cluster_run.py --partition=learnfair --name=S100_Pretrain_ContZ_NewKL_klwt0pt05_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S100_Pretrain_ContZ_NewKL_klwt0pt05_rightip --entropy=0 --data=ContinuousNonZero --kl_weight=0.05 --discrete_z=0 --z_dimensions=4'
python Master.py --train=0 --setting=pretrain_sub --name=S100_Pretrain_ContZ_NewKL_klwt0pt05_rightip --entropy=0 --data=ContinuousNonZero --kl_weight=0.05 --discrete_z=0 --z_dimensions=4 --model=Experiment_Logs/S100_Pretrain_ContZ_NewKL_klwt0pt05_rightip/saved_models/Model_epoch20

python cluster_run.py --partition=learnfair --name=S101_Pretrain_ContZ_NewKL_klwt0pt1_rightip --cmd='python Master.py --train=1 --setting=pretrain_sub --name=S101_Pretrain_ContZ_NewKL_klwt0pt1_rightip --entropy=0 --data=ContinuousNonZero --kl_weight=0.1 --discrete_z=0 --z_dimensions=4'
python Master.py --train=0 --setting=pretrain_sub --name=S101_Pretrain_ContZ_NewKL_klwt0pt1_rightip --entropy=0 --data=ContinuousNonZero --kl_weight=0.1 --discrete_z=0 --z_dimensions=4

###############################################################
######################## BATCH TRIALS #########################
###############################################################

# # Running with batch: Running trials. 
# python Master.py --train=1 --setting=batchgtsub --traj_length=-1 --likelihood_penalty=5 --name=TBatchTrial --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=20

# python Master.py --train=1 --setting=batchgtsub --traj_length=-1 --likelihood_penalty=5 --name=TBatchTrial2 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=20 --batch_size=20

# # Running actual runs with varying batch sizes to try speed. 
# python cluster_run.py --name=B1_bs1 --cmd='python Master.py --train=1 --setting=batchgtsub --traj_length=-1 --likelihood_penalty=5 --name=B1_bs1 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=20 --batch_size=1'

# python cluster_run.py --name=B2_bs5 --cmd='python Master.py --train=1 --setting=batchgtsub --traj_length=-1 --likelihood_penalty=5 --name=B2_bs5 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=20 --batch_size=5'

# python cluster_run.py --name=B3_bs10 --cmd='python Master.py --train=1 --setting=batchgtsub --traj_length=-1 --likelihood_penalty=5 --name=B3_bs10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=20 --batch_size=10'

# python cluster_run.py --name=B4_bs20 --cmd='python Master.py --train=1 --setting=batchgtsub --traj_length=-1 --likelihood_penalty=5 --name=B4_bs20 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=20 --batch_size=20'

# python cluster_run.py --name=B5_bs50 --cmd='python Master.py --train=1 --setting=batchgtsub --traj_length=-1 --likelihood_penalty=5 --name=B5_bs50 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.3 --data=ContinuousDir --z_ex_bias=0.3 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=20 --batch_size=50'

# python cluster_run.py --name=B6_bs20_len10 --cmd='python Master.py --train=1 --setting=batchgtsub --traj_length=10 --likelihood_penalty=5 --name=B6_bs20_len10 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --b_ex_bias=0.1 --data=ContinuousDir --z_ex_bias=0.1 --epsilon_from=0.5 --epsilon_to=0.05 --epsilon_over=20 --batch_size=20'

###############################################################
##################### CONTINUOUS TRIALS #######################
###############################################################

python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=5 --name=C1_loadS29 --ent_weight=0. --entropy=0 --var_entropy=1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --data=Continuous --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --subpolicy_model=Experiment_Logs/S29_Pretrain_ContZ_NewKL_klwt0pt1/saved_models/Model_epoch23

# C1 with latent policy ratio set to 0.1
python cluster_run.py --name=C1_loadS29_subpolrat0pt1_latpolrat0pt1 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=5 --name=C1_loadS29_subpolrat0pt1_latpolrat0pt1 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=Continuous --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --subpolicy_model=Experiment_Logs/S29_Pretrain_ContZ_NewKL_klwt0pt1/saved_models/Model_epoch23'

python cluster_run.py --name=C2_loadS29_subpolrat0pt1_latpolrat0pt01 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=5 --name=C2_loadS29_subpolrat0pt1_latpolrat0pt01 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=Continuous --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --subpolicy_model=Experiment_Logs/S29_Pretrain_ContZ_NewKL_klwt0pt1/saved_models/Model_epoch23'

python cluster_run.py --name=C3_loadS29_subpolrat0pt1_latpolrat0pt001 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=5 --name=C2_loadS29_subpolrat0pt1_latpolrat0pt001 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=Continuous --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --subpolicy_model=Experiment_Logs/S29_Pretrain_ContZ_NewKL_klwt0pt1/saved_models/Model_epoch23'

# Now running with newly trained autoencoding networks S55, S58, S60, S61. 
python cluster_run.py --name=C4_loadS55_subpolrat0pt1_latpolrat0pt1 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=5 --name=C4_loadS55_subpolrat0pt1_latpolrat0pt1 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=Continuous --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --subpolicy_model=Experiment_Logs/S55_Pretrain_ContZ_NewKL_klwt0pt1_rightip/saved_models/Model_epoch38'

python cluster_run.py --name=C5_loadS55_subpolrat0pt1_latpolrat0pt01 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=5 --name=C5_loadS55_subpolrat0pt1_latpolrat0pt01 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=Continuous --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --subpolicy_model=Experiment_Logs/S55_Pretrain_ContZ_NewKL_klwt0pt1_rightip/saved_models/Model_epoch38'

python cluster_run.py --name=C6_loadS55_subpolrat0pt1_latpolrat0pt001 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=5 --name=C6_loadS55_subpolrat0pt1_latpolrat0pt001 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=Continuous --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --subpolicy_model=Experiment_Logs/S55_Pretrain_ContZ_NewKL_klwt0pt1_rightip/saved_models/Model_epoch38'

# Run with S55 without tanh nonlinearity. This might be weird, but who cares. 
python cluster_run.py --name=C7_loadS55_subpolrat0pt1_latpolrat0pt1 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=5 --name=C7_loadS55_subpolrat0pt1_latpolrat0pt1 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --mean_nonlinearity=0 --data=Continuous --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --subpolicy_model=Experiment_Logs/S55_Pretrain_ContZ_NewKL_klwt0pt1_rightip/saved_models/Model_epoch38'

python cluster_run.py --name=C8_loadS55_subpolrat0pt1_latpolrat0pt01 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=5 --name=C8_loadS55_subpolrat0pt1_latpolrat0pt01 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --mean_nonlinearity=0 --data=Continuous --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --subpolicy_model=Experiment_Logs/S55_Pretrain_ContZ_NewKL_klwt0pt1_rightip/saved_models/Model_epoch38'

python cluster_run.py --name=C9_loadS55_subpolrat0pt1_latpolrat0pt001 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=5 --name=C9_loadS55_subpolrat0pt1_latpolrat0pt001 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --mean_nonlinearity=0 --data=Continuous --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --subpolicy_model=Experiment_Logs/S55_Pretrain_ContZ_NewKL_klwt0pt1_rightip/saved_models/Model_epoch38'

# Run with S55 and KL weights.
python cluster_run.py --name=C10_loadS55_subpolrat0pt1_latpolrat0pt1_kl0pt1 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=5 --name=C10_loadS55_subpolrat0pt1_latpolrat0pt1_kl0pt1 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=Continuous --kl_weight=0.1 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --subpolicy_model=Experiment_Logs/S55_Pretrain_ContZ_NewKL_klwt0pt1_rightip/saved_models/Model_epoch49'

python cluster_run.py --name=C11_loadS55_subpolrat0pt1_latpolrat0pt01_kl0pt1 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=5 --name=C11_loadS55_subpolrat0pt1_latpolrat0pt01_kl0pt1 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=Continuous --kl_weight=0.1 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --subpolicy_model=Experiment_Logs/S55_Pretrain_ContZ_NewKL_klwt0pt1_rightip/saved_models/Model_epoch49'

python cluster_run.py --name=C12_loadS55_subpolrat0pt1_latpolrat0pt001_kl0pt1 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=5 --name=C12_loadS55_subpolrat0pt1_latpolrat0pt001_kl0pt1 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=Continuous --kl_weight=0.1 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --subpolicy_model=Experiment_Logs/S55_Pretrain_ContZ_NewKL_klwt0pt1_rightip/saved_models/Model_epoch49'

python cluster_run.py --name=C13_loadS55_subpolrat0pt1_latpolrat0pt1_kl0pt01 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=5 --name=C13_loadS55_subpolrat0pt1_latpolrat0pt1_kl0pt01 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.1 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=Continuous --kl_weight=0.01 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --subpolicy_model=Experiment_Logs/S55_Pretrain_ContZ_NewKL_klwt0pt1_rightip/saved_models/Model_epoch38'

python cluster_run.py --name=C14_loadS55_subpolrat0pt1_latpolrat0pt01_kl0pt01 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=5 --name=C14_loadS55_subpolrat0pt1_latpolrat0pt01_kl0pt01 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=Continuous --kl_weight=0.01 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --subpolicy_model=Experiment_Logs/S55_Pretrain_ContZ_NewKL_klwt0pt1_rightip/saved_models/Model_epoch38'

python cluster_run.py --name=C15_loadS55_subpolrat0pt1_latpolrat0pt001_kl0pt01 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=5 --name=C15_loadS55_subpolrat0pt1_latpolrat0pt001_kl0pt01 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.001 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=Continuous --kl_weight=0.01 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --subpolicy_model=Experiment_Logs/S55_Pretrain_ContZ_NewKL_klwt0pt1_rightip/saved_models/Model_epoch38'

# Debug
python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=5 --name=C1star_debug --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=Continuous --kl_weight=0.1 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --subpolicy_model=Experiment_Logs/S55_Pretrain_ContZ_NewKL_klwt0pt1_rightip/saved_models/Model_epoch49 --debug=1

# Running with downweighted variational loss, and documenting KL and prior loglikelihoods.
python cluster_run.py --name=C16_loadS55_subpolrat0pt1_latpolrat0pt01_kl0pt1_vlwt0pt1 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=5 --name=C16_loadS55_subpolrat0pt1_latpolrat0pt01_kl0pt1_vlwt0pt1 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=Continuous --kl_weight=0.1 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.1 --subpolicy_model=Experiment_Logs/S55_Pretrain_ContZ_NewKL_klwt0pt1_rightip/saved_models/Model_epoch49'
# Eval at 1
python Master.py --train=0 --setting=learntsub --discrete_z=0 --traj_length=5 --name=C16_Eval --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=Continuous --kl_weight=0.1 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.1 --model=Experiment_Logs/C16_loadS55_subpolrat0pt1_latpolrat0pt01_kl0pt1_vlwt0pt1/saved_models/Model_epoch5

python cluster_run.py --name=C17_loadS55_subpolrat0pt1_latpolrat0pt01_kl0pt01_vlwt0pt1 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=5 --name=C17_loadS55_subpolrat0pt1_latpolrat0pt01_kl0pt01_vlwt0pt1 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=Continuous --kl_weight=0.01 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.1 --subpolicy_model=Experiment_Logs/S55_Pretrain_ContZ_NewKL_klwt0pt1_rightip/saved_models/Model_epoch49'

python cluster_run.py --name=C18_loadS55_subpolrat0pt1_latpolrat0pt01_kl0pt1_vlwt0pt01 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=5 --name=C18_loadS55_subpolrat0pt1_latpolrat0pt01_kl0pt1_vlwt0pt01 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=Continuous --kl_weight=0.1 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.01 --subpolicy_model=Experiment_Logs/S55_Pretrain_ContZ_NewKL_klwt0pt1_rightip/saved_models/Model_epoch49'

python cluster_run.py --name=C19_loadS55_subpolrat0pt1_latpolrat0pt01_kl0pt01_vlwt0pt01 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=5 --name=C19_loadS55_subpolrat0pt1_latpolrat0pt01_kl0pt01_vlwt0pt01 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=Continuous --kl_weight=0.01 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.01 --subpolicy_model=Experiment_Logs/S55_Pretrain_ContZ_NewKL_klwt0pt1_rightip/saved_models/Model_epoch49'
# Eval
python Master.py --train=0 --setting=learntsub --discrete_z=0 --traj_length=5 --name=C19_Eval --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=Continuous --kl_weight=0.01 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.01 --subpolicy_model=Experiment_Logs/S55_Pretrain_ContZ_NewKL_klwt0pt1_rightip/saved_models/Model_epoch49 --model=Experiment_Logs/C19_loadS55_subpolrat0pt1_latpolrat0pt01_kl0pt01_vlwt0pt01/saved_models/Model_epoch14

# Run on longer trajectories: 
python cluster_run.py --name=C20_loadS55_subpolrat0pt1_latpolrat0pt01_kl0pt1_vlwt0pt1 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=10 --name=C20_loadS55_subpolrat0pt1_latpolrat0pt01_kl0pt1_vlwt0pt1 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=Continuous --kl_weight=0.1 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.1 --subpolicy_model=Experiment_Logs/S55_Pretrain_ContZ_NewKL_klwt0pt1_rightip/saved_models/Model_epoch49'

python cluster_run.py --name=C21_loadS55_subpolrat0pt1_latpolrat0pt01_kl0pt01_vlwt0pt1 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=10 --name=C21_loadS55_subpolrat0pt1_latpolrat0pt01_kl0pt01_vlwt0pt1 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=Continuous --kl_weight=0.01 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.1 --subpolicy_model=Experiment_Logs/S55_Pretrain_ContZ_NewKL_klwt0pt1_rightip/saved_models/Model_epoch49'

python cluster_run.py --name=C22_loadS55_subpolrat0pt1_latpolrat0pt01_kl0pt1_vlwt0pt01 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=10 --name=C22_loadS55_subpolrat0pt1_latpolrat0pt01_kl0pt1_vlwt0pt01 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=Continuous --kl_weight=0.1 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.01 --subpolicy_model=Experiment_Logs/S55_Pretrain_ContZ_NewKL_klwt0pt1_rightip/saved_models/Model_epoch49'

python cluster_run.py --name=C23_loadS55_subpolrat0pt1_latpolrat0pt01_kl0pt01_vlwt0pt01 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=10 --name=C23_loadS55_subpolrat0pt1_latpolrat0pt01_kl0pt01_vlwt0pt01 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=Continuous --kl_weight=0.01 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.01 --subpolicy_model=Experiment_Logs/S55_Pretrain_ContZ_NewKL_klwt0pt1_rightip/saved_models/Model_epoch49'

# Debug
python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=10 --name=Clen10_debug --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=Continuous --kl_weight=0.01 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.01 --subpolicy_model=Experiment_Logs/S55_Pretrain_ContZ_NewKL_klwt0pt1_rightip/saved_models/Model_epoch49 --debug=1

###############################################################
############### Continuous trials on CNZ data #################
###############################################################
python cluster_run.py --name=C24_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt1_vlwt0pt1 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=10 --name=C24_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt1_vlwt0pt1 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=ContinuousNonZero --kl_weight=0.1 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.1 --subpolicy_model=Experiment_Logs/S90_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch5'

python cluster_run.py --name=C25_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt01_vlwt0pt1 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=10 --name=C25_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt01_vlwt0pt1 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=ContinuousNonZero --kl_weight=0.01 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.1 --subpolicy_model=Experiment_Logs/S90_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch5'

python cluster_run.py --name=C26_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt1_vlwt0pt01 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=10 --name=C26_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt1_vlwt0pt01 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=ContinuousNonZero --kl_weight=0.1 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.01 --subpolicy_model=Experiment_Logs/S90_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch5'

python cluster_run.py --name=C27_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt01_vlwt0pt01 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=10 --name=C27_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt01_vlwt0pt01 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=ContinuousNonZero --kl_weight=0.01 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.01 --subpolicy_model=Experiment_Logs/S90_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch5'

python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=10 --name=CNZJ_debug --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=ContinuousNonZero --kl_weight=0.01 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.01 --subpolicy_model=Experiment_Logs/S90_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch5 --debug=1

python cluster_run.py --name=C28_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt1_vlwt0pt1 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=5 --name=C28_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt1_vlwt0pt1 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=ContinuousNonZero --kl_weight=0.1 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.1 --subpolicy_model=Experiment_Logs/S90_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch5'

python cluster_run.py --name=C29_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt01_vlwt0pt1 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=5 --name=C29_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt01_vlwt0pt1 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=ContinuousNonZero --kl_weight=0.01 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.1 --subpolicy_model=Experiment_Logs/S90_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch5'

python cluster_run.py --name=C30_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt1_vlwt0pt01 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=5 --name=C30_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt1_vlwt0pt01 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=ContinuousNonZero --kl_weight=0.1 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.01 --subpolicy_model=Experiment_Logs/S90_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch5'

python cluster_run.py --name=C31_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt01_vlwt0pt01 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=5 --name=C31_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt01_vlwt0pt01 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=ContinuousNonZero --kl_weight=0.01 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.01 --subpolicy_model=Experiment_Logs/S90_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch5'

# Running C24-C31 again with clamped likelihoods. 
python cluster_run.py --name=C32_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt1_vlwt0pt1 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=10 --name=C32_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt1_vlwt0pt1 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=ContinuousNonZero --kl_weight=0.1 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.1 --subpolicy_model=Experiment_Logs/S90_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch5 --subpolicy_clamp_value=-20'

python cluster_run.py --name=C33_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt01_vlwt0pt1 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=10 --name=C33_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt01_vlwt0pt1 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=ContinuousNonZero --kl_weight=0.01 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.1 --subpolicy_model=Experiment_Logs/S90_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch5 --subpolicy_clamp_value=-20'

python cluster_run.py --name=C34_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt1_vlwt0pt01 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=10 --name=C34_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt1_vlwt0pt01 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=ContinuousNonZero --kl_weight=0.1 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.01 --subpolicy_model=Experiment_Logs/S90_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch5 --subpolicy_clamp_value=-20'

python cluster_run.py --name=C35_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt01_vlwt0pt01 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=10 --name=C35_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt01_vlwt0pt01 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=ContinuousNonZero --kl_weight=0.01 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.01 --subpolicy_model=Experiment_Logs/S90_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch5 --subpolicy_clamp_value=-20'

python cluster_run.py --name=C36_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt1_vlwt0pt1 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=5 --name=C36_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt1_vlwt0pt1 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=ContinuousNonZero --kl_weight=0.1 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.1 --subpolicy_model=Experiment_Logs/S90_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch5 --subpolicy_clamp_value=-20'

python cluster_run.py --name=C37_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt01_vlwt0pt1 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=5 --name=C37_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt01_vlwt0pt1 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=ContinuousNonZero --kl_weight=0.01 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.1 --subpolicy_model=Experiment_Logs/S90_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch5 --subpolicy_clamp_value=-20'

python cluster_run.py --name=C38_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt1_vlwt0pt01 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=5 --name=C38_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt1_vlwt0pt01 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=ContinuousNonZero --kl_weight=0.1 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.01 --subpolicy_model=Experiment_Logs/S90_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch5 --subpolicy_clamp_value=-20'

python cluster_run.py --name=C39_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt01_vlwt0pt01 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=5 --name=C39_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt01_vlwt0pt01 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=ContinuousNonZero --kl_weight=0.01 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.01 --subpolicy_model=Experiment_Logs/S90_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch5 --subpolicy_clamp_value=-20'
python Master.py --train=0 --setting=learntsub --discrete_z=0 --traj_length=5 --name=C39_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt01_vlwt0pt01 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=ContinuousNonZero --kl_weight=0.01 --epsilon_from=0.5 --epsilon_to=0.3 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.01  --subpolicy_clamp_value=-20 --model=Experiment_Logs/C39_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt01_vlwt0pt01/saved_models/Model_epoch35 --subpolicy_model=Experiment_Logs/S90_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch5

# Running C32-C39 runs with alternate gradient implementation.
python cluster_run.py --name=C40 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=10 --name=C40_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt1_vlwt0pt1 --ent_weight=0. --entropy=0 --var_entropy=0 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=ContinuousNonZero --kl_weight=0.1 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.1 --subpolicy_model=Experiment_Logs/S90_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch5 --subpolicy_clamp_value=-20'

python cluster_run.py --name=C41 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=10 --name=C41_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt01_vlwt0pt1 --ent_weight=0. --entropy=0 --var_entropy=0 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=ContinuousNonZero --kl_weight=0.01 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.1 --subpolicy_model=Experiment_Logs/S90_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch5 --subpolicy_clamp_value=-20'

python cluster_run.py --name=C42 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=10 --name=C42_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt1_vlwt0pt01 --ent_weight=0. --entropy=0 --var_entropy=0 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=ContinuousNonZero --kl_weight=0.1 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.01 --subpolicy_model=Experiment_Logs/S90_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch5 --subpolicy_clamp_value=-20'

python cluster_run.py --name=C43 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=10 --name=C43_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt01_vlwt0pt01 --ent_weight=0. --entropy=0 --var_entropy=0 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=ContinuousNonZero --kl_weight=0.01 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.01 --subpolicy_model=Experiment_Logs/S90_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch5 --subpolicy_clamp_value=-20'

python cluster_run.py --name=C44 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=5 --name=C44_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt1_vlwt0pt1 --ent_weight=0. --entropy=0 --var_entropy=0 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=ContinuousNonZero --kl_weight=0.1 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.1 --subpolicy_model=Experiment_Logs/S90_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch5 --subpolicy_clamp_value=-20'
# Eval at 13
python Master.py --train=0 --setting=learntsub --discrete_z=0 --traj_length=5 --name=C44_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt1_vlwt0pt1 --ent_weight=0. --entropy=0 --var_entropy=0 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=ContinuousNonZero --kl_weight=0.1 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.1 --subpolicy_model=Experiment_Logs/S90_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch5 --subpolicy_clamp_value=-20 --model=Experiment_Logs/C44_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt1_vlwt0pt1/saved_models/Model_epoch10

python cluster_run.py --name=C45 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=5 --name=C45_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt01_vlwt0pt1 --ent_weight=0. --entropy=0 --var_entropy=0 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=ContinuousNonZero --kl_weight=0.01 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.1 --subpolicy_model=Experiment_Logs/S90_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch5 --subpolicy_clamp_value=-20'

python cluster_run.py --name=C46 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=5 --name=C46_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt1_vlwt0pt01 --ent_weight=0. --entropy=0 --var_entropy=0 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=ContinuousNonZero --kl_weight=0.1 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.01 --subpolicy_model=Experiment_Logs/S90_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch5 --subpolicy_clamp_value=-20'

python cluster_run.py --name=C47 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=5 --name=C47_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt01_vlwt0pt01 --ent_weight=0. --entropy=0 --var_entropy=0 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=ContinuousNonZero --kl_weight=0.01 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.01 --subpolicy_model=Experiment_Logs/S90_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch5 --subpolicy_clamp_value=-20'

# Debuggity.
python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=5 --name=C4debug --ent_weight=0. --entropy=0 --var_entropy=0 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=ContinuousNonZero --kl_weight=0.01 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.01 --subpolicy_model=Experiment_Logs/S90_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch5 --subpolicy_clamp_value=-20

# Run with var entropy original.
# Run with variational entropy = 1
python cluster_run.py --name=C50 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=10 --name=C50_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt1_vlwt0pt1 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=ContinuousNonZero --kl_weight=0.1 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.1 --subpolicy_model=Experiment_Logs/S90_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch5 --subpolicy_clamp_value=-20'

python cluster_run.py --name=C51 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=10 --name=C51_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt01_vlwt0pt1 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=ContinuousNonZero --kl_weight=0.01 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.1 --subpolicy_model=Experiment_Logs/S90_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch5 --subpolicy_clamp_value=-20'

python cluster_run.py --name=C52 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=10 --name=C52_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt1_vlwt0pt01 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=ContinuousNonZero --kl_weight=0.1 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.01 --subpolicy_model=Experiment_Logs/S90_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch5 --subpolicy_clamp_value=-20'

python cluster_run.py --name=C53 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=10 --name=C53_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt01_vlwt0pt01 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=ContinuousNonZero --kl_weight=0.01 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.01 --subpolicy_model=Experiment_Logs/S90_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch5 --subpolicy_clamp_value=-20'

python cluster_run.py --name=C54 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=5 --name=C54_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt1_vlwt0pt1 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=ContinuousNonZero --kl_weight=0.1 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.1 --subpolicy_model=Experiment_Logs/S90_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch5 --subpolicy_clamp_value=-20'

python cluster_run.py --name=C55 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=5 --name=C55_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt01_vlwt0pt1 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=ContinuousNonZero --kl_weight=0.01 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.1 --subpolicy_model=Experiment_Logs/S90_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch5 --subpolicy_clamp_value=-20'

python cluster_run.py --name=C56 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=5 --name=C56_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt1_vlwt0pt01 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=ContinuousNonZero --kl_weight=0.1 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.01 --subpolicy_model=Experiment_Logs/S90_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch5 --subpolicy_clamp_value=-20'

python cluster_run.py --name=C57 --cmd='python Master.py --train=1 --setting=learntsub --discrete_z=0 --traj_length=5 --name=C57_loadS90_subpolrat0pt1_latpolrat0pt01_kl0pt01_vlwt0pt01 --ent_weight=0. --entropy=0 --var_entropy=1 --subpolicy_ratio=0.1 --latentpolicy_ratio=0.01 --b_ex_bias=0.5 --b_probability_factor=0.01 --min_variance_bias=0.05 --data=ContinuousNonZero --kl_weight=0.01 --epsilon_from=0.5 --epsilon_to=0.1 --epsilon_over=20 --fix_subpolicy=1 --new_z_selection=1 --var_loss_weight=0.01 --subpolicy_model=Experiment_Logs/S90_Pretrain_ContZ_NewKL_klwt0pt01_rightip/saved_models/Model_epoch5 --subpolicy_clamp_value=-20'



###############################################################
################### MIME PRETRAIN TRIALS ######################
###############################################################

python cluster_run.py --name=M1_Pretrain_ContZ_NewKL_klwt0pt01_notanh --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M1_Pretrain_ContZ_NewKL_klwt0pt01_notanh --entropy=0 --data=MIME --kl_weight=0.01 --discrete_z=0 --mean_nonlinearity=0 --z_dimensions=32'
# Eval
python Master.py --train=0 --setting=pretrain_sub --name=M1_Pretrain_ContZ_NewKL_klwt0pt01_notanh --entropy=0 --data=MIME --kl_weight=0.01 --discrete_z=0 --mean_nonlinearity=0 --z_dimensions=32 --model=Experiment_Logs/M1_Pretrain_ContZ_NewKL_klwt0pt01_notanh/saved_models/Model_epoch8 --debug=1

python cluster_run.py --name=M2_Pretrain_ContZ_NewKL_klwt0pt1_notanh --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M2_Pretrain_ContZ_NewKL_klwt0pt1_notanh --entropy=0 --data=MIME --kl_weight=0.1 --discrete_z=0 --mean_nonlinearity=0 --z_dimensions=32'
# Eval
python Master.py --train=0 --setting=pretrain_sub --name=M2_Pretrain_ContZ_NewKL_klwt0pt1_notanh --entropy=0 --data=MIME --kl_weight=0.1 --discrete_z=0 --mean_nonlinearity=0 --z_dimensions=32 --model=Experiment_Logs/M2_Pretrain_ContZ_NewKL_klwt0pt1_notanh/saved_models/Model_epoch49

python cluster_run.py --name=M3_Pretrain_ContZ_NewKL_klwt0pt5_notanh --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M3_Pretrain_ContZ_NewKL_klwt0pt5_notanh --entropy=0 --data=MIME --kl_weight=0.5 --discrete_z=0 --mean_nonlinearity=0 --z_dimensions=32'
# Eval
python Master.py --train=0 --setting=pretrain_sub --name=M3_Pretrain_ContZ_NewKL_klwt0pt5_notanh --entropy=0 --data=MIME --kl_weight=0.5 --discrete_z=0 --mean_nonlinearity=0 --z_dimensions=32 --model=Experiment_Logs/M3_Pretrain_ContZ_NewKL_klwt0pt5_notanh/saved_models/Model_epoch49

python cluster_run.py --name=M4_Pretrain_ContZ_NewKL_klwt1pt0_notanh --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M4_Pretrain_ContZ_NewKL_klwt1pt0_notanh --entropy=0 --data=MIME --kl_weight=1. --discrete_z=0 --mean_nonlinearity=0 --z_dimensions=32'
# Eval
python Master.py --train=0 --setting=pretrain_sub --name=M4_Pretrain_ContZ_NewKL_klwt1pt0_notanh --entropy=0 --data=MIME --kl_weight=1. --discrete_z=0 --mean_nonlinearity=0 --z_dimensions=32 --model=Experiment_Logs/M4_Pretrain_ContZ_NewKL_klwt1pt0_notanh/saved_models/Model_epoch49
python Master.py --train=0 --setting=pretrain_sub --name=M4_Pretrain_ContZ_NewKL_klwt1pt0_notanh --entropy=0 --data=MIME --kl_weight=1. --discrete_z=0 --mean_nonlinearity=0 --z_dimensions=32 --model=Experiment_Logs/M4_Pretrain_ContZ_NewKL_klwt1pt0_notanh/saved_models/Model_epoch49 --debug=1

# Rerun for more epochs with loading.
python cluster_run.py --partition=priority --name=M5_Pretrain_ContZ_NewKL_klwt0pt01_notanh --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M5_Pretrain_ContZ_NewKL_klwt0pt01_notanh --entropy=0 --data=MIME --kl_weight=0.01 --discrete_z=0 --mean_nonlinearity=0 --z_dimensions=32 --model=Experiment_Logs/M1_Pretrain_ContZ_NewKL_klwt0pt01_notanh/saved_models/Model_epoch43'

python cluster_run.py --partition=priority --name=M6_Pretrain_ContZ_NewKL_klwt0pt1_notanh --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M6_Pretrain_ContZ_NewKL_klwt0pt1_notanh --entropy=0 --data=MIME --kl_weight=0.1 --discrete_z=0 --mean_nonlinearity=0 --z_dimensions=32 --model=Experiment_Logs/M2_Pretrain_ContZ_NewKL_klwt0pt1_notanh/saved_models/Model_epoch49'

python cluster_run.py --partition=priority --name=M7_Pretrain_ContZ_NewKL_klwt0pt5_notanh --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M7_Pretrain_ContZ_NewKL_klwt0pt5_notanh --entropy=0 --data=MIME --kl_weight=0.5 --discrete_z=0 --mean_nonlinearity=0 --z_dimensions=32 --model=Experiment_Logs/M3_Pretrain_ContZ_NewKL_klwt0pt5_notanh/saved_models/Model_epoch49'
python Master.py --train=0 --setting=pretrain_sub --name=M7_Pretrain_ContZ_NewKL_klwt0pt5_notanh --entropy=0 --data=MIME --kl_weight=0.5 --discrete_z=0 --mean_nonlinearity=0 --z_dimensions=32 --model=Experiment_Logs/M7_Pretrain_ContZ_NewKL_klwt0pt5_notanh/saved_models/Model_epoch8

python cluster_run.py --partition=priority --name=M8_Pretrain_ContZ_NewKL_klwt1pt0_notanh --cmd='python Master.py --train=1 --setting=pretrain_sub --name=M8_Pretrain_ContZ_NewKL_klwt1pt0_notanh --entropy=0 --data=MIME --kl_weight=1. --discrete_z=0 --mean_nonlinearity=0 --z_dimensions=32 --model=Experiment_Logs/M4_Pretrain_ContZ_NewKL_klwt1pt0_notanh/saved_models/Model_epoch49'
