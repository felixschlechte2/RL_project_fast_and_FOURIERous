## SAC-SimBa on Hockey Environment

![SAC-SimBa vs strong opponent](../../assets/SimBa_vs_strong_opp.gif)  
*SAC-SimBa (red) vs strong opponent (blue)*

Here you can see the agent from checkpoint SAC_23600_auto from [checkpoints](./checkpoints/). 
In [run_info](./run_info/) you can find notes, the log file and the arguments/hyperparamter for each training run.  

sac_NN_extra_init.py and NNs_extra_init.py are used for: SimBa_LRe-4.py, SimBa_auto.py, SimBa_ohne_ELO.py, SimBa_mlp_mlp_extra_init

sac_NN.py and NNs.py are used for: SimBa_mlp_mlp.py, SimBa_mlp_residual.py, SimBa_residual_mlp.py, SimBa_residual_mlp.py, SimBa_residual_residual.py

sac.py and model.py are used for the opponents whose checkpoints are in enemies
