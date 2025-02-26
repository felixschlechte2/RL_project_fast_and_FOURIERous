## SAC-RNN on Hockey environment

![QR-SAC vs strong opponent](../../assets/sac_rnn_vs_strong_opp.gif)  
*SAC-RNN (red) vs strong opponent (blue)*

Here you can see the agent from checkpoint checkpoint_hockey_vs_4_all_rew_ep_62000_sac_rnn2_run2 from [checkpoints](./checkpoints/) that also competed in the tournament as our team agent fast_and_FOURIERous. In [run_info](./run_info/) you can find notes, the log file and the arguments/hyperparamter for each training run.  

remarks: 
- in folder [runs](./runs/), the tensorboard statistics about the runs cannot be provided due to uploading limits, but the most important information can bo found in the other files. 
-  in [test.py](./test.py), we compare QR-SAC and SAC-RNN
