# A minimalistic implementation of FlowMatching for Fluid Flows

Run simple examples: 

```
export CUDA_VISIBLE_DEVICES=0;  python train_ae.py --run-name simpleflow_ae --dataset simpleflow --ae_option ae --ae_epochs 2000 --snapshots-per-sample 25 --ae_lr_scheduler cosine --ae_learning_rate 0.001 --path_to_ae_checkpoints your-path

export CUDA_VISIBLE_DEVICES=0; python train.py --run-name simpleflow_ours_sigma0.01_samplingsteps10_euler_separate_sigmasam0.0 --dataset simpleflow --train_option separate --probpath_option ours  --epochs 2000 --sampling_steps 10 --sigma 0.01 --sigma_sam 0.0 --solver euler --snapshots-per-sample 25 --snapshots-to-generate 20 --path_to_ae_checkpoints your-path --path_to_results your-path

```