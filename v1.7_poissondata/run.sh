export CUDA_VISIBLE_DEVICES=0
python data_pipeline.py
python gnn_optuna_tuning.py
