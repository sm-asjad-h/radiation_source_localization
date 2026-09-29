export CUDA_VISIBLE_DEVICES=2
python trainmlp.py
echo "training complete for mlp"
python testmlp.py
