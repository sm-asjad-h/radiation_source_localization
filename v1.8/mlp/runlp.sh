export CUDA_VISIBLE_DEVICES=1
python trainmlp.py
echo "training complete for mlp"
python testmlp.py
