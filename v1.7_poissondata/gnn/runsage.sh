export CUDA_VISIBLE_DEVICES=0
 python trainSage.py
echo "training complete for  sage gnn"
 python testsage.py
