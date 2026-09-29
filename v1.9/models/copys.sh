cd /home/lus04/trainee4/radiation_source_localization/v1.6/models/xgboost
sbatch /home/lus04/trainee4/radiation_source_localization/v1.6/models/xgboost/xg.sh

cd /home/lus04/trainee4/radiation_source_localization/v1.6/models/xg_tuned
sbatch /home/lus04/trainee4/radiation_source_localization/v1.6/models/xg_tuned/xg.sh

cd /home/lus04/trainee4/radiation_source_localization/v1.6/models/decision_tree
sbatch /home/lus04/trainee4/radiation_source_localization/v1.6/models/decision_tree/dt.sh

cd /home/lus04/trainee4/radiation_source_localization/v1.6/models/dt_tuned
sbatch /home/lus04/trainee4/radiation_source_localization/v1.6/models/dt_tuned/dt.sh

cd /home/lus04/trainee4/radiation_source_localization/v1.6/models/rf_tuned
sbatch /home/lus04/trainee4/radiation_source_localization/v1.6/models/rf_tuned/rf.sh

cd /home/lus04/trainee4/radiation_source_localization/v1.6/models/random_forest
sbatch /home/lus04/trainee4/radiation_source_localization/v1.6/models/random_forest/rf.sh

cd /home/lus04/trainee4/radiation_source_localization/v1.6/models/svr
sbatch /home/lus04/trainee4/radiation_source_localization/v1.6/models/svr/svr.sh