How to train
-----------------
- After setup the environments, 
```bash
python train.py -c "path/to/configuration"
```
*In my case, Path to configuration is ./fullband_baseline/train.toml

Results in Train
-----------------
tensorboard --logdir ~/Experiments/Fullband_Baseline/train/logs


How to inference
-----------------
```
python inference.py \
  -C fullsubnet/inference.toml \
  -M /path/to/your/checkpoint_dir/best_model.tar \
  -O /path/to/your/enhancement/dir
```

*In my case
```
python inference.py \
  -C fullband_baseline/inference.toml \
  -M ~/Experiments/Fullband_Baseline/train/checkpoints/best_model.tar \
  -O ~/Experiments/Fullband_Baseline/test/enhance
```

Paper Results
-----------------
                Para(M)  Look Ahead(ms)       Without Reverb
                                        WB-PESQ  NB-PESQ   STOI   SI-SDR
Full-band Model   6.0         32         2.731    3.256   95.71   16.190
  FullSubNet      5.6         32         2.969    3.473   96.11   17.290

Our Results
Full-band Model    -           -         2.544     -      87.96   14.47


Logs
-----------------
=============== epoch 152 ===============
[0 seconds] Begin training...
Training: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 242/242 [33:52<00:00,  8.40s/it]
         Saving the model checkpoint of epoch 152...
[2032 seconds] Training is finished, and validation is in progress...
Validation: 10394it [16:43, 10.36it/s]
/home/daniel0413/.conda/envs/torch_171_daniel/lib/python3.8/site-packages/pypesq/__init__.py:53: UserWarning: Processing Error! return NaN
  warnings.warn('Processing Error! return NaN')
/home/daniel0413/.conda/envs/torch_171_daniel/lib/python3.8/site-packages/pypesq/__init__.py:53: UserWarning: Processing Error! return NaN
  warnings.warn('Processing Error! return NaN')
         Saving the model checkpoint of epoch 152...
         😃 Found a best score in the epoch 152, saving...
[3239 seconds] This epoch is finished.
=============== epoch 153 ===============
[0 seconds] Begin training...
Training: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 242/242 [34:08<00:00,  8.47s/it]
         Saving the model checkpoint of epoch 153...
[2048 seconds] Training is finished, and validation is in progress...
Validation: 10394it [16:40, 10.39it/s]
/home/daniel0413/.conda/envs/torch_171_daniel/lib/python3.8/site-packages/pypesq/__init__.py:53: UserWarning: Processing Error! return NaN
  warnings.warn('Processing Error! return NaN')
/home/daniel0413/.conda/envs/torch_171_daniel/lib/python3.8/site-packages/pypesq/__init__.py:53: UserWarning: Processing Error! return NaN
  warnings.warn('Processing Error! return NaN')
[3253 seconds] This epoch is finished.
=============== epoch 154 ===============
[0 seconds] Begin training...
Training: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 242/242 [34:00<00:00,  8.43s/it]
         Saving the model checkpoint of epoch 154...
[2040 seconds] Training is finished, and validation is in progress...
Validation: 10394it [16:35, 10.44it/s]
/home/daniel0413/.conda/envs/torch_171_daniel/lib/python3.8/site-packages/pypesq/__init__.py:53: UserWarning: Processing Error! return NaN
  warnings.warn('Processing Error! return NaN')
/home/daniel0413/.conda/envs/torch_171_daniel/lib/python3.8/site-packages/pypesq/__init__.py:53: UserWarning: Processing Error! return NaN
  warnings.warn('Processing Error! return NaN')
         Saving the model checkpoint of epoch 154...
         😃 Found a best score in the epoch 154, saving...
[3239 seconds] This epoch is finished.
=============== epoch 155 ===============
[0 seconds] Begin training...
Training: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 242/242 [33:52<00:00,  8.40s/it]
         Saving the model checkpoint of epoch 155...
[2032 seconds] Training is finished, and validation is in progress...
