export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniconda3/lib

#### TREES ####
#python -m awol.code.experiments.main --object='tree' --name='tree_ns_mask' --flow_type='realnvp' --train_mask=True --num_epochs=6000 --compress_params=True --normalize=True --map_categorical=True

#python -m awol.code.experiments.main --object='tree' --name='tree_ns_mask_w_image' --flow_type='realnvp' --train_mask=True --num_epochs=1500 --compress_params=True --normalize=True --map_categorical=True --use_images=True --save_epoch_freq=500
#python -m awol.code.experiments.main --object='tree' --name='tree_ns_mask_w_image' --flow_type='realnvp' --train_mask=True --num_epochs=3000 --compress_params=True --normalize=True --map_categorical=True --use_images=True --save_epoch_freq=500 --num_pretrain_epochs=1500 --learning_rate=1e-5

#### ANIMALS ####

#python -m awol.code.experiments.main --object='animal' --name='animal_realnvp_mask' --flow_type='realnvp' --train_mask=True --num_epochs=6000 --save_epoch_freq=500 --learning_rate=0.5e-4
#python -m awol.code.experiments.main --object='animal' --name='animal_realnvp_mask' --flow_type='realnvp' --train_mask=True --num_epochs=6000 --save_epoch_freq=500 --learning_rate=1e-5 --num_pretrain_epochs=3000

#python -m awol.code.experiments.main --object='animal' --name='animal_realnvp_mask_w_images' --flow_type='realnvp' --train_mask=True --num_epochs=6000 --save_epoch_freq=500 --learning_rate=0.5e-4 --use_images=True
#python -m awol.code.experiments.main --object='animal' --name='animal_realnvp_mask_w_images' --flow_type='realnvp' --train_mask=True --num_epochs=6000 --save_epoch_freq=500 --learning_rate=1e-5 --use_images=True --num_pretrain_epochs=1500
#python -m awol.code.experiments.main --object='animal' --name='animal_realnvp_mask_w_images' --flow_type='realnvp' --train_mask=True --num_epochs=6000 --save_epoch_freq=500 --learning_rate=0.5e-5 --use_images=True --num_pretrain_epochs=2500



