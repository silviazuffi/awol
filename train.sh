export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniconda3/lib

######
# These are the scripts to train also with images, and are provided as examples
#python -m awol.code.experiments.main --object='tree' --name='tree_realnvp_mask_w_image' --flow_type='realnvp' --train_mask=True --num_epochs=1500 --compress_params=True --normalize=True --map_categorical=True --use_images=True 
#python -m awol.code.experiments.main --object='tree' --name='tree_realnvp_mask_w_image' --flow_type='realnvp' --train_mask=True --num_epochs=3000 --compress_params=True --normalize=True --map_categorical=True --use_images=True --num_pretrain_epochs=1500 --learning_rate=1e-5

# Train the animal model from text
#python -m awol.code.experiments.main --object='animal' --name='animal_realnvp_mask' --flow_type='realnvp' --train_mask=True --num_epochs=2500 --save_epoch_freq=10000 --learning_rate=0.5e-4
#python -m awol.code.experiments.main --object='animal' --name='animal_realnvp_mask' --flow_type='realnvp' --train_mask=True --num_epochs=6000 --save_epoch_freq=10000 --learning_rate=1e-5 --num_pretrain_epochs=2500

# Train the tree model from text
python -m awol.code.experiments.main --object='tree' --name='tree_realnvp_mask' --flow_type='realnvp' --train_mask=True --num_epochs=6000 --compress_params=True --normalize=True --map_categorical=True


