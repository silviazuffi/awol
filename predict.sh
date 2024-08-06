export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniconda3/lib


python -m awol.code.experiments.predict --object='tree' --name='tree_realnvp_mask' --flow_type='realnvp' --train_mask=True --num_epochs=6000 --compress_params=True --normalize=True --map_categorical=True --testset=True 

#python -m awol.code.experiments.predict --object='animal' --name='animal_realnvp_mask' --flow_type='realnvp' --train_mask=True --num_epochs=6000 --testset=True 
