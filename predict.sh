export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniconda3/lib


python -m awol.code.experiments.predict --object='tree' --name='submission_tree_ns_mask' --flow_type='realnvp' --train_mask=True --num_epochs=6000 --compress_params=True --normalize=True --map_categorical=True --testset=True 

#python -m dogs_and_trees.code.experiments.predict --object='tree' --name='submission_tree_ns_mask_w_image' --flow_type='realnvp' --train_mask=True --num_epochs=3000 --compress_params=True --normalize=True --map_categorical=True --testset=True --images=True


#python -m awol.code.experiments.predict --object='animal' --name='submission_animal_realnvp_mask' --flow_type='realnvp' --train_mask=True --num_epochs=6000 --testset=True 


#python -m awol.code.experiments.predict_from_images --object='animal' --name='submission_animal_realnvp_mask_w_images' --flow_type='realnvp' --train_mask=True --num_epochs=3000 --img_dir='/home/szuffi/projects/trees_and_dogs/dogs_and_trees/cats_images/' --extension='jpeg'
