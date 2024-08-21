'''
    Author: Silvia Zuffi
    Note: you need to install the mesh library to visualize the meshes or save the objs (https://github.com/MPI-IS/mesh), otherwise use another mesh package

'''
from smpl_webuser.serialization import load_model
import os
import pickle as pkl
import numpy as np
from psbody.mesh.meshviewer import MeshViewer, MeshViewers
from psbody.mesh.mesh import Mesh

color =  np.array([0.5, 0.7, 0.8])

dog_breeds = ['Bernese Mountain Dog', 'Corgi', 'Chow-chow', 'Pomeranian', 'Samoyed', 'Great Pyrenees', 'Newfoundland', 'Leonberger', 'Pug', 'Basenji', 'Affenpinscher', 'Siberian Husky',
        'Alaskan Malamute', 'Eskimo Dog', 'Saint Bernard', 'Great Dane', 'French Bulldog', 'Tibetan Mastiff', 'Bull Mastiff',
        'Boxer', 'Entlebucher Mountain', 'Appenzeller', 'Bearnese Mountain', 'Greater Swiss Mountain', 'Miniature Pinscher',
        'Doberman', 'German Shepherd', 'Rottweiler', 'Bouvier des Flandres', 'Border Collie', 'Collie', 'Shetland Sheepdog',
        'Old English Sheepdog', 'Komondor', 'Kelpie', 'Briard', 'Malinois', 'Schipperke', 'Kuvasz', 'Irish Water Spaniel',
        'Sussex Spaniel', 'Cocker Spaniel', 'Welsh Springer Spaniel', 'English Springer', 'Clumber', 'Brittany Spaniel',
        'Gordon Setter', 'Irish Setter', 'English Setter', 'Vizsla', 'German Short-haired Pointer', 'Chesapeake Bay Retriever',
        'Labrador Retriever', 'Golden Retriever', 'Curly-coated Retriever', 'Flat-coated Retriever', 'Lhasa Apso', 'West Highland White Terrier',
        'Soft-coated Wheaten Terrier', 'Silky Terrier', 'Tibetan Terrier', 'Scotch Terrier', 'Standard Schnauzer', 'Giant Schnauzer',
        'Miniature Schnauzer', 'Boston Bull', 'Dandie Dinmont', 'Australian Terrier', 'Cairn Terrier', 'Airedale Terrier',
        'Sealyham Terrier', 'Lakeland Terrier', 'Wire-haired Fox Terrier', 'Yorkshire Terrier', 'Norwich Terrier', 'Norfolk Terrier',
        'Irish Terrier', 'Kerry Blue Terrier', 'Border Terrier', 'Bedlington Terrier', 'American Staffordshire Terrier',
        'Staffordshire Bull Terrier', 'Weimaraner', 'Scottish Deerhound', 'Saluki', 'Otterhound', 'Norwegian Elkhound',
        'Ibizan Hound', 'Whippet', 'Italian Greyhound', 'Irish Wolfhound', 'Borzoi', 'Redbone Coonhound', 'English Foxhound',
        'Walker Hound', 'Black and Tan Coonhound', 'Bluetick Coonhound', 'Bloodhound', 'Beagle', 'Basset Hound', 'Afghan Hound',
        'Rhodesian Ridgeback', 'Toy Terrier', 'Papillon', 'Blenheim Spaniel', 'Shih-Tzu', 'Pekinese', 'Maltese', 'Japanese Spaniel',
        'Chihuahua']

animal_species_no_dogs = ['Donkey', 'Lynx', 'Goat', 'Alpaca', 'Dingo', 'Thylacine', 'Pig', 'Piglet', 'Rabbit', 'Polar Bear', 'Rhinoceros', 'Impala', 'Jaguar', 'Mink', 'Buffalo', 'Opossum', 'Antelope', 'Cougar', 'Coyote', 'Jackal', 'African Wild Dog', 'LLama', 'Okapi', 'Hare', 'Reindeer', 'Wild Boar', 'Panda', 'Racoon', 'Meerkat', 'Serval', 'Skunk', 'Aardvark', 'Chinchilla', 'Foal', 'Tiger Cub', 'Arabian Horse', 'Pony', 'Andalusian Horse', 'Bison', 'Cow', 'Cat', 'Panther']


ears_down = ['Bernese Mountain Dog', 'Saluki', 'Scottish Deerhound', 'Gordon Setter', 'Galgo', 'Poodle', 'Saint Bernard', 'Dachshund', 'Basset Hound', 'Anatolian Shepherd', 'Dalmatian',
        'Great Pyrenees', 'Newfoundland', 'Leonberger', 
        'Saint Bernard', 'Great Dane', 
        'Entlebucher Mountain', 'Appenzeller', 'Bearnese Mountain', 'Greater Swiss Mountain', 
        'Rottweiler', 'Borzoi', 'Pekingese',
        'Komondor', 'Kuvasz', 'Irish Water Spaniel',
        'Sussex Spaniel', 'Cocker Spaniel', 'Welsh Springer Spaniel', 'English Springer', 'Clumber', 'Brittany Spaniel',
        'Gordon Setter', 'Irish Setter', 'English Setter', 'Vizsla', 'German Short-haired Pointer', 'Chesapeake Bay Retriever',
        'Labrador Retriever', 'Golden Retriever', 'Curly-coated Retriever', 'Flat-coated Retriever', 'Lhasa Apso', 'West Highland White Terrier',
        'Weimaraner', 'Saluki', 'Otterhound', 'Tibetan Terrier', 'Dandie Dinmont', 'Old English Sheepdog', 
        'Ibizan Hound', 'Whippet', 'Italian Greyhound', 'Redbone Coonhound', 'English Foxhound',
        'Walker Hound', 'Black and Tan Coonhound', 'Bluetick Coonhound', 'Bloodhound', 'Beagle', 'Basset Hound', 'Afghan Hound',
        'Rhodesian Ridgeback', 'Blenheim Spaniel', 'Shih-Tzu', 'Pekinese', 'Maltese', 'Japanese Spaniel','Napoletan Mastiff']

tail_up = ['Boxer', 'Dobermann', 'Pug', 'Skunk', 'Poodle', 'Bulldog', 'LLama', 'Alpaca']

interpolation = ['Giant Schnauzer', 'Stadard Schnauzer', 'Toy Schnauzer', 'Miniature Schnauzer', 'Giant Poodle', 'Standard Poodle', 'Miniature Poodle', 'Toy Poodle',
        'fat Schnauzer', 'slim Schnauzer', 'fat Poodle', 'slim poodle', 'obese  Schnauzer', 'bony Schnauzer', 'obese Poodle', 'bony poodle',
        'cheetah cub', 'young cheetah', 'adult cheetah', 'old cheetah', 'obese cheetah', 'slim cat', 'obese cat', 'kitten', 'young cat',  'adult cat',
        'baby giraffe', 'young giraffe', 'adult giraffe', 'baby llama', 'young llama', 'adult llama', 'old llama',
        'old giraffe', 'old cat', 'baby wolf', 'young wolf', 'adult wolf', 'old wolf']



if __name__ == '__main__':

    animal_species = interpolation #dog_breeds + animal_species_no_dogs

    #file1 = "out_testset_animal_realnvp_mask_animal_clip_features_testset_dogs_and_animals.npy" 
    #file1 = "out_testset_animal_realnvp_mask_animal_clip_features_testset_interpolation.npy" 
    file1 = "out_testset_animals_eccv_animal_clip_features_testset_interpolation.npy" 

    model_filename = '../awol/data/animal/smal_plus.pkl'
    smpl = load_model(model_filename)

    mv = MeshViewer()
    mv.set_background_color(np.ones(3))
    n_samples=1
    err = 0
    j = 0
    for file in [file1]:
        print(file)
        Dir = 'results/'+file[:-4]+'/'
        print(Dir)
        betas = np.load(file) 
        nB1 = betas.shape[1]
        print(betas.shape)

        for i in range(betas.shape[0]):
            print(animal_species[i])
            if True:
                smpl.betas[:] = 0
                smpl.betas[:nB1] = betas[i*n_samples+j,:]
                smpl.trans[0] = 0
                if animal_species[i] in ears_down:
                    smpl.pose[3*34] = 1.8 
                    smpl.pose[3*34+1] = 1
                    smpl.pose[3*33] = -1.8 
                    smpl.pose[3*33+1] = 1
                else:
                    smpl.pose[:] = 0
                if animal_species[i] in tail_up:
                    smpl.pose[25*3+1] = 0
                else:
                    smpl.pose[25*3+1] = -1.2
                M = Mesh(v=smpl.r, f=smpl.f).set_vertex_colors(color)
                mv.set_static_meshes([M]) 
                M.write_obj(filename=Dir + animal_species[i]+'_'+str(j)+'.obj')
                import pdb; pdb.set_trace()
