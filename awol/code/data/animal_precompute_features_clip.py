'''


    Author: Silvia Zuffi


'''
from __future__ import absolute_import

import torch
from glob import glob
from os import path, listdir
from os.path import exists, join
import numpy as np

from animals_prototypes_data import animal_species, animal_name 
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')


n_species = len(animal_species)

clip_features = torch.zeros(n_species, 512)

if True:
    for i,animal in enumerate(animal_species):
        with torch.no_grad(), torch.cuda.amp.autocast():
            desc = 'A photo of a ' + animal_name[animal] 
            text = tokenizer(desc)
            text_features = model.encode_text(text)
            clip_features[i,:] = text_features[0,:]

    torch.save(clip_features,'animal_clip_features.pt')


