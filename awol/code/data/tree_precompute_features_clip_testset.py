'''


    Author: Silvia Zuffi


'''
from __future__ import absolute_import

import torch
from glob import glob
from os import path, listdir
from os.path import exists, join
import numpy as np

import open_clip

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

tree_species = ['Ginkgo', 'Coconut', 'Cedar of Lebanon', 'Maritime Pine', 'Fig', 'Cocoa', 'Bigleaf Maple', 'Deodar Cedar', 'Eucalyptus', 'Tulip', 'Oak', 'Banyan', 'American Elm', 'Magnolia', 'Acer', 'Coast Redwood', 'Sequoia', 'Western Red Cedar', 'European Larch', 'Scots Pine', 'White Spruce', 'Italian Cypress']

#tree_species = ['Beech', 'Corkscrew Hazel', 'Maple', 'Oak', 'Pine', 'Tulip', 'Walnut', 'Willow', 'Lime', 'Acacia', 'Baobab', 'Fir']


#tree_species = ['Small Gingko', 'Young Gingko', 'Tall Gingko', 'Old Gingko', 'Small Acer', 'Young Acer', 'Tall Acer', 'Old Acer',
#        'Small Palm', 'Young Palm', 'Tall Palm', 'Old Palm', 'Small Cypress', 'Young Cypress', 'Tall Cypress', 'Old Cypress',
#        'Small Weeping Willow', 'Young Weeping Willow', 'Tall Weeping Willow', 'Old Weeping Willow']

n_species = len(tree_species)

clip_features = torch.zeros(n_species, 512)

if True:
    for i,tree in enumerate(tree_species):
        with torch.no_grad(), torch.cuda.amp.autocast():
            desc = 'A photo of a ' + tree + ' tree.' 
            text = tokenizer(desc)
            text_features = model.encode_text(text)
            clip_features[i,:] = text_features[0,:]

    torch.save(clip_features,'tree_clip_features_testset.pt')
    #torch.save(clip_features,'tree_clip_features_testset_deeptree.pt')
    #torch.save(clip_features,'tree_clip_features_testset_interpolation.pt')


