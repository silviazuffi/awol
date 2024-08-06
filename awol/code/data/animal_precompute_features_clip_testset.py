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

animal_species = dog_breeds + animal_species_no_dogs

#animal_species = ['Giant Schnauzer', 'Stadard Schnauzer', 'Toy Schnauzer', 'Miniature Schnauzer', 'Giant Poodle', 'Standard Poodle', 'Miniature Poodle', 'Toy Poodle',
#        'fat Schnauzer', 'slim Schnauzer', 'fat Poodle', 'slim poodle', 'obese  Schnauzer', 'bony Schnauzer', 'obese Poodle', 'bony poodle',
#        'cheetah cub', 'young cheetah', 'adult cheetah', 'old cheetah', 'obese cheetah', 'slim cat', 'obese cat', 'kitten', 'young cat',  'adult cat',
#        'baby giraffe', 'young giraffe', 'adult giraffe', 'baby llama', 'young llama', 'adult llama', 'old llama',
#        'old giraffe', 'old cat', 'baby wolf', 'young wolf', 'adult wolf', 'old wolf']


n_species = len(animal_species)

clip_features = torch.zeros(n_species, 512)

if True:
    for i,animal in enumerate(animal_species):
        with torch.no_grad(), torch.cuda.amp.autocast():
            desc = 'A photo of a ' + animal 
            text = tokenizer(desc)
            text_features = model.encode_text(text)
            clip_features[i,:] = text_features[0,:]

    torch.save(clip_features,'animal_clip_features_testset_dogs_and_animals.pt')
    #torch.save(clip_features,'animal_clip_features_testset_interpolation.pt')


