'''


    Author: Silvia Zuffi


'''


from __future__ import absolute_import
import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from absl import flags
from glob import glob
from os import path, listdir
from os.path import exists, join
import numpy as np
import pickle as pkl

flags.DEFINE_integer('n_data_workers', 4, 'Number of data loading workers')
flags.DEFINE_integer('num_samples', 10000, 'Number of samples')
flags.DEFINE_boolean('shuffle', True, '')
flags.DEFINE_string('animal_data_file', 'my_smpl_data_0791_to_root_n_skel_n_back_joints_leg_scaled_all.pkl', 'File that contains the shape variables')
flags.DEFINE_string('object_dir', '/home/szuffi/projects/awol/awol/data/', 'Data Directory')
flags.DEFINE_boolean('compress_params', False, 'Used by trees')
flags.DEFINE_boolean('normalize', False, 'Use by trees')
flags.DEFINE_boolean('use_images', False, '')
flags.DEFINE_boolean('reorder_tree_shape', False, '')
flags.DEFINE_boolean('map_categorical', False, 'Used by trees')
flags.DEFINE_integer('max_images', 20, '')


opts = flags.FLAGS

curr_path = path.dirname(path.abspath(__file__))
#from .dogs_prototypes_data import dog_breeds, dog_name, dog_betas_id 
from .trees_prototypes_data import tree_species, tree_name 
from .animals_prototypes_data import animal_species, animal_name, animal_dir_name

from PIL import Image
import open_clip


def tree_compress(params):
    non_zero_idx = [  0,   1,   3,   4,   5,   6,   7,   8,   9,  10,  11,  13,  14, 15,  17,  18,  19,  21,  22,  23,  29,  30,  31,  32,  33,  34, 35,  40,  41,  42,  43,  44,  45,  46,  48,  49,  50,  60,  61, 62,  63,  64,  65,  66,  68,  69,  81,  86,  88,  89,  90,  91, 92,  93, 94,  95,  98,  99, 100, 101, 102]
    if len(params.shape) == 1:
        return params[non_zero_idx]
    else:
        return params[:,non_zero_idx]

def tree_uncompress(params):
    non_zero_idx = [  0,   1,   3,   4,   5,   6,   7,   8,   9,  10,  11,  13,  14, 15,  17,  18,  19,  21,  22,  23,  29,  30,  31,  32,  33,  34, 35,  40,  41,  42,  43,  44,  45,  46,  48,  49,  50,  60,  61, 62,  63,  64,  65,  66,  68,  69,  81,  86,  88,  89,  90,  91, 92,  93, 94,  95,  98,  99, 100, 101, 102]
    fixed_values = [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  0. ,  0. , 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. , 10. , 10. , 10. , 10. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. , 0. ,  0. ,  0. ,  0. ,  1. ,  1. ,  1. , 0. ,  0. ,  0.5, 11. ]
    zero_idx = [  2,  12,  16,  20,  24,  25,  26,  27,  28,  36,  37,  38,  39, 47,  51,  52,  53,  54,  55,  56,  57,  58,  59,  67,  70,  71, 72,  73,  74,  75,  76,  77,  78,  79,  80,  82,  83,  84,  85, 87,  96,  97, 103, 104]

    if torch.is_tensor(params): 
        out = torch.zeros(params.shape[0], 105) 
        if len(params.shape) == 1:
            out[zero_idx] = torch.Tensor(fixed_values)
            out[non_zero_idx] = params
        else:
            out[:,zero_idx] = torch.Tensor(fixed_values)
            out[:,non_zero_idx] = params
    else:
        out = np.zeros((params.shape[0], 105)) 
        if len(params.shape) == 1:
            out[zero_idx] = fixed_values
            out[non_zero_idx] = params
        else:
            out[:,zero_idx] = fixed_values
            out[:,non_zero_idx] = params
    
    return out

def tree_reorder_shape(params, direct='to_net'):

    shape_blender2net = [0,5,4,3,2,6,3,1,8]
    shape_net2blender = [0,7,4,3,2,1,5,6,8]

    shape = int(np.round(params[0]))
    if direct == 'to_net':
        new_shape = shape_blender2net[shape]
    elif direct == 'to_blender':
        new_shape = shape_net2blender[shape]
    else:
        print('Unrecognized option')
        return params
    params[0] = new_shape
    return params

def encode(value, min_v, max_v):
    vect = np.zeros((max_v-min_v), dtype='int')
    idx = int(np.round((np.clip(value, min_v, max_v)))) - min_v
    vect[idx] = 1 
    return vect

def decode(vect, min_v):
    value = np.argmax(vect)
    return value + min_v

def map_categorical(params, direction='encode'):
    #Transform the categorical data into 0-1 encoding
    cat_data_idx = [0, 89, 93] # Corresponds to shape, leaf_shape, blossom_shape
    cat_data_size = [9, 10, 3]

    if direction == 'encode':
        new_size = len(params) + 19 # add the new data (9+10+3)-3
        new_params = np.zeros((new_size))
        new_params[9:57] = params[1:49]
        new_params[67:70] = params[50:53]
        new_params[73:80] = params[54:61]

        new_params[0:9] = encode(params[0], 0, 9)
        new_params[57:67] = encode(params[49], 1, 11)
        new_params[70:73] = encode(params[53], 1, 4)

    if direction == 'decode':
        new_size = len(params) - 19 
        new_params = np.zeros((new_size))
        new_params[1:49] = params[9:57]
        new_params[50:53] = params[67:70]
        new_params[54:61] = params[73:80]

        new_params[0] = decode(params[0:9], 0)
        new_params[49] = decode(params[57:67], 1)
        new_params[53] = decode(params[70:73], 1)

    return new_params
        


class ObjectDataset(Dataset):
    def __init__(self, opts):

        self.opts = opts

        self.data = []
        data_path = path.join(curr_path, '../../', 'data', opts.object)
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')

        if opts.object == 'animal':
            fname = join(opts.object_dir, opts.object, opts.animal_data_file)
            data = pkl.load(open(fname, 'rb'))
            betas = data['toys_betas']
            features = torch.load('./awol/code/data/animal_clip_features.pt')
            features /= features.norm(dim=-1, keepdim=True)
            for idx, animal in enumerate(animal_species):
                data = {'animal': animal, 'idx':idx, 'params': betas[idx,:opts.animal_emb_dim], 'text_features': features[idx,:]} 
                self.data += [data]

            if opts.use_images: 
                for k in range(opts.max_images):
                    for idx, animal in enumerate(animal_species):
                        print(animal)
                        # Read the images and generate an entry for each image
                        img_path = join(opts.object_dir, opts.object, 'training_set', animal_dir_name[animal])
                        I = sorted(glob(join(img_path, '*.jpg')))
                        print(len(I))
                        if len(I) == 0:
                            import pdb; pdb.set_trace()
                        for filename in I[k:k+1]:
                            print(filename)
                            raw_image = Image.open(filename) #.convert("RGB")
                            image = preprocess(raw_image).unsqueeze(0)
                            with torch.no_grad(), torch.cuda.amp.autocast():
                                image_features = model.encode_image(image)
                            image_features /= image_features.norm(dim=-1, keepdim=True)
                            data = {'animal':animal, 'params': betas[idx,:opts.animal_emb_dim], 'idx': idx, 'text_features': image_features[0,:]} 
                            self.data += [data]
            
            self.num_samples = len(self.data)

        if opts.object == 'tree':
            features = torch.load('./awol/code/data/tree_clip_features.pt')
            features /= features.norm(dim=-1, keepdim=True)

            for idx, tree in enumerate(tree_species):
                tree_path = join(opts.object_dir, opts.object, tree)
                print(tree)
                if exists(tree_path):
                    F = sorted(glob(join(tree_path, '*.npy')))

                    for i in range(len(F)):
                        params = np.load(open(F[i], 'rb'))

                        if self.opts.reorder_tree_shape:
                            params = tree_reorder_shape(params, direct='to_net')
                        if self.opts.compress_params:
                            params = tree_compress(params)
                        if self.opts.map_categorical:
                            params = map_categorical(params, direction='encode')

                        data = {'params': params, 'idx': idx, 'text_features': features[idx,:]} 
                        print(params.shape)
                        self.data += [data]
                num_records = len(self.data)
            
            if opts.use_images: 
                for k in range(opts.max_images):
                    for idx, tree in enumerate(tree_species):
                        print(tree)
                        tree_path = join(opts.object_dir, opts.object, tree)
                        if exists(tree_path):
                            F = sorted(glob(join(tree_path, '*.npy')))
                            for i in range(len(F)):
                                params = np.load(open(F[i], 'rb'))
                                if self.opts.reorder_tree_shape:
                                    params = tree_reorder_shape(params, direct='to_net')
                                if self.opts.compress_params:
                                    params = tree_compress(params)
                                if self.opts.map_categorical:
                                    params = map_categorical(params, direction='encode')
                                # Read the images and generate an entry for each image
                                img_path = join(opts.object_dir, opts.object, 'training_set', tree)
                                I = sorted(glob(join(img_path, '*.jpg')))
                                print(len(I))
                                for filename in I[k:k+1]:
                                    print(filename)
                                    raw_image = Image.open(filename) #.convert("RGB")
                                    image = preprocess(raw_image).unsqueeze(0)
                                    with torch.no_grad(), torch.cuda.amp.autocast():
                                        image_features = model.encode_image(image)
                                    image_features /= image_features.norm(dim=-1, keepdim=True)
                                    data = {'params': params, 'idx': idx, 'text_features': image_features[0,:]} 
                                    self.data += [data]
            

            self.num_samples = len(self.data)
            
            N = self.num_samples
            if self.opts.compress_params:
                all_params = np.zeros((N, 61))
                if self.opts.map_categorical:
                    all_params = np.zeros((N, 80))
            else:
                all_params = np.zeros((N, 105))
            for i in range(N):
                all_params[i,:] = self.data[i]['params']
            

            if self.opts.normalize:
                if opts.map_categorical:
                    idx = [*range(9,57)] + [*range(67,70)] + [*range(73,80)]
                    all_params = all_params[:,idx]
                self.mean_params = np.mean(all_params, axis=0)
                self.std_params = np.std(all_params, axis=0)
                np.save(opts.name+'_mean_params', self.mean_params)
                np.save(opts.name+'_std_params', self.std_params)
                if opts.map_categorical:
                    for i in range(self.num_samples):
                        self.data[i]['params'][idx] = (self.data[i]['params'][idx] - self.mean_params)/self.std_params
                else:
                    for i in range(self.num_samples):
                        self.data[i]['params'] = (self.data[i]['params'] - self.mean_params)/self.std_params
    def forward(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.forward(index)

    def __get_item__(self, index):
        return self.forward(index)

def object_data_loader(opts):
    dset = ObjectDataset(opts)
    dloader = DataLoader(
        dset,
        batch_size=opts.batch_size,
        shuffle=opts.shuffle,
        num_workers=opts.n_data_workers,
        drop_last=True)
    return dloader, dset
