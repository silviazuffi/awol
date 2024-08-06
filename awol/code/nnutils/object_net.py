'''
    Author: Silvia Zuffi
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

clip_dim = 512 

from .latent_flows import get_generator

flags.DEFINE_string('model_type', 'flow', 'flow or diffusion or mlp')
flags.DEFINE_string('flow_type', 'realnvp_half', '')
flags.DEFINE_string('object', 'dog', 'dog or tree')
flags.DEFINE_boolean('train_mask', 'False', '')
flags.DEFINE_integer('num_hidden', 1024, 'Use 1024')
flags.DEFINE_integer('num_blocks', 5, 'Use 5')
flags.DEFINE_integer('animal_emb_dim', 145, '')
flags.DEFINE_boolean('noise', 'True', '')
flags.DEFINE_boolean('add_mask_cond', 'False', '')
flags.DEFINE_boolean('no_compression', 'False', '')
flags.DEFINE_integer('hidden_encode_dim', 512, 'Extra hidden layer in the scale/trans moduele, 0 means no layer')


class ObjectParamsPredictor(nn.Module):
    '''
        This module should implement the prediction of the tree/dog parameters from the textual description
    '''
    def __init__(self, opts):
        super(ObjectParamsPredictor, self).__init__()
        self.opts = opts

        if opts.object == 'tree':
            if opts.compress_params:
                self.emb_dim = 61 
                if opts.map_categorical:
                    self.emb_dim = 80
            else:
                self.emb_dim = 105
        if opts.object == 'animal':
            self.emb_dim = opts.animal_emb_dim

        if opts.model_type == 'flow':
            emb_dims = self.emb_dim
            cond_emb_dim = clip_dim
            device = 'cuda:0'
            flow_type = opts.flow_type #'realnvp_half' #'realnvp_half' # help='flow type: realnvp, real_nvp_half '
            num_blocks = opts.num_blocks
            num_hidden = opts.num_hidden
            self.pred_layer = get_generator(emb_dims, cond_emb_dim, device, flow_type=flow_type, num_blocks=num_blocks, num_hidden=num_hidden,
                    train_mask=opts.train_mask, mask_conditioning=opts.add_mask_cond, no_compression=opts.no_compression) 

    def forward(self, text_features, shape_features=None, predict=False, sigma=1.0):
        if self.opts.model_type == 'flow':
            if predict:
                num_samples = text_features.shape[0]
                if self.opts.noise:
                    noise = None
                else:
                    noise = torch.zeros(num_samples, self.emb_dim)
                x = self.pred_layer.sample(num_samples=num_samples, noise=noise, cond_inputs=text_features, num_inputs=self.emb_dim, sigma=sigma)
            else: # Return the log prob
                x = self.pred_layer.log_prob(shape_features, text_features).mean()  

        return x


class ObjectNet(nn.Module):
    def __init__(self, opts):
        super(ObjectNet, self).__init__()
        self.opts = opts
        self.object_params_predictor = ObjectParamsPredictor(opts)


    def forward(self, text, params=None, predict=False, sigma=1.):
        x = self.object_params_predictor(text, params, predict, sigma)
        return x


