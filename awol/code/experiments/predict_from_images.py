'''
    Author: Silvia Zuffi

'''


from __future__ import absolute_import
  
import torch
from absl import app
from absl import flags
import numpy as np

from ..nnutils import object_net
from ..data.data_loader import tree_uncompress, tree_reorder_shape, map_categorical
import os
import os.path as osp
from glob import glob
from os.path import join, basename
from PIL import Image

curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', 'cachedir')
flags.DEFINE_integer('gpu_id', 0, 'Which gpu to use')
flags.DEFINE_string('name', '', '')
flags.DEFINE_string('checkpoint_dir', osp.join(cache_path, 'snapshots'),
                            'Root directory for output files')
flags.DEFINE_integer('num_epochs', 1000, 'epochs')
flags.DEFINE_boolean('testset', False, '')
flags.DEFINE_boolean('images', False, '')
flags.DEFINE_integer('n_pred_samples', 1, '')
flags.DEFINE_float('sigma', 1., '')
flags.DEFINE_string('images_dir', '', '')
flags.DEFINE_string('extension', 'jpg', '')

import open_clip

opts = flags.FLAGS

def load_network(network, network_label, epoch_label, opts):
    save_filename = '{}_net_{}.pth'.format(network_label, epoch_label)
    network_dir = os.path.join(opts.checkpoint_dir, opts.name)
    save_path = os.path.join(network_dir, save_filename)
    print('loading {}..'.format(save_path))
    network.load_state_dict(torch.load(save_path), strict=False)
    return


def predict():

    base_dir = './awol/code/data/' 
    out_dim = opts.animal_emb_dim

    if opts.object == 'tree':
        out_dim = 105

    if opts.normalize:
        mean_params = torch.Tensor(np.load(opts.name+'_mean_params.npy')).to('cuda:0')
        std_params = torch.Tensor(np.load(opts.name+'_std_params.npy')).to('cuda:0')

    model = object_net.ObjectNet(opts)
    load_network(model, 'pred', opts.num_epochs, opts)
    model.eval()
    model = model.cuda(device=opts.gpu_id)
    n_samples = opts.n_pred_samples
    sigma = opts.sigma


    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    files = glob(join(opts.images_dir, '*.'+opts.extension))
    out = np.zeros((n_samples, out_dim))
    for filename in files:
        raw_image = Image.open(filename) 
        image = preprocess(raw_image).unsqueeze(0)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = clip_model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        input_text_features = image_features.cuda(device=opts.gpu_id)
        for j in range(n_samples):
            if opts.model_type == 'flow' or opts.model_type == 'diffusion':
                pred_params = model.forward(input_text_features, predict=True, sigma=sigma)
            else:
                pred_params = model.forward(input_text_features)
            if opts.compress_params:
                if opts.normalize:
                    if opts.map_categorical:
                        idx = [*range(9,57)] + [*range(67,70)] + [*range(73,80)]
                        pred_params[0,idx] = (pred_params[0,idx] * std_params) + mean_params
                    else:
                        pred_params = (pred_params * std_params) + mean_params
                if opts.map_categorical:
                    p_params = map_categorical(pred_params[0,:].detach().cpu().numpy(), direction='decode')
                    out[j,:] = tree_uncompress(p_params[None,:])
                else:
                    out[j,:] = tree_uncompress(pred_params.detach().cpu()).numpy()
            else:
                out[j,:] = pred_params.detach().cpu().numpy()
        if opts.reorder_tree_shape:
            out[j,:] = tree_reorder_shape(out[i*n_samples+j,:], direct='to_blender')
        print(out[j,:])
        image_name = basename(filename)[:-5]
        np.save("out_images_"+opts.name+"_"+image_name, out)

def main(_):
    torch.manual_seed(0)
    np.random.seed(0)
    predict()

if __name__ == '__main__':
    app.run(main)






