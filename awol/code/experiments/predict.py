from __future__ import absolute_import
  
import torch
from absl import app
from absl import flags
import numpy as np

from ..nnutils import object_net
from ..data.data_loader import tree_uncompress, tree_reorder_shape, map_categorical
import os
import os.path as osp

curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', 'cachedir')
flags.DEFINE_integer('gpu_id', 0, 'Which gpu to use')
flags.DEFINE_string('name', '', '')
flags.DEFINE_string('checkpoint_dir', osp.join(cache_path, 'snapshots'),
                            'Root directory for output files')
flags.DEFINE_integer('num_epochs', 1000, 'epochs')
#flags.DEFINE_integer('dog_emb_dim', 39, '')
flags.DEFINE_boolean('testset', False, '')
#flags.DEFINE_boolean('compute_PCA', False, '')
#flags.DEFINE_boolean('compress_params', False, '')
flags.DEFINE_boolean('images', False, '')
flags.DEFINE_integer('n_pred_samples', 1, '')
flags.DEFINE_float('sigma', 1., '')


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
    if opts.object == 'animal':
        if opts.images:
            if opts.testset:
                dataset = 'animal_image_clip_features_testset'
            else:
                dataset = 'animal_image_clip_features'
        else:
            if opts.testset:
                #dataset = 'animal_clip_features_testset_dogs_and_animals'
                dataset = 'animal_clip_features_testset_interpolation'
            else:
                dataset = 'animal_clip_features'
        out_dim = opts.animal_emb_dim

    if opts.object == 'tree':
        if opts.images:
            if opts.testset:
                dataset = 'tree_image_clip_features_testset'
            else:
                dataset = 'tree_image_clip_features'
        else:
            if opts.testset:
                dataset = 'tree_clip_features_testset'
            else:
                features = 'tree_clip_features'
        out_dim = 105

    features = torch.load(base_dir+dataset+'.pt') 
    features /= features.norm(dim=-1, keepdim=True)

    if opts.normalize:
        mean_params = torch.Tensor(np.load(opts.name+'_mean_params.npy')).to('cuda:0')
        std_params = torch.Tensor(np.load(opts.name+'_std_params.npy')).to('cuda:0')

    N = features.shape[0]
    print(N)
    n_samples = opts.n_pred_samples
    sigma = opts.sigma
    out = np.zeros((N*n_samples, out_dim))

    model = object_net.ObjectNet(opts)
    load_network(model, 'pred', opts.num_epochs, opts)
    model.eval()
    model = model.cuda(device=opts.gpu_id)

    for i in range(N):
        input_text_features = features[i,:].cuda(device=opts.gpu_id)[None,:]
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
                    out[i*n_samples+j,:] = tree_uncompress(p_params[None,:])
                else:
                    out[i*n_samples+j,:] = tree_uncompress(pred_params.detach().cpu()).numpy()
            else:
                out[i*n_samples+j,:] = pred_params.detach().cpu().numpy()
        if opts.reorder_tree_shape:
            out[i*n_samples+j,:] = tree_reorder_shape(out[i*n_samples+j,:], direct='to_blender')
        print(out[i*n_samples+j,:])
    if opts.testset:
        np.save("out_testset_"+opts.name+"_"+dataset, out)
    else:
        np.save("out", out)

def main(_):
    torch.manual_seed(0)
    np.random.seed(0)
    predict()

if __name__ == '__main__':
    app.run(main)






