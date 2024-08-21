from smpl_webuser.serialization import save_model, load_model
import os
import pickle as pkl
import numpy as np
from psbody.mesh.meshviewer import MeshViewer, MeshViewers
from psbody.mesh.mesh import Mesh
from glob import glob
from os.path import basename, join

Dir = './results/'

Animate = True

if __name__ == '__main__':

    model_filename = '../awol/data/animal/smal_plus.pkl'
    smpl = load_model(model_filename)

    mv = MeshViewer()
    mv.set_background_color(np.ones(3))
    n_samples=1
    err = 0
    j = 0
    filenames = glob(join(Dir, '*.npy'))

    for file in filenames:
        print(file)
        betas = np.load(file)
        nB1 = betas.shape[1]
        print(betas.shape)
        if True:
            smpl.betas[:] = 0
            smpl.betas[:nB1] = betas[0,:]
            smpl.trans[0] = 0
            smpl.pose[:] = 0
            smpl.pose[25*3+1] = -1.2
            M = Mesh(v=smpl.r, f=smpl.f) #.set_vertex_colors(color)
            mv.set_static_meshes([M])
            import pdb; pdb.set_trace()
            M.write_obj(filename = Dir+basename(file)[:-4]+'_awol.obj')
