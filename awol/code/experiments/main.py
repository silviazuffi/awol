from __future__ import absolute_import
  
import torch
from torch.autograd import Variable
from absl import app
from absl import flags
from collections import OrderedDict
import numpy as np

from ..data.data_loader import object_data_loader
from ..nnutils import train_utils, loss_utils, object_net
from os.path import basename

flags.DEFINE_boolean('add_log_prob', False, '')
flags.DEFINE_float('log_prob_weight', 0.1, '')
flags.DEFINE_string('loss_type', 'L1', '')

opts = flags.FLAGS

class ObjectTrainer(train_utils.Trainer):

    def define_model(self):
        opts = self.opts
        self.model = object_net.ObjectNet(opts)

        if opts.num_pretrain_epochs > 0:
            self.load_network(self.model, 'pred', opts.num_pretrain_epochs)

        self.model = self.model.cuda(device=opts.gpu_id)

        if opts.num_pretrain_epochs > 0:
            self.load_network(self.model, 'pred', opts.num_pretrain_epochs)
        return


    def init_dataset(self):
        self.dataloader, self.dataset = object_data_loader(opts)
        self.num_datapoints = self.opts.batch_size*len(self.dataloader)

        #if self.opts.object == 'dog':
        #    self.debug_data = np.zeros((39,39))
        print(self.num_datapoints)

    def define_losses(self):
        return

    def set_input(self, batch):
        input_params = batch['params'].type(torch.FloatTensor)
        input_text_features = batch['text_features'] 
        self.input_idx = batch['idx']
        self.input_text_features = Variable(input_text_features.cuda(device=self.opts.gpu_id), requires_grad=False)
        self.input_params = Variable(input_params.cuda(device=self.opts.gpu_id), requires_grad=False)


    def forward(self, epoch):
        self.loss = 0

        if self.opts.model_type == 'flow' or self.opts.model_type == 'glow':
            loss_log_prob = - self.model.forward(self.input_text_features, self.input_params, predict=False)
            pred_params = self.model.forward(self.input_text_features, predict=True)

            if self.opts.object == 'dog':
                self.debug_data[self.input_idx,:] = pred_params.detach().cpu().numpy()

            self.loss_pred = loss_utils.mse_loss(pred_params, self.input_params, self.opts.loss_type)
            self.loss_log_prob = loss_log_prob
            if self.opts.add_log_prob:
                self.loss += opts.log_prob_weight*loss_log_prob
            self.loss += self.loss_pred

        elif self.opts.model_type == 'diffusion':
            pred_params, loss_mse = self.model.forward(self.input_text_features, self.input_params, predict=False)
            self.loss_pred = torch.mean(loss_mse)
            self.debug_data[self.input_idx,:] = pred_params.detach().cpu().numpy()
            self.loss += self.loss_pred

        else:
            pred_params = self.model.forward(self.input_text_features, self.input_params)
            # Compute the loss
            self.loss_pred = loss_utils.mse_loss(pred_params, self.input_params, self.opts.loss_type)
            self.loss += self.loss_pred

    def get_current_scalars(self):
        sc_dict = OrderedDict([
            ('smoothed_loss', self.smoothed_loss.item()),
            ('loss_pred', self.loss_pred.item()),
            ('loss', self.loss.item()),
        ])
        if self.opts.add_log_prob:
            sc_dict['loss_log_prob'] = self.loss_log_prob.item()
        return sc_dict


def main(_):
    torch.manual_seed(0)
    np.random.seed(0)
    trainer = ObjectTrainer(opts)
    trainer.init_training()
    trainer.train()
    #if opts.object == 'dog':
    #    np.save("out_training", trainer.debug_data)

if __name__ == '__main__':
    app.run(main)






