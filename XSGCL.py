import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss,contrastLoss_ln_var,bpr_loss_pop
import os
import time
import pandas as pd

class XSGCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(XSGCL, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['XSGCL'])
        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.temp = float(args['-tau'])
        self.n_layers = int(args['-n_layer'])
        self.layer_cl = int(args['-l*'])
        self.high_freq_ratio = int(args['-high_freq_ratio'])
        self.noise_scale = int(args['-noise_scale'])
        self.noise_type = args['-noise_type']
        self.model = XSGCL_Encoder(self.data, self.emb_size, self.eps, self.n_layers,self.layer_cl)


    def train(self):
        model = self.model.cuda()
        ipop = self.pop()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            start_time = time.time()
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb= model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                pos_ipop = ipop[pos_idx]
                neg_ipop = ipop[neg_idx]
                rec_loss = bpr_loss_pop(user_emb, pos_item_emb, neg_item_emb,pos_ipop, neg_ipop,alpha=1e2)
                # rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                cl_loss = self.cal_cl_loss(user_emb, pos_item_emb)
                cl_loss = self.cl_rate * cl_loss
                batch_loss =  rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            self.fast_evaluation(epoch)
            end_time = time.time()
            print(f"code time:{end_time-start_time}s")
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb


    def cal_cl_loss(self, user_view1,item_view1):
        user_sslLoss = contrastLoss_ln_var(user_view1,self.temp,self.high_freq_ratio, self.noise_scale, self.noise_type)
        item_sslLoss = contrastLoss_ln_var(item_view1,self.temp,self.high_freq_ratio, self.noise_scale, self.noise_type)
        sslLoss = user_sslLoss + item_sslLoss
        return sslLoss

    def pop(self):
        current_d = os.getcwd()
        full_path = current_d + '/dataset/douban-book/train.txt'
        data = pd.read_csv(full_path, sep=' ', header=None,
                           names=['user_id', 'item_id', 'interaction'])

        item_interactions = data['item_id'].value_counts()

        total_interactions = item_interactions.sum()

        item_popularity = item_interactions / total_interactions

        item_popularity = item_popularity.sort_index()

        item_popularity_tensor = torch.tensor(item_popularity.values,dtype=torch.float32).cuda()

        min_val = item_popularity_tensor.min()
        max_val = item_popularity_tensor.max()
        item_popularity_tensor_normalized = (item_popularity_tensor - min_val) / (max_val - min_val)
        return item_popularity_tensor

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()
        content = {
            'model':self.model,
        }

        # Check if the directory exists, if not, create it
        model_dir = "../Models"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        # save model
        # torch.save(content, os.path.join(model_dir, 'lr_db_simgcl.mod'))

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

class XSGCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers, layer_cl):
        super(XSGCL_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.layer_cl = layer_cl
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()
        # self.soomthrelu = SmoothReLU()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def forward(self, perturbed=False):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        # all_embeddings_cl = ego_embeddings
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        final_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(final_embeddings, [self.data.user_num, self.data.item_num])
        return user_all_embeddings, item_all_embeddings

