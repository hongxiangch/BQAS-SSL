import torch
import torch.nn as nn
import numpy as np
from torch import optim
from tqdm import tqdm
from util import load_json, save_checkpoint_encoder, preprocessing
from model import Model_rl
from configs import configs
from torch.utils.data import TensorDataset, DataLoader

class Pretraining:
    def __init__(self, args):
        self.a = 1
        self.device = 'cuda' if args.cuda_for_embedding else 'cpu'

    def _build_dataset(self, list, expressibilities, dataset, ind_list):
        X_adj = []
        X_ops = []
        X_list = []
        X_express = []
        for ind in ind_list:
            X_adj.append(torch.Tensor(dataset[ind][0]))
            X_ops.append(torch.Tensor(dataset[ind][1]))
            X_list.append(torch.Tensor(list[ind]))
            X_express.append(torch.Tensor([expressibilities[ind].tolist()]))
        X_adj = torch.stack(X_adj)
        X_ops = torch.stack(X_ops)
        X_list = torch.stack(X_list)
        X_express = torch.stack(X_express)
        return X_list, X_express, X_adj, X_ops, torch.Tensor(ind_list)

    def pretraining_model(self, list, expressibilities, dataset, cfg, args):
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        end_training = False

        indices = np.random.permutation(range(len(dataset)))
        train_ind_list = indices[0:int(len(dataset))]
        X_list_train, X_express_train, X_adj_train, X_ops_train, indices_train = self._build_dataset(list, expressibilities, dataset, train_ind_list)

        model = Model_rl(input_dim=args.num_qubits+len(args.gate_type)+2, hidden_dim=args.hidden_dim, gmp_dim=args.gmp_dim, latent_dim=args.dim,
                           num_hops=args.hops, num_mlp_layers=3, dropout=args.dropout, num_ops=args.max_gate_num+2, device=self.device,args=args, **cfg['GAE'])

        optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08)
        loss_func = torch.nn.HuberLoss(reduction='mean', delta=0.5)
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        epochs = args.epochs
        bs = args.bs
        loss_total = []
        train_data = TensorDataset(X_list_train, X_express_train, X_adj_train, X_ops_train)
        train_loader = DataLoader(train_data, batch_size=bs, shuffle=True, drop_last=True)

        for epoch in range(0, epochs):
            model.train()

            loss_epoch = []
            for i, (list, express, adj, ops) in tqdm(enumerate(train_loader)):
                optimizer.zero_grad()
                # preprocessing
                adj, ops, prep_reverse = preprocessing(adj, ops, **cfg['prep'])
                # forward
                pred_expressibility = model(ops, adj.to(torch.long))
                express = express.squeeze()
                loss = loss_func(pred_expressibility, express)
                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                loss_epoch.append(loss.item())

            print('epoch {}: average loss {:.5f}'.format(epoch, sum(loss_epoch) / len(loss_epoch)))
            loss_total.append(sum(loss_epoch) / len(loss_epoch))
            model.eval()
            save_checkpoint_encoder(model, optimizer, epoch,
                                sum(loss_epoch) / len(loss_epoch), args.dim, args.dropout, args.seed, args.task, args.pretraining_num)

        print('loss for epochs: \n', loss_total)
        return model

    def pretraining(self, list_arcs_for_encoder, expressibilities, arc_for_training_encoder, args):
        cfg = configs[args.cfg]
        model = self.pretraining_model(list_arcs_for_encoder, expressibilities, arc_for_training_encoder, cfg, args)
        return model

    def load_encoder_model(self, model_loc, args):
        cfg = configs[args.cfg]
        model = Model_rl(input_dim=(len(args.gate_type) + args.num_qubits + 2), hidden_dim=args.hidden_dim, gmp_dim=args.gmp_dim,
                        latent_dim=args.dim,
                        num_hops=args.hops, num_mlp_layers=3, dropout=args.dropout, args=args, **cfg['GAE'])
        model.load_state_dict(torch.load(model_loc)['model_state'])
        return model

    def generate_embedding(self, model, arc_for_embedding, args):
        cfg = configs[args.cfg]
        Z1 = []
        adj_s = []
        ops_s = []
        for i in range(len(arc_for_embedding)):
            adj_s.append(arc_for_embedding[i][0])
            ops_s.append(arc_for_embedding[i][1])
            if len(adj_s) == 20 or i == (len(arc_for_embedding)-1):
                adj = torch.Tensor(np.array(adj_s))
                ops = torch.Tensor(np.array(ops_s))
                adj, ops, prep_reverse = preprocessing(adj, ops, **cfg['prep'])
                x = model._encoder(ops, adj)
                Z1.extend(x.detach().cpu().numpy())  # matrix embedding
                adj_s = []
                ops_s = []
                if (i+1) % 1000 == 0:
                    print('Generation progress:%d/%d' % (i+1, len(arc_for_embedding)))

        return Z1
