import torch
import numpy as np
from scipy.stats import norm
from torch.utils.data import DataLoader, TensorDataset
import lds.smooth_label_space as smooth_func
from lds.loss import weighted_mse_loss, weighted_bce_loss, BPRLoss

class BANAQAS:
    def __init__(self, args):
        self.args = args
        self.E0 = self.args.E0

    def train(self, model, train_loader, args):
        print("Starting training...")
        # End-to-end training
        for param in model.encoder.parameters():
            param.requires_grad = True
        model.train()
        loss_func = BPRLoss()  # Define loss function
        train_loss = []  # Record the loss
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_for_bo)

        for i in range(args.epochs_for_init):
            for step, batch in enumerate(train_loader):
                if args.lds:
                    (b_adj, b_ops, b_y, w) = batch
                else:
                    (b_adj, b_ops, b_y) = batch
                optimizer.zero_grad()  # Zero the gradient buffer of the optimizer to prepare for the gradients of the new batch of data

                forward, _ = model(b_adj, b_ops)
                if args.lds:
                    loss = loss_func.compute_loss(forward, b_y, w)
                else:
                    loss = loss_func.compute_loss(forward, b_y)
                loss.backward()
                optimizer.step()  # Use the optimizer to update the model parameters.
                train_loss.append(loss.item())

        # Clear gradients
        optimizer.zero_grad()

        return model.encoder, model

    def predict(self, model, test_loader):
        print("Starting prediction...")
        model.eval()
        prediction = []
        with torch.no_grad():
            for (adj, ops) in test_loader:
                pred = model(adj, ops)[0].detach().cpu().tolist()
                prediction.extend(pred)

        # Return the indices of these samples
        return prediction

    def acq_fn(self, predictions, ytrain=None, stds=None, explore_type='its'):
        predictions = np.array(predictions)

        if stds is None:
            stds = np.sqrt(np.var(predictions, axis=0))

        # Upper confidence bound (UCB) acquisition function
        if explore_type == 'ucb':
            explore_factor = 0.5
            mean = np.mean(predictions, axis=0)
            ucb = mean - explore_factor * stds
            sorted_indices = np.argsort(ucb)

        # Expected improvement (EI) acquisition function
        elif explore_type == 'ei':
            ei_calibration_factor = 5.
            mean = list(np.mean(predictions, axis=0))
            factored_stds = list(stds / ei_calibration_factor)
            min_y = ytrain.min()
            gam = [(min_y - mean[i]) / factored_stds[i] for i in range(len(mean))]
            ei = [-1 * factored_stds[i] * (gam[i] * norm.cdf(gam[i]) + norm.pdf(gam[i]))
                  for i in range(len(mean))]
            sorted_indices = np.argsort(ei)

        # Probability of improvement (PI) acquisition function
        elif explore_type == 'pi':
            mean = list(np.mean(predictions, axis=0))
            stds = list(stds)
            min_y = ytrain.min()
            pi = [-1 * norm.cdf(min_y, loc=mean[i], scale=stds[i]) for i in range(len(mean))]
            sorted_indices = np.argsort(pi)

        # Thompson sampling (TS) acquisition function
        elif explore_type == 'ts':
            rand_ind = np.random.randint(predictions.shape[0])
            ts = predictions[rand_ind,:]
            sorted_indices = np.argsort(ts)

        # Top exploitation
        elif explore_type == 'percentile':
            min_prediction = np.min(predictions, axis=0)
            sorted_indices = np.argsort(min_prediction)

        # Top mean
        elif explore_type == 'mean':
            mean = np.mean(predictions, axis=0)
            sorted_indices = np.argsort(mean)

        elif explore_type == 'confidence':
            confidence_factor = 2
            mean = np.mean(predictions, axis=0)
            conf = mean + confidence_factor * stds
            sorted_indices = np.argsort(conf)

        # Independent Thompson sampling (ITS) acquisition function
        elif explore_type == 'its':
            mean = np.mean(predictions, axis=0)
            samples = np.random.normal(mean, stds)
            sorted_indices = np.argsort(samples)

        else:
            print('{} is not a valid exploration type'.format(explore_type))
            raise NotImplementedError()

        return sorted_indices

    def prepare_data(self, adj, ops, labels, visited):
        # Preparing training data
        if self.args.task == 'vqe_36gate_h_head_new':
            self.E0 = -7.7274067
            param = 14
        elif self.args.task == 'vqe_5_constraint':
            self.E0 = -8.472136
            param = 20
        elif self.args.task == 'Maxcut_12':
            if self.args.seed2 < 3:
                self.E0 = -14
                param = 28
            elif self.args.seed2 == 3:
                self.E0 = -18
                param = 36
            elif self.args.seed2 == 4:
                self.E0 = -17
                param = 34
        else:
            print('Invalid task.')
            exit(0)

        x_train_adj = torch.Tensor(adj[visited])
        x_train_ops = torch.Tensor(ops[visited])
        y_train = labels[visited].astype(np.float64)
        y_train = ((y_train - self.E0) / param)
        y_train = torch.Tensor(y_train)

        train_data = TensorDataset(x_train_adj, x_train_ops, y_train)

        train_loader = DataLoader(train_data, batch_size=self.args.bs_for_train_bo, shuffle=True, drop_last=True)
        return train_loader



