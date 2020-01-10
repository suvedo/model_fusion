# -*- coding=utf-8 -*-


import torch
from torch import nn
from util import DataIter, cal_eer, cal_auc


class NormLinearModel(nn.Module):

    def __init__(self, conf):
        super(NormLinearModel, self).__init__()
        self.conf = conf
        self.build_model()
        self.train_iter = DataIter(self.conf.train_file, self.conf.batch_size)
        self.eval_iter  = DataIter(self.conf.eval_file)

    def train_and_eval(self):
        # 定义loss
        if self.conf.training_loss == "mse":
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.NLLLoss()

        # 定义优化算法
        if self.conf.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.conf.lr)

        for i in range(self.conf.epoch):
            for x, y in self.train_iter:
                y_hat = self.forward(torch.tensor(x))
                if self.conf.training_loss == "mse":
                    y = [[float(yi[0])] for yi in y]
                else:
                    y_hat = torch.log(torch.cat((1 - y_hat, y_hat), dim=1))

                y = torch.tensor(y)

                # 计算loss
                if self.conf.training_loss == "mse":
                    l = self.loss(y_hat, y)
                else:
                    l = self.loss(y_hat, y.view(y.shape[0]))
                # 梯度清零
                self.optimizer.zero_grad()
                # 计算梯度
                l.backward()

                # 更新参数
                self.optimizer.step()

            print('{} epoch weights: {}'.format(i, self.weights / torch.sum(self.weights)))

            for x, y in self.eval_iter:
                y_hat = self.forward(torch.tensor(x))
                if self.conf.eval_metrics == "eer":
                    eval_val = cal_eer(y, y_hat.detach().numpy().tolist())
                    print("{} epoch eer: {}".format(i, eval_val))
                else:
                    eval_val = cal_auc(y, y_hat.detach().numpy().tolist())
                    print("{} epoch auc: {}".format(i, eval_val))

                if self.conf.do_early_stop:
                    if i == 0: # first epoch
                        no_eval_impr_cnt = 0
                        last_eval_val = eval_val
                        eval_epoch = 0
                        eval_weights = self.weights / torch.sum(self.weights)
                    else:
                        if self.conf.eval_metrics == "eer":
                            if eval_val <= last_eval_val - self.conf.min_eval_impr:
                                no_eval_impr_cnt = 0
                                last_eval_val = eval_val
                                eval_epoch = i
                                eval_weights = self.weights / torch.sum(self.weights)
                            else:
                                no_eval_impr_cnt += 1
                        else:
                            if eval_val - self.conf.min_eval_impr >= last_eval_val:
                                no_eval_impr_cnt = 0
                                last_eval_val = eval_val
                                eval_epoch = i
                                eval_weights = self.weights / torch.sum(self.weights)
                            else:
                                no_eval_impr_cnt += 1
                        if no_eval_impr_cnt == self.conf.early_stop_rounds:
                            print('early stop on {} epoch, weights: '.format(eval_epoch), 
                                    eval_weights)
                            return

        print("runs {} epoch: ".format(self.conf.epoch), self.weights / torch.sum(self.weights))


    def build_model(self): 
        model_num = self.conf.model_num
        #self.weights = nn.Parameter(torch.rand(model_num, 1, dtype = torch.float32))
        self.weights = nn.Parameter(torch.ones(model_num, 1))

    def forward(self, x):
        norm_weights = self.weights / torch.sum(self.weights)
        return torch.mm(x, norm_weights)


