# -*- coding=utf-8 -*-


class Config(object):

    def __init__(self):
        self.train_file = "data/transformer2.transformer.device.decoder.score.no-1.shuf"
        self.eval_file  = "data/transformer2.transformer.device.decoder.score.no-1.shuf"

        self.model_num = 4

        # metrics on eval set
        # err for equal error rate, auc for area under curve
        self.eval_metrics = "auc"
        # training loss
        # nll for negetive log likelihood loss, mse for mean squared error loss
        self.training_loss = "nll"
        # optimizer
        self.optimizer = "sgd"
        # lr
        self.lr = 0.01

        # training epoch
        self.epoch = 300
        # training batch size
        self.batch_size = 4096
        # do early stop or not
        self.do_early_stop = True
        # early stop rounds
        self.early_stop_rounds = 10
        # min eval improvement for early stop
        self.min_eval_impr = 0.000000001

