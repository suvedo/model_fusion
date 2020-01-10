# -*- coding=utf-8 -*-

from config import Config
from linear_model import NormLinearModel

if __name__ == "__main__":
    conf = Config()
    model = NormLinearModel(conf)
    model.train_and_eval()
