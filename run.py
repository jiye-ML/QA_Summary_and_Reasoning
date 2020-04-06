# -*- coding:utf-8 -*-
# Created by LuoJie at 12/7/19

from seq2seq_tf2.train import train
from utils.params_utils import get_params


def main():
    # 获得参数
    params = get_params()
    # 训练模型
    if params["mode"] == "train":
        train(params)


if __name__ == '__main__':
    main()
