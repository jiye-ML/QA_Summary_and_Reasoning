# -*- coding:utf-8 -*-
# Created by LuoJie at 11/22/19
from gensim.models.word2vec import LineSentence, Word2Vec
import numpy as np
import codecs
# 引入日志配置
import logging

from utils.config import embedding_matrix_path


def load_word2vec_file(save_wv_model_path):
    # 保存词向量模型
    wv_model = Word2Vec.load(save_wv_model_path)
    embedding_matrix = wv_model.wv.vectors
    return embedding_matrix


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def get_vocab(save_wv_model_path):
    # 保存词向量模型
    wv_model = Word2Vec.load(save_wv_model_path)
    reverse_vocab = {index: word for index, word in enumerate(wv_model.wv.index2word)}
    vocab = {word: index for index, word in enumerate(wv_model.wv.index2word)}
    return vocab, reverse_vocab


def load_vocab(file_path):
    """
    读取字典
    :param file_path: 文件路径
    :return: 返回读取后的字典
    """
    vocab = {}
    reverse_vocab = {}
    for line in open(file_path, "r", encoding='utf-8').readlines():
        word, index = line.strip().split("\t")
        index = int(index)
        vocab[word] = index
        reverse_vocab[index] = word
    return vocab, reverse_vocab


def load_embedding_matrix():
    """
    加载 embedding_matrix_path
    """
    return np.load(embedding_matrix_path + '.npy')
