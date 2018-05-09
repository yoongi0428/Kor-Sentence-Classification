# -*- coding: utf-8 -*-
import argparse
import datetime

import numpy as np
import tensorflow as tf
from util.Dataset import Dataset

from models.Char_CNN import Char_CNN
from models.Wide_Deep import WD_CNN
from models.Wide import Wide_CNN
from models.VDCNN2 import VDCNN2
from models.LSTM import LSTM
from models.MULTI_LSTM import MULTI_LSTM

def print_info(config):
    """
    Print Training Setup
    """
    print('Output : ', config.output)
    print('Number of Epochs : ', config.epochs)
    print('Learning Rate : ', config.lr)
    print('Batch Size : ', config.batch)
    print('Char Vocab Size : ', config.charsize)
    print('String Max. Length : ', config.strmaxlen)
    print('Eumjeol : ', config.eumjeol)

def get_model(model):
    """
    Get Model instance
    """
    assert model in ['CHAR', 'WD', 'WIDE', 'VDCNN', 'LSTM', 'MULTI_LSTM']

    if model == 'CHAR': return Char_CNN(config, conv_layers, fc_layers)
    elif model == 'WD': return WD_CNN(config, conv_layers, wconv_layers, fc_layers)
    elif model == 'WIDE' : return Wide_CNN(config, wconv_layers)
    elif model == 'VDCNN': return VDCNN2(config)
    elif model == 'LSTM' : return LSTM(config, fc_layers)
    elif model == 'MULTI_LSTM': return MULTI_LSTM(config, fc_layers, rnn_layers)

def _batch_loader(iterable, n=1):
    """
    Yield Data by batch size like DataLodader in PyTorch
    """
    length = len(iterable)
    for n_idx in range(0, length, n):
        yield iterable[n_idx:min(n_idx + n, length)]


def is_better_result(best, cur):
    return best < cur

if __name__ == '__main__':
    # Argument
    args = argparse.ArgumentParser(description="Sentence Classification in Korean")

    args.add_argument('--output', type=int, default=2, help="Number of Classes")
    args.add_argument('--epochs', type=int, default=50, help="Number of Training Epoch")
    args.add_argument('--batch', type=int, default=100, help="Batch Size")
    args.add_argument('--lr', type=float, default=0.005, help="Learning rate")
    args.add_argument('--strmaxlen', type=int, default=250, help="Maximum Length of char/eum To Process")
    args.add_argument('--charsize', type=int, default=148, help="Character Vocab Size")
    args.add_argument('--rnn_hidden', type=int, default=256, help="RNN Hidden Size")
    args.add_argument('--filter_num', type=int, default=256, help="Number of CNN Filter")
    args.add_argument('--emb', type=int, default=0, help="Embedding Size")
    args.add_argument('--bi', action='store_true', default=False, help="Bidirectional if Specified")
    args.add_argument('--eumjeol', action='store_true', default=False, help="Use Eumjeol if Specified")
    args.add_argument('--model', type=str, default='WIDE', help="CHAR / WIDE / WD / VDCNN / LSTM / MULTI_LSTM") # CHAR, VDCNN, WD, WIDE, LSTM, MULTI_LSTM
    config = args.parse_args()
    if config.eumjeol:
        config.charsize = 2431

    DATASET_PATH = './data/'
    DISPLAY_STEP = 50
    SUBTEST_STEP = 1
    NUM_CLASSES = config.output
    # Conv layer : Char_CNN, WD, Wconv_layer : Wide, WD
    # [Filter Num, Filter Size, Pool Size]
    conv_layers = [
        [config.filter_num, 7, 3],
        [config.filter_num, 7, 3],
        [config.filter_num, 3, -1],
        [config.filter_num, 3, -1],
        [config.filter_num, 3, -1],
        [config.filter_num, 3, 3]
    ]
    wconv_layers = [
        [config.filter_num, 2, -1],
        [config.filter_num, 3, -1],
        [config.filter_num, 4, -1],
        [config.filter_num, 5, -1],
    ]
    # Output Layer
    fc_layers = [1024, 1024]
    # Multilayer LSTM, len : Num of layers
    rnn_layers = [256, 256]

    # Load Data
    ####################################################################################
    print("Loading Dataset...")
    DATASET = Dataset(DATASET_PATH, num_classes=NUM_CLASSES, eumjeol=config.eumjeol, max_len=config.strmaxlen)

    train_len = len(DATASET)
    one_batch_size = train_len // config.batch
    if train_len % config.batch != 0:
        one_batch_size += 1
    config.num_batch = one_batch_size

    # Model specification
    ####################################################################################
    tf.set_random_seed(234)
    model = get_model(config.model)

    prediction = model.prediction
    accuracy = model.accuracy
    train_step = model.train_step
    loss = model.loss
    x1 = model.x1
    y_ = model.y_
    if config.model == 'VDCNN':
        is_train = model.train    # For VDCNN

    ####################################################################################
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))

    print('=' * 15 + "TRAINING DETAIL" + '=' * 15)
    print_info(config)
    print('=' * 17 + "MODEL INFO" + '=' * 17)
    print(model)
    print()

    # TRAIN

    patience = 5        # Patience for early stop. Stop if no improvement has been made for patience epoch after best
    best_result = -1
    best_epoch = -1
    print('=' * 15 + "TRAINING START" + '=' * 15)

    tf.global_variables_initializer().run()
    for epoch in range(1, config.epochs + 1):
        epoch_loss = 0.0
        DATASET.shuffle()   # Shuffle at every epoch
        for i, data in enumerate(_batch_loader(DATASET, config.batch)):
            data1 = data[0]
            labels = data[1].flatten()

            feed_dict = {x1: data1, y_: labels}
            # For VDCNN
            if config.model == 'VDCNN':
                feed_dict[is_train] = True

            _, l = sess.run([train_step, loss], feed_dict=feed_dict)

            if (i + 1) % DISPLAY_STEP == 0 or (i + 1) == one_batch_size:
                time_str = datetime.datetime.now().isoformat()
                print('[%s] Batch : (%3d/%3d), LOSS in this minibatch : %.4f' % (time_str, i + 1, one_batch_size, float(l)))
            epoch_loss += l
        print('\nepoch:', epoch, ' train_loss:', epoch_loss)

        # Test every SUBTEST_STEP
        if epoch % SUBTEST_STEP == 0:
            print('=' * 8 + "[Epoch %d] SUBTEST" % epoch + '=' * 8)
            pred = []
            for i, data in enumerate(_batch_loader(DATASET.test[0], config.batch)):
                feed_dict = {x1: data}
                if config.model == 'VDCNN':
                    feed_dict[is_train] = False
                p = sess.run(prediction, feed_dict=feed_dict)
                pred += p.tolist()
            correct = len(np.where(pred == DATASET.test[1])[0])
            total = len(DATASET.test[0])
            acc = correct / total

            print("[TEST ACC] : %.4f" % acc)
            if is_better_result(best_result, acc) or best_result == -1:
                best_result = acc
                best_epoch = epoch
            elif epoch - best_epoch > patience:
                print("Early Stop...!")
                break
            print("")

    print("Best ACCURACY : %.4f At Epoch %d" % (best_result, best_epoch))