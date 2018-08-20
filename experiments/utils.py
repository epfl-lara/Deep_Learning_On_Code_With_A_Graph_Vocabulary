import itertools
import logging
import logging.handlers
import math
import os
import pprint
import socket
import subprocess
import time
from typing import Tuple

import editdistance
import mxnet as mx
import numpy as np
import sklearn.metrics as metrics
from mxnet import nd, gluon
from mxnet.gluon.loss import SigmoidBinaryCrossEntropyLoss
from tqdm import tqdm

from data import AsyncDataLoader
from experiments.aws_config import aws_config

logger = logging.getLogger()


class PaddedArray:
    def __init__(self, values, value_lengths):
        self.values = values
        self.value_lengths = value_lengths

    def as_in_context(self, ctx):
        new_PA = PaddedArray(self.values.as_in_context(ctx), self.value_lengths.as_in_context(ctx))
        return new_PA


def get_time():
    os.environ['TZ'] = 'US/Pacific'
    time.tzset()
    t = time.strftime('%a_%b_%d_%Y_%H%Mhrs', time.localtime())
    return t


def start_logging(log_dir, debug: bool = False):
    logger = logging.getLogger()
    if debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    logger.setLevel(log_level)

    if not any(type(i) == logging.StreamHandler for i in logger.handlers):
        sh = logging.StreamHandler()
        sh.setLevel(log_level)
        logger.addHandler(sh)
    file_handlers = [i for i in logger.handlers if type(i) == logging.FileHandler]
    for h in file_handlers:
        logger.removeHandler(h)
    os.makedirs(log_dir, exist_ok=True)
    logger.info('Logging to {}'.format(log_dir))
    fh = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
    fh.setLevel(log_level)
    fh.setFormatter(
        logging.Formatter(fmt='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M'))
    logger.addHandler(fh)
    # Only works if you have an SMTP server like postfix running on your box
    if aws_config['email_to_send_alerts_to'] and not any(
            type(i) == logging.handlers.SMTPHandler for i in logger.handlers):
        eh = logging.handlers.SMTPHandler('localhost',
                                          'logger@{}'.format(socket.getfqdn()),
                                          aws_config['email_to_send_alerts_to'],
                                          'Log Message from {} about {}'.format(socket.getfqdn(), log_dir))
        eh.setLevel(logging.ERROR)
        eh.setFormatter(
            logging.Formatter('Hey there, heads up:\n\n%(name)-12s: %(levelname)-8s %(message)s'))
        logger.addHandler(eh)

    return logger


def tuple_of_tuples_to_padded_array(tup_of_tups: Tuple[Tuple[int, ...], ...], ctx, pad_amount=None):
    '''
    Converts a tuple of tuples into a PaddedArray (i.e. glorified pair of nd.Arrays for working with SequenceMask)
    Pads to the length of the longest tuple in the outer tuple, unless pad_amount is specified.
    '''
    value_lengths = nd.array([len(i) for i in tup_of_tups], dtype='float32',
                             ctx=ctx)  # float type to play nice with SequenceMask later
    if pad_amount is not None and value_lengths.max().asscalar() < pad_amount:
        tup_of_tups = list(tup_of_tups)
        tup_of_tups[0] = tup_of_tups[0] + (0,) * (pad_amount - len(tup_of_tups[0]))
    values = list(itertools.zip_longest(*tup_of_tups, fillvalue=0))
    values = nd.array(values, dtype='int32', ctx=ctx).T[:, :pad_amount]
    return PaddedArray(values, value_lengths)


def evaluate_loss(data_loader: AsyncDataLoader, model, loss_fxn):
    with data_loader as data_loader:
        total_loss = nd.zeros((1,), ctx=data_loader.ctx[0])
        for split_batch, batch_length in tqdm(data_loader, total=data_loader.total_batches):
            losses = [loss_fxn(model(batch.data), batch.label, model.data_encoder) for batch in split_batch]
            loss_sums = nd.concat(*[loss.sum().as_in_context(data_loader.ctx[0]) for loss in losses], dim=0)
            total_loss += nd.sum(loss_sums)
            total_loss.wait_to_read()
    return total_loss.asscalar() / len(data_loader)


def evaluate_FITB_accuracy(data_loader: AsyncDataLoader, model):
    '''
    Measures the accuracy of the model in indicating the correct variable
    '''
    with data_loader as data_loader:
        correct = 0
        for split_batch, batch_length in tqdm(data_loader, total=data_loader.total_batches):
            batches_outputs = [(batch, model(batch.data)) for batch in split_batch]
            for batch, output in batches_outputs:
                predictions_labels = model.unbatchify(batch, output)
                for prediction, label in predictions_labels:
                    correct += int(nd.dot(prediction, label).asscalar())
    return correct / len(data_loader)


def evaluate_full_name_accuracy(data_loader: AsyncDataLoader, model):
    '''
    Measures the accuracy of the model in predicting the full true name, in batches
    '''
    logged_example = False
    with data_loader as data_loader:
        correct = 0
        for split_batch, batch_length in tqdm(data_loader, total=data_loader.total_batches):
            batches_outputs = [(batch, model(batch.data)) for batch in split_batch]
            for batch, output in batches_outputs:
                predictions_labels = model.unbatchify(batch, output)
                for prediction, label in predictions_labels:
                    if not logged_example:
                        logger.info('Some example predictions:\n{}'.format(pprint.pformat(predictions_labels[:10])))
                        logged_example = True
                    if prediction == label:
                        correct += 1
    return correct / len(data_loader)


def evaluate_subtokenwise_accuracy(data_loader: AsyncDataLoader, model):
    '''
    Measures the accuracy of the model in predicting each subtoken in the true names (with penalty for extra subtokens)
    '''
    logged_example = False
    with data_loader as data_loader:
        correct = 0
        total = 0
        for split_batch, batch_length in tqdm(data_loader, total=data_loader.total_batches):
            batches_outputs = [(batch, model(batch.data)) for batch in split_batch]
            for batch, output in batches_outputs:
                predictions_labels = model.unbatchify(batch, output)
                for prediction, label in predictions_labels:
                    if not logged_example:
                        logger.info('Some example predictions:\n{}'.format(pprint.pformat(predictions_labels[:10])))
                        logged_example = True
                    for i in range(min(len(prediction), len(label))):
                        if prediction[i] == label[i]:
                            correct += 1
                    total += max(len(prediction), len(label))
    return correct / total


def evaluate_edit_distance(data_loader: AsyncDataLoader, model):
    '''
    Measures the mean (over instances) of the characterwise edit distance (Levenshtein distance) between predicted and true names
    '''
    logged_example = False
    with data_loader as data_loader:
        cum_edit_distance = 0
        for split_batch, batch_length in tqdm(data_loader, total=data_loader.total_batches):
            batches_outputs = [(batch, model(batch.data)) for batch in split_batch]
            for batch, output in batches_outputs:
                predictions_labels = model.unbatchify(batch, output)
                for prediction, label in predictions_labels:
                    if not logged_example:
                        logger.info('Some example predictions:\n{}'.format(pprint.pformat(predictions_labels[:10])))
                        logged_example = True
                    pred_name = ''.join(prediction)
                    real_name = ''.join(label)
                    cum_edit_distance += editdistance.eval(pred_name, real_name)
    return cum_edit_distance / len(data_loader)

    pred = []
    true = []
    for i in tqdm(range(0, math.ceil(len(dataset) / n_batch))):
        data = dataset[n_batch * i:n_batch * (i + 1)]
        graph, label = model.batchify(data, ctx)
        output = model(graph)
        predictions = nd.argmax(output, axis=2)
        # Masking output to max(length_of_output, length_of_label)
        output_preds = predictions.asnumpy()
        output_lengths = []
        for row in output_preds:
            end_token_idxs = np.where(row == 0)[0]
            if len(end_token_idxs):
                output_lengths.append(int(min(end_token_idxs)))
            else:
                output_lengths.append(model.max_name_length)
        output_lengths = nd.array(output_lengths, ctx=ctx)
        mask_lengths = nd.maximum(output_lengths, label.value_lengths)

        output = nd.SequenceMask(predictions, value=-1, use_sequence_length=True, sequence_length=mask_lengths,
                                 axis=1).asnumpy().astype('int32')
        labels = nd.SequenceMask(label.values, value=-1, use_sequence_length=True,
                                 sequence_length=mask_lengths.astype('int32'), axis=1).asnumpy()

        pred += [i for i in output.flatten().tolist() if i >= 0]
        true += [i for i in labels.flatten().tolist() if i >= 0]
    return metrics.f1_score(true, pred, average='weighted')


class FITBLoss(mx.gluon.HybridBlock):
    def hybrid_forward(self, F, output, *args, **kwargs):
        label, _ = args
        loss = SigmoidBinaryCrossEntropyLoss()
        return loss(output, label)


class VarNamingLoss(mx.gluon.HybridBlock):
    def hybrid_forward(self, F, output, *args, **kwargs):
        '''
        Masks the outputs and returns the SoftMaxCrossEntropy loss
        output is a (batch x max_name_length x len(all_node_name_subtokens)) tensor of name predictions for each graph
        Note: last dimension of output are pre-softmax values - SoftmaxCrossEntropy does the softmax
        '''
        (label, _), data_encoder = args
        softmax_xent = gluon.loss.SoftmaxCrossEntropyLoss(axis=2)

        # Masking output to max(where_RNN_emitted_PAD_token, length_of_label)
        output_preds = F.argmax(output, axis=2).asnumpy()
        output_lengths = []
        for row in output_preds:
            end_token_idxs = np.where(row == data_encoder.all_node_name_subtokens['__PAD__'])[0]
            if len(end_token_idxs):
                output_lengths.append(int(min(end_token_idxs)))
            else:
                output_lengths.append(output_preds.shape[1])
        output_lengths = F.array(output_lengths, ctx=output.context)
        mask_lengths = F.maximum(output_lengths, label.value_lengths)
        output = F.SequenceMask(output, use_sequence_length=True, sequence_length=mask_lengths, axis=1)
        return softmax_xent(output, label.values)


class VarNamingGraphVocabLoss(mx.gluon.HybridBlock):
    def hybrid_forward(self, F, output, *args, **kwargs):
        '''
        Returns the Softmax Cross Entropy loss of a model with a graph vocab, in the style of a sentinel pointer network
        Note: Unlike VarNamingLoss, this Loss DOES expect the last dimension of output to be probabilities summing to 1
        '''
        (label, _), data_encoder = args
        joint_label, label_lengths = label.values, label.value_lengths
        # We're using pick and not just sparse labels for XEnt b/c there can be multiple ways to point to the correct subtoken
        loss = nd.pick(output, joint_label, axis=2)

        # Masking outputs to max(length_of_output (based on emitting value 0), length_of_label)
        output_preds = nd.argmax(output, axis=2).asnumpy()
        output_lengths = []
        for row in output_preds:
            end_token_idxs = np.where(row == 0)[0]
            if len(end_token_idxs):
                output_lengths.append(int(min(end_token_idxs)) + 1)
            else:
                output_lengths.append(output.shape[1])
        output_lengths = nd.array(output_lengths, ctx=output.context)
        mask_lengths = nd.maximum(output_lengths, label_lengths)
        loss = nd.SequenceMask(loss, value=1.0, use_sequence_length=True, sequence_length=mask_lengths, axis=1)

        return nd.mean(-nd.log(loss), axis=0, exclude=True)


def s3_sync(source_path: str, target_path: str):
    '''
    Syncs the directory/file at source_path to target_path via the aws s3 CLI
    '''
    cmd = "aws s3 sync {} {} --profile {}".format(source_path, target_path, aws_config['local_config_profile_name'])
    logger.info('Running: {}'.format(cmd))
    subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ.copy())


def s3_cp(source_path: str, target_path: str, recursive=False):
    '''
    Copies the directory/file at source_path to target_path via the aws s3 CLI
    '''
    if recursive:
        recursive = '--recursive'
    else:
        recursive = ''
    cmd = "aws s3 cp {} {} {} --profile {}".format(recursive, source_path, target_path,
                                                   aws_config['local_config_profile_name'])
    logger.info('Running: {}'.format(cmd))
    subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ.copy())
