import os
import argparse
import json
import logging
import copy

from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from data_loader import BiTrainDataset, BidirectionalOneShotIterator
from kge_model import KGEModel

from optimizer import WarmupCosineSchedule

from utils import set_seeds


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', default=False,
                        action='store_true', help='use GPU')
    parser.add_argument('--do_train', default=True, action='store_true')
    parser.add_argument('--do_valid', default=True, action='store_true')
    parser.add_argument('--do_test', default=True, action='store_true')
    parser.add_argument('--evaluate_train', action='store_true',
                        help='Evaluate on training data')
    parser.add_argument('--model', default='ConvE', type=str)
    parser.add_argument('--data_path', type=str, default='data/umls')

    parser.add_argument('-d', '--hidden_dim', default=256, type=int)
    parser.add_argument('-ee', '--ent_embed_dim', default=200, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('-er', '--rel_embed_dim', default=200, type=int)
    parser.add_argument('-e', '--embed_dim', default=256, type=int)
    parser.add_argument('-g', '--gamma', default=0.01,
                        type=float, help='smoothing factor')
    parser.add_argument('-b', '--batch_size', default=2, type=int)
    parser.add_argument('--test_batch_size', default=4,
                        type=int, help='valid/test batch size')

    parser.add_argument('-lr', '--learning_rate', default=0.01, type=float)
    parser.add_argument("--adam_weight_decay", type=float,
                        default=3e-8, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float,
                        default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float,
                        default=0.98, help="adam second beta value")
    parser.add_argument('-cpu', '--cpu_num', default=6, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path',
                        default='saved/models/wn18rr', type=str)
    parser.add_argument('--max_steps', default=40000, type=int)
    parser.add_argument('--warm_up_ratio', default=0.1, type=float)
    # For DistMult based warm-up for KG embedding
    parser.add_argument('--warm_up_steps', default=-1, type=int)
    parser.add_argument('--reszero', default=1, type=int)

    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=1000, type=int)
    parser.add_argument('--log_steps', default=10, type=int,
                        help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=2000,
                        type=int, help='valid/test log every xx steps')

    parser.add_argument('--nentity', type=int, default=0,
                        help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0,
                        help='DO NOT MANUALLY SET')
    parser.add_argument('--ntriples', type=int, default=0,
                        help='DO NOT MANUALLY SET')
    parser.add_argument("--seed", type=int, default=1, help="random seed")

    parser.add_argument('--conv_embed_shape1', type=int, default=20, help='The first dimension of the reshaped 2D embedding. '
                        'The second dimension is infered. Default: 20')
    parser.add_argument('--conv_channels', type=int,
                        default=64, help='number of filters')
    parser.add_argument('--conv_filter_size', type=int, default=3)
    parser.add_argument('--conv_bias', default=True, action='store_true',
                        help='Use a bias in the convolutional layer. Default: True')

    parser.add_argument("--input_drop", type=float,
                        default=0.25, help="input feature drop out")
    parser.add_argument("--fea_drop", type=float,
                        default=0.2, help="feature drop out")
    parser.add_argument('--loss_type', type=int, default=1,
                        help="0: onevsall, 1: kvsall")
    parser.add_argument('--patience', type=int, default=30,
                        help="used for early stop")
    return parser.parse_args(args)


def save_model(model, step, model_path):
    if isinstance(model_path, Path):
        model_path = str(model_path)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model_state_dict = model.state_dict()
    for key in model_state_dict:
        model_state_dict[key] = model_state_dict[key].cpu()
    torch.save({
        'step': step,
        'model_state_dict': model_state_dict
    }, model_path)


def save_kge_model(model, args, step, mrr):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''

    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    data_path = args.data_path
    if 'FB' in data_path:
        data_name = 'FB'
    else:
        data_name = 'WN'
    model_name = data_name + '_model_' + args.model + '_lr_' + \
        str(args.learning_rate) + '_step_' + str(step) + '_mrr_' + str(mrr)
    save_model(model, model_path=os.path.join(args.save_path,
               'kge_model' + model_name + '.pt'), step=step)
    return model_name


def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(
            args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(
            args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' %
                     (mode, metric, step, metrics[metric]))


def train_dag_kge_model(args, nentity, nrelation, ntriples, all_true_triples, train_triples, valid_triples, test_triples):
    """
    :param args:
    :param nentity:
    :param nrelation:
    :param all_true_triples:
    :param train_triples:
    :param valid_triples:
    :param test_triples:
    :return:
    """
    # +++++++
    data_path = args.data_path
    min_baseline_mrr = 0.05
    if 'FB' in data_path:
        baseline_mrr = 0.34
    else:
        baseline_mrr = 0.45

    num_train_triples = len(train_triples)
    batch_size = args.batch_size
    step_in_epoch = num_train_triples // batch_size + 1
    max_step = args.max_steps
    if max_step < 400 * step_in_epoch:
        max_step = 400 * step_in_epoch
    args.max_step = max_step

    warm_up_step = args.warm_up_steps
    if warm_up_step > 0:
        warm_up_step = 300 * step_in_epoch
    args.warm_up_steps = warm_up_step

    # +++++++
    kge_model = KGEModel(nentity=nentity, nrelation=nrelation,
                         ntriples=ntriples, args=args)
    if args.cuda:
        kge_model = kge_model.cuda()
    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' %
                     (name, str(param.size()), str(param.requires_grad)))

    logging.info('Ramdomly Initializing {} KGE Model...'.format(args.model))
    init_step = 0
    step = init_step
    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)

    if args.do_train:
        # print('Testing taining size {}'.format(len(train_triples)))
        train_dataloader_head = DataLoader(
            BiTrainDataset(train_triples, nentity, nrelation, 'head-batch'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=BiTrainDataset.collate_fn
        )

        train_dataloader_tail = DataLoader(
            BiTrainDataset(train_triples, nentity, nrelation, 'tail-batch'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=BiTrainDataset.collate_fn
        )

        train_iterator = BidirectionalOneShotIterator(
            train_dataloader_head, train_dataloader_tail)

        logging.info('learning_rate = %f' % args.learning_rate)
        training_logs = []
        if args.warm_up_steps > 0:
            # ++++++++++++++++++++++++++++++++++
            current_learning_rate = args.learning_rate
            if args.adam_weight_decay <= 0:
                adam_weight_decay = 0
            else:
                adam_weight_decay = args.adam_weight_decay
            optimizer = Adam(params=filter(lambda p: p.requires_grad, kge_model.parameters()),
                             lr=current_learning_rate, betas=(
                                 args.adam_beta1, args.adam_beta2),
                             weight_decay=adam_weight_decay)
            scheduler = CosineAnnealingLR(
                optimizer=optimizer, T_max=args.max_steps, eta_min=1e-6)
            # ++++++++++++++++++++++++++++++++++
            for step in range(init_step, args.warm_up_steps):
                log = kge_model.train_step(
                    kge_model, optimizer, train_iterator, args, True)
                scheduler.step()
                training_logs.append(log)
                if step % args.log_steps == 0:
                    metrics = {}
                    for metric in training_logs[0].keys():
                        metrics[metric] = sum(
                            [log[metric] for log in training_logs]) / len(training_logs)
                    log_metrics('Training average', step, metrics)
                    training_logs = []
                if args.do_valid and step % 5000 == 0 and step > 0:
                    logging.info('Evaluating on Valid Dataset...')
                    metrics = kge_model.test_step(
                        kge_model, valid_triples, all_true_triples, args, None, True)
                    log_metrics('Valid', step, metrics)

            if args.do_valid:
                logging.info('Evaluating on Valid Dataset...')
                metrics = kge_model.test_step(
                    kge_model, valid_triples, all_true_triples, args, None,  True)
                log_metrics('Valid', step, metrics)
            if args.do_test:
                logging.info('Evaluating on Test Dataset...')
                metrics = kge_model.test_step(
                    kge_model, test_triples, all_true_triples, args, None, True)
                log_metrics('Test', step, metrics)
        # +++++++++++++++++++
        # +++++++++++++++++++
        init_step = 0
        num_warmup_steps = int(args.max_steps * args.warm_up_ratio)
        current_learning_rate = args.learning_rate
        if args.adam_weight_decay <= 0:
            adam_weight_decay = 0
        else:
            adam_weight_decay = args.adam_weight_decay
        optimizer = Adam(params=filter(lambda p: p.requires_grad, kge_model.parameters()), lr=current_learning_rate,
                         weight_decay=adam_weight_decay)
        scheduler = WarmupCosineSchedule(
            optimizer, warmup_steps=num_warmup_steps, t_total=args.max_steps)  # PyTorch scheduler
        # ++++++++++++++++++++++++++++++++++
        # Training Loop
        best_valid_mrr = 0.0
        best_model_name = ''
        early_stop_step = 0
        best_kge_model = None
        for step in range(init_step, args.max_steps):
            log = kge_model.train_step(
                kge_model, optimizer, train_iterator, args)
            scheduler.step()
            training_logs.append(log)
            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum(
                        [log[metric] for log in training_logs]) / len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []
            if args.do_valid and step % args.valid_steps == 0 and step > 0:
                logging.info('Evaluating on Valid Dataset...')
                metrics = kge_model.test_step(
                    kge_model, valid_triples, all_true_triples, args)
                log_metrics('Valid', step, metrics)
                mrr = metrics['MRR']
                if best_valid_mrr < mrr:
                    best_valid_mrr = mrr
                    best_model_name = save_kge_model(
                        kge_model, args, step, best_valid_mrr)
                    best_kge_model = copy.deepcopy(kge_model)
                if mrr < best_valid_mrr:
                    early_stop_step = early_stop_step + 1
                else:
                    early_stop_step = 0
                if early_stop_step >= args.patience:
                    break

            if best_valid_mrr < baseline_mrr and step >= 0.8 * args.max_steps:
                break

            if best_valid_mrr < min_baseline_mrr and step >= 0.15 * args.max_steps:
                break
            torch.cuda.empty_cache()

    logging.info(
        '\n\nThe best MRR of KG embedding model is {}'.format(best_valid_mrr))
    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics = best_kge_model.test_step(
            best_kge_model, valid_triples, all_true_triples, args)
        log_metrics('Valid', step, metrics)

    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        metrics = best_kge_model.test_step(
            best_kge_model, test_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)
    # save_kge_model(model=best_kge_model, args=args, step='best', mrr=best_valid_mrr)
    logging.info('Best model name" {}'.format(best_model_name))
    return kge_model


def main(args):
    random_seed = args.seed
    set_seeds(random_seed)
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Write logs to checkpoint and console
    set_logger(args)
    logging.info("Model information...")
    for key, value in vars(args).items():
        logging.info('\t{} = {}'.format(key, value))

    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)

    nentity = len(entity2id)
    nrelation = len(relation2id)

    args.nentity = nentity
    args.nrelation = nrelation

    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)

    train_triples = read_triple(os.path.join(
        args.data_path, 'train.txt'), entity2id, relation2id)
    logging.info('#train: %d' % len(train_triples))
    valid_triples = read_triple(os.path.join(
        args.data_path, 'valid.txt'), entity2id, relation2id)
    logging.info('#valid: %d' % len(valid_triples))
    test_triples = read_triple(os.path.join(
        args.data_path, 'test.txt'), entity2id, relation2id)
    logging.info('#test: %d' % len(test_triples))

    args.ntriples = len(train_triples)

    all_true_triples = train_triples + valid_triples + test_triples
    train_dag_kge_model(args, nentity=nentity, nrelation=nrelation, ntriples=len(train_triples),
                        train_triples=train_triples,
                        test_triples=test_triples, valid_triples=valid_triples, all_true_triples=all_true_triples)


if __name__ == '__main__':
    main(parse_args())
