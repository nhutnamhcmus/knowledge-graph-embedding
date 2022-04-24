import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from loss_fcn import CESmoothLossKvsAll, CESmoothLossOnevsAll
from data_loader import BiTrainDataset, TestDataset

import logging


from models import ConvE, TransConvE, DistMult, TuckER, ComplEx


class KGEModel(torch.nn.Module):
    def __init__(self, nentity, nrelation, ntriples, args):
        """
        Support ConvE, distMult and Cross Entropy and Binary Cross Entropy Loss
        :param nentity:
        :param nrelation:
        :param args:
        :param bce_loss:
        """
        super(KGEModel, self).__init__()
        self.model_name = args.model

        self._nentity = nentity
        self._nrelation = nrelation
        self._ntriples = ntriples

        self._ent_emb_size = args.ent_embed_dim
        self._rel_emb_size = args.rel_embed_dim

        self.entity_embedding = torch.nn.Parameter(
            torch.zeros(nentity, self._ent_emb_size), requires_grad=True)
        self.relation_embedding = torch.nn.Parameter(torch.zeros(
            nrelation * 2 + 1, self._rel_emb_size), requires_grad=True)  # inverse realtion + self-loop
        self.inp_drop = torch.nn.Dropout(p=args.input_drop)
        self.feature_drop = torch.nn.Dropout(p=args.fea_drop)

        if self.model_name == 'DistMult':
            self.score_function = DistMult(args=args)
        elif self.model_name == 'ConvE':
            self.score_function = ConvE(num_entities=nentity, args=args)
        elif self.model_name == 'TransConvE':
            self.score_function = TransConvE(num_entities=nentity, args=args)
        elif self.model_name == 'TuckER':
            self.score_function = ComplEx(args=args)
        elif self.model_name == 'ComplEx':
            self.score_function = ComplEx(args=args)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        if args.warm_up_steps > 0:
            self.warm_up_score_function = DistMult(args=args)
        else:
            self.warm_up_score_function = None

        self.loss_type = args.loss_type

        if self.loss_type == 0:
            self.loss_function_onevsall = CESmoothLossOnevsAll(
                smoothing=args.gamma)
        else:
            self.loss_function_onevsall = None

        if self.loss_type == 1:
            self.loss_function_kvsall = CESmoothLossKvsAll(
                smoothing=args.gamma)
        else:
            self.loss_function_kvsall = None

        self.init()

    def init(self):
        torch.nn.init.xavier_uniform_(tensor=self.entity_embedding)
        torch.nn.init.xavier_uniform_(tensor=self.relation_embedding)

    def get_embedding(self):
        entity_embedder, relation_embedder = self.entity_embedding, self.relation_embedding
        return entity_embedder, relation_embedder

    def forward(self, sample, entity_embed, relation_embed, predict_mode, true_labels=None, warm_up=False):
        head_part, rel_part, tail_part, inv_rel_part = sample[:,
                                                              0], sample[:, 1], sample[:, 2], sample[:, 3]
        relation = torch.index_select(relation_embed, dim=0, index=rel_part)
        if predict_mode == 'head-batch':
            ent_embed = torch.index_select(
                entity_embed, dim=0, index=tail_part)
            labels = head_part
        elif predict_mode == 'tail-batch':
            ent_embed = torch.index_select(
                entity_embed, dim=0, index=head_part)
            labels = tail_part
        else:
            raise ValueError('mode %s not supported' % predict_mode)
        if warm_up and self.warm_up_score_function is not None:
            scores = self.warm_up_score_function(
                ent_embed, relation, entity_embed)  # DistMult warmup
        else:
            if self.model_name == 'DistMult':
                scores = self.score_function(
                    ent_embed, relation, entity_embed)  # The score is symetric
            elif self.model_name == 'ConvE' or self.model_name == 'TransConvE':
                # the score function is not symetric
                scores = self.score_function(ent_embed, relation, entity_embed)
            elif self.model_name == 'TuckER':
                scores = self.score_function(ent_embed, relation, entity_embed)
            elif self.model_name == 'ComplEx':
                scores = self.score_function(ent_embed, relation, entity_embed)
            else:
                raise ValueError('model %s not supported' % self.model_name)
        if self.training:
            if self.loss_type == 0:
                loss = self.loss_function_onevsall(scores, labels)
            elif self.loss_type == 1:
                loss = self.loss_function_kvsall(scores, true_labels)
            else:
                ValueError('loss %s not supported' % self.loss_type)
            return loss
        else:
            return scores

    @staticmethod
    def train_step(model, optimizer, train_iterator, args, warm_up=False):
        model.train()
        optimizer.zero_grad()
        samples, true_labels, edge_ids, mode = next(train_iterator)

        if args.cuda:
            samples = samples.cuda()
            edge_ids = edge_ids.cuda()
            true_labels = true_labels.cuda()

        entity_embedder, relation_embedder = model.get_embedding()
        loss = model(samples, entity_embedder, relation_embedder,
                     predict_mode=mode, true_labels=true_labels, warm_up=warm_up)
        loss.backward()
        optimizer.step()
        log = {
            'loss': loss.item()
        }
        return log

    @staticmethod
    def test_step(model, test_triples, all_true_triples, args, load_mode=None, warm_up=False):
        '''
                Evaluate the model on test or valid datasets
        '''
        model.eval()
        # Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
        # Prepare dataloader for evaluation
        test_dataloader_head = DataLoader(
            TestDataset(
                test_triples,
                all_true_triples,
                args.nentity,
                args.nrelation,
                'head-batch'
            ),
            batch_size=args.test_batch_size,
            num_workers=1,
            collate_fn=TestDataset.collate_fn
        )

        test_dataloader_tail = DataLoader(
            TestDataset(
                test_triples,
                all_true_triples,
                args.nentity,
                args.nrelation,
                'tail-batch'
            ),
            batch_size=args.test_batch_size,
            num_workers=1,
            collate_fn=TestDataset.collate_fn
        )

        if load_mode is not None:
            if load_mode == 'head-batch':
                test_dataset_list = [test_dataloader_head]
            else:
                test_dataset_list = [test_dataloader_tail]
        else:
            test_dataset_list = [test_dataloader_head, test_dataloader_tail]

        logs = []

        step = 0
        # total_steps = sum([len(dataset) for dataset in test_dataset_list])

        with torch.no_grad():
            entity_embedder, relation_embedder = model.get_embedding()
            for test_dataset in test_dataset_list:
                # for positive_sample, _, filter_bias, mode in test_dataset:
                for positive_sample, filter_bias, mode in test_dataset:
                    if args.cuda:
                        positive_sample = positive_sample.cuda()
                        filter_bias = filter_bias.cuda()

                    batch_size = positive_sample.size(0)
                    score = model(positive_sample, entity_embedder, relation_embedder,
                                  predict_mode=mode, true_labels=None, warm_up=warm_up)
                    score = torch.sigmoid(score)
                    score += filter_bias

                    # Explicitly sort all the entities to ensure that there is no test exposure bias
                    argsort = torch.argsort(score, dim=1, descending=True)

                    if mode == 'head-batch':
                        positive_arg = positive_sample[:, 0]
                    elif mode == 'tail-batch':
                        positive_arg = positive_sample[:, 2]
                    else:
                        raise ValueError('mode %s not supported' % mode)

                    for i in range(batch_size):
                        # Notice that argsort is not ranking
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                        assert ranking.size(0) == 1

                        # ranking + 1 is the true ranking used in evaluation metrics
                        ranking = 1 + ranking.item()
                        logs.append({
                            'MRR': 1.0 / ranking,
                            'MR': float(ranking),
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                        })

                    if step % args.test_log_steps == 0:
                        # logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))
                        logging.info('Evaluating the model... (%d)' % (step))

                    step += 1
                    torch.cuda.empty_cache()

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

        return metrics
