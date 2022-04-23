import numpy as np
import torch

from torch.utils.data import Dataset


class BiTrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, mode):
        self.len = len(triples)
        self.triples = triples
        self.edge_id_triples = list(enumerate(triples))
        self.nentity = nentity
        self.mode = mode
        self.nrelation = nrelation
        self.true_head, self.true_tail = self.get_true_head_and_tail(
            self.triples, nentity, nrelation)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        e_id, positive_sample = self.edge_id_triples[idx]
        head, relation, tail = positive_sample

        if self.mode == 'head-batch':
            bi_labels = self.true_head[(relation, tail)]
            r_bi_labels = self.true_head[(relation + self.nrelation, head)]
        elif self.mode == 'tail-batch':
            bi_labels = self.true_tail[(head, relation)]
            r_bi_labels = self.true_tail[(tail, relation + self.nrelation)]
        else:
            raise ValueError(
                'Training batch mode %s not supported' % self.mode)

        labels = np.concatenate((bi_labels, r_bi_labels),
                                axis=0).reshape(2, self.nentity)
        labels = torch.from_numpy(labels).float()
        positive_sample = (head, relation, tail, self.nrelation + relation)
        r_positive_sample = (tail, relation + self.nrelation, head, relation)

        positive_sample = torch.LongTensor(
            [positive_sample, r_positive_sample])
        edge_ids = torch.LongTensor([e_id, e_id + self.len])
        return positive_sample, labels, edge_ids, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.cat([_[0] for _ in data], dim=0)
        bi_labels = torch.cat([_[1] for _ in data], dim=0)
        edge_ids = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, bi_labels, edge_ids, mode

    @staticmethod
    def get_true_head_and_tail(triples, nentity, nrelation):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''

        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)

            if (tail, relation + nrelation) not in true_tail:  # reverse edges
                true_tail[(tail, relation + nrelation)] = []
            true_tail[(tail, relation + nrelation)].append(head)

            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

            if (relation + nrelation, head) not in true_head:
                true_head[(relation + nrelation, head)] = []
            true_head[(relation + nrelation, head)].append(tail)

        def binary_labels(label_idxes):
            bi_labels = np.zeros(nentity, dtype=np.float)
            bi_labels[label_idxes] = 1
            return bi_labels

        for relation, tail in true_head:
            # true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
            true_head[(relation, tail)] = binary_labels(
                list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            # true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))
            true_tail[(head, relation)] = binary_labels(
                list(set(true_tail[(head, relation)])))

        return true_head, true_tail


class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        # if self.mode == 'head-batch':
        #     tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
        #            else (-1, head) for rand_head in range(self.nentity)]
        #     tmp[head] = (0, head)
        # elif self.mode == 'tail-batch':
        #     tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
        #            else (-1, tail) for rand_tail in range(self.nentity)]
        #     tmp[tail] = (0, tail)
        # else:
        #     raise ValueError('negative batch mode %s not supported' % self.mode)

        # tmp = torch.LongTensor(tmp)
        # filter_bias = tmp[:, 0].float()
        # negative_sample = tmp[:, 1]

        # positive_sample = torch.LongTensor((head, relation, tail, self.nrelation + relation))
        #
        # return positive_sample, negative_sample, filter_bias, self.mode

        if self.mode == 'head-batch':
            tmp = [0 if (rand_head, relation, tail) not in self.triple_set
                   else -1 for rand_head in range(self.nentity)]
            tmp[head] = 0
        elif self.mode == 'tail-batch':
            tmp = [0 if (head, relation, rand_tail) not in self.triple_set
                   else -1 for rand_tail in range(self.nentity)]
            tmp[tail] = 0
        else:
            raise ValueError(
                'negative batch mode %s not supported' % self.mode)

        tmp = torch.LongTensor(tmp)
        filter_bias = tmp.float()
        positive_sample = torch.LongTensor(
            (head, relation, tail, self.nrelation + relation))

        return positive_sample, filter_bias, self.mode

    @staticmethod
    def collate_fn(data):
        # positive_sample = torch.stack([_[0] for _ in data], dim=0)
        # negative_sample = torch.stack([_[1] for _ in data], dim=0)
        # filter_bias = torch.stack([_[2] for _ in data], dim=0)
        # mode = data[0][3]
        # return positive_sample, negative_sample, filter_bias, mode
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        filter_bias = torch.stack([_[1] for _ in data], dim=0)
        mode = data[0][2]
        return positive_sample, filter_bias, mode


class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data
