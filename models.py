import torch
import torch.nn.functional as F

import math
import numpy as np


class ConvE(torch.nn.Module):
    def __init__(self, num_entities, args):
        super(ConvE, self).__init__()

        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.hidden_drop = torch.nn.Dropout(args.fea_drop)
        self.emb_dim1 = args.conv_embed_shape1
        self.filter_size = args.conv_filter_size
        self.channels = args.conv_channels
        self.padding = 0
        self.stride = 1

        self.emb_dim2 = args.ent_embed_dim // self.emb_dim1
        self.conv1 = torch.nn.Conv2d(in_channels=1,
                                     out_channels=self.channels,
                                     kernel_size=(self.filter_size,
                                                  self.filter_size),
                                     stride=self.stride,
                                     padding=self.padding,
                                     bias=args.conv_bias)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.channels)
        self.bn2 = torch.nn.BatchNorm1d(args.ent_embed_dim)
        self.register_parameter('b', torch.nn.Parameter(
            torch.zeros(num_entities), requires_grad=True))

        conv_output_1 = int(
            ((self.emb_dim1 * 2) - self.filter_size + (2 * self.padding)) / self.stride) + 1
        conv_output_2 = int(
            (self.emb_dim2 - self.filter_size + (2 * self.padding)) / self.stride) + 1
        assert self.filter_size < self.emb_dim2 and self.filter_size < self.emb_dim1
        self.conv_hid_size = self.channels * conv_output_1 * \
            conv_output_2  # as 3x3 filter is used

        self.fc = torch.nn.Linear(self.conv_hid_size, args.ent_embed_dim)
        self.initial_parameters()

    def initial_parameters(self):
        torch.nn.init.kaiming_normal_(tensor=self.conv1.weight.data)
        torch.nn.init.xavier_normal_(tensor=self.fc.weight.data)

    def score_computation(self, e1_emb, rel_emb, all_ent_emb):
        e1_embedded = e1_emb.view(-1, 1, self.emb_dim1, self.emb_dim2)
        rel_embedded = rel_emb.view(-1, 1, self.emb_dim1, self.emb_dim2)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.inp_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, all_ent_emb.transpose(1, 0))
        x += self.b.expand_as(x)
        scores = x
        return scores

    def forward(self, e1_emb, rel_emb, all_ent_emb):
        scores = self.score_computation(
            e1_emb=e1_emb, rel_emb=rel_emb, all_ent_emb=all_ent_emb)
        return scores


class TransConvE(torch.nn.Module):
    def __init__(self, num_entities, args):
        super(TransConvE, self).__init__()
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.hidden_drop = torch.nn.Dropout(args.fea_drop)
        self.filter_size = args.conv_filter_size
        self.padding = int(math.floor(self.filter_size/2))
        self.stride = 1
        self.channels = args.conv_channels
        self.emb_dim = args.ent_embed_dim
        self.conv1 = torch.nn.Conv1d(in_channels=2, out_channels=self.channels, kernel_size=self.filter_size, stride=self.stride,
                                     padding=self.padding, bias=args.conv_bias)
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(self.channels)
        self.bn2 = torch.nn.BatchNorm1d(self.emb_dim)
        self.register_parameter('b', torch.nn.Parameter(
            torch.zeros(num_entities), requires_grad=True))

        conv_output = int((self.emb_dim - self.filter_size +
                          (2 * self.padding)) / self.stride) + 1
        assert self.filter_size < self.emb_dim
        self.conv_hid_size = self.channels * conv_output
        self.fc = torch.nn.Linear(self.conv_hid_size, self.emb_dim)
        self.initial_parameters()

    def initial_parameters(self):
        torch.nn.init.kaiming_uniform_(tensor=self.conv1.weight.data)
        torch.nn.init.xavier_normal_(tensor=self.fc.weight.data)

    def score_computation(self, e1_emb, rel_emb, all_ent_emb):
        e1_embedded = e1_emb.view(-1, 1, self.emb_dim)
        rel_embedded = rel_emb.view(-1, 1, self.emb_dim)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)

        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.inp_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, all_ent_emb.transpose(1, 0))
        x += self.b.expand_as(x)
        scores = x
        return scores

    def forward(self, e1_emb, rel_emb, all_ent_emb):
        scores = self.score_computation(
            e1_emb=e1_emb, rel_emb=rel_emb, all_ent_emb=all_ent_emb)
        return scores


class DistMult(torch.nn.Module):
    def __init__(self, args):
        super(DistMult, self).__init__()
        self.inp_drop = torch.nn.Dropout(args.input_drop)

    def forward(self, e1_emb, rel_emb, all_ent_emb, inverse_rel_emb=None):
        e1_embedded = self.inp_drop(e1_emb)
        rel_embedded = self.inp_drop(rel_emb)
        if inverse_rel_emb is not None:
            inv_rel_embedded = self.inp_drop(inverse_rel_emb)
            comb_rel_embedded = (rel_embedded + inv_rel_embedded) * 0.5
            pred = torch.mm(e1_embedded*comb_rel_embedded,
                            all_ent_emb.transpose(1, 0))
        else:
            pred = torch.mm(e1_embedded*rel_embedded,
                            all_ent_emb.transpose(1, 0))
        return pred


class TuckER(torch.nn.Module):
    def __init__(self, args):
        super(TuckER, self).__init__()

        self.input_dropout = torch.nn.Dropout(args.input_drop)
        self.hidden_dropout = torch.nn.Dropout(args.fea_drop)

        self.ent_emb_dim = args.ent_embed_dim
        self.rel_emb_dim = args.rel_embed_dim
        if args.cuda:
            self.W = torch.nn.Parameter(
                torch.tensor(np.random.uniform(-1, 1, (self.rel_emb_dim, self.ent_emb_dim, self.ent_emb_dim)),
                             dtype=torch.float, device='cuda', requires_grad=True), requires_grad=True)
        else:
            self.W = torch.nn.Parameter(
                torch.tensor(np.random.uniform(-1, 1, (self.rel_emb_dim, self.ent_emb_dim, self.ent_emb_dim)),
                             dtype=torch.float, requires_grad=True), requires_grad=True)

        self.bn0 = torch.nn.BatchNorm1d(self.ent_emb_dim)
        self.bn1 = torch.nn.BatchNorm1d(self.ent_emb_dim)

    def forward(self, e1_emb, rel_emb, all_ent_emb):
        e1 = e1_emb
        x = self.bn0(e1_emb)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        r = rel_emb
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout(W_mat)

        x = torch.bmm(x, W_mat)
        x = x.view(-1, e1.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout(x)
        x = torch.mm(x, all_ent_emb.transpose(1, 0))
        return x


class ComplEx(torch.nn.Module):
    def __init__(self, args):
        super(ComplEx, self).__init__()
        self.ent_emb_dim = args.ent_embed_dim
        self.rel_emb_dim = args.rel_embed_dim
        self.input_dropout = torch.nn.Dropout(args.input_drop)
        self.bn0 = torch.nn.BatchNorm1d(self.ent_emb_dim)
        self.bn1 = torch.nn.BatchNorm1d(self.ent_emb_dim)

    def forward(self, e1_emb, rel_emb, all_ent_emb):
        re_e1_emb, im_e1_emb = e1_emb, e1_emb
        re_rel_emb, im_rel_emb = rel_emb, rel_emb

        re_e1_emb = self.bn0(re_e1_emb)
        re_e1_emb = self.input_dropout(re_e1_emb)
        im_e1_emb = self.bn1(im_e1_emb)
        im_e1_emb = self.input_dropout(im_e1_emb)
        pred = torch.mm(re_e1_emb * re_rel_emb, all_ent_emb.transpose(1, 0)) + \
            torch.mm(re_e1_emb * im_rel_emb, all_ent_emb.transpose(1, 0)) + \
            torch.mm(im_e1_emb * re_rel_emb, all_ent_emb.transpose(1, 0)) - \
            torch.mm(im_e1_emb * im_rel_emb, all_ent_emb.transpose(1, 0))
        pred = torch.sigmoid(pred)
        return pred


class InteractE(torch.nn.Module):
    def __init__(self, args):
        super(InteractE, self).__init__()
        self.ent_emb_dim = args.ent_embed_dim
        self.rel_emb_dim = args.rel_embed_dim
        self.num_filters = args.num_filters
        self.perm = args.perm
        self.kernel_size = args.kernel_size

        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.hidden_drop = torch.nn.Dropout(args.fea_drop)
        self.feature_map_drop = torch.nn.Dropout(args.fea_drop)
        self.bn0 = torch.nn.BatchNorm2d(self.perm)

        self.flat_size_height = args.k_h
        self.flat_size_width = 2 * args.k_w
        self.padding = 0
        self.bn1 = torch.nn.BatchNorm2d(self.num_filters*self.perm)
        self.flat_size = self.flat_size_height * \
            self.flat_size_width * self.num_filters*self.perm
        self.bn2 = torch.nn.BatchNorm1d(self.ent_emb_dim)
        self.fc = torch.nn.Linear(self.flat_size, self.ent_emb_dim)

        self.chequer_perm = self.get_chequer_perm()

        self.register_parameter('conv_filt', torch.nn.Parameter(
            torch.zeros(self.num_filters, 1, self.kernel_size,  self.kernel_size)))
        torch.nn.init.xavier_normal_(self.conv_filt)

    def circular_padding_chw(self, batch, padding):
        upper_pad = batch[..., -padding:, :]
        lower_pad = batch[..., :padding, :]
        temp = torch.cat([upper_pad, batch, lower_pad], dim=2)

        left_pad = temp[..., -padding:]
        right_pad = temp[..., :padding]
        padded = torch.cat([left_pad, temp, right_pad], dim=3)
        return padded

    def get_chequer_perm(self):
        ent_perm = np.int32([np.random.permutation(self.ent_emb_dim)
                            for _ in range(self.perm)])
        rel_perm = np.int32([np.random.permutation(self.ent_emb_dim)
                            for _ in range(self.perm)])
        comb_idx = []
        for k in range(self.perm):
            temp = []
            ent_idx, rel_idx = 0, 0
            for i in range(self.flat_size_height):
                for j in range(self.flat_size_width // 2):
                    if k % 2 == 0:
                        if i % 2 == 0:
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perm[k, rel_idx]+self.ent_emb_dim)
                            rel_idx += 1
                        else:
                            temp.append(rel_perm[k, rel_idx]+self.ent_emb_dim)
                            rel_idx += 1
                    else:
                        if i % 2 == 0:
                            temp.append(rel_perm[k, rel_idx]+self.ent_emb_dim)
                            rel_idx += 1
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                        else:
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perm[k, rel_idx]+self.ent_emb_dim)
                            rel_idx += 1
            comb_idx.append(temp)
        chequer_perm = torch.LongTensor(np.int32(comb_idx)).cuda()
        return chequer_perm

    def score_computation(self, e1_emb, rel_emb, all_ent_emb):
        comb_emb = torch.cat([e1_emb, rel_emb], dim=1)
        chequer_perm = comb_emb[:, self.chequer_perm]
        stack_inp = chequer_perm.reshape(
            (-1, self.perm, self.flat_size_width, self.flat_size_height))
        stack_inp = self.bn0(stack_inp)
        x = self.inp_drop(stack_inp)
        x = self.circular_padding_chw(x, self.kernel_size // 2)
        x = F.conv2d(x, self.conv_filt.repeat(self.perm, 1, 1, 1),
                     padding=self.padding, groups=self.perm)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(-1, self.flat_size)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, all_ent_emb.transpose(1, 0))
        x += self.b.expand_as(x)
        scores = x
        return scores

    def forward(self, e1_emb, rel_emb, all_ent_emb):
        scores = self.score_computation(
            e1_emb=e1_emb, rel_emb=rel_emb, all_ent_emb=all_ent_emb)
        return scores


class HypER(torch.nn.Module):
    def __init__(self, args):
        super(HypER, self).__init__()
        self.ent_emb_dim = args.ent_embed_dim
        self.rel_emb_dim = args.rel_embed_dim

        self.in_channels = args.in_channels
        self.out_channels = args.out_channels
        self.kernel_h = args.kernel_h
        self.kernel_w = args.kernel_w

        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.hidden_drop = torch.nn.Dropout(args.fea_drop)
        self.feature_map_drop = torch.nn.Dropout(args.fea_drop)

        self.bn0 = torch.nn.BatchNorm2d(self.in_channels)
        self.bo1 = torch.nn.BatchNorm2d(self.out_channels)
        self.bn2 = torch.nn.BatchNorm1d(self.ent_emb_dim)
        fc_length = (1-self.kernel_h+1)*(self.ent_emb_dim -
                                         self.kernel_w+1)*self.out_channels
        self.fc = torch.nn.Linear(fc_length, self.ent_emb_dim)
        fc1_length = self.in_channels*self.out_channels*self.kernel_h*self.kernel_w
        self.fc1 = torch.nn.Linear(self.rel_emb_dim, fc1_length)

    def score_computation(self, e1_emb, rel_emb, all_ent_emb):
        e1_emb = e1_emb.view(-1, 1, 1, e1_emb.size(1))
        x = self.bn0(e1_emb)
        x = self.inp_drop(x)

        k = self.fc1(rel_emb)
        k = k.view(-1, self.in_channels, self.out_channels,
                   self.kernel_h, self.kernel_w)
        k = k.view(e1_emb.size(0)*self.in_channels *
                   self.out_channels, 1, self.kernel_h, self.kernel_w)

        x = x.permute(1, 0, 2, 3)

        x = F.conv2d(x, k, groups=e1_emb.size(0))
        x = x.view(e1_emb.size(0), 1, self.out_channels, 1 -
                   self.kernel_h+1, e1_emb.size(3)-self.kernel_w+1)
        x = x.permute(0, 3, 4, 1, 2)
        x = torch.sum(x, dim=3)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.bn1(x)
        x = self.feature_map_drop(x)
        x = x.view(e1_emb.size(0), -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, all_ent_emb.weight.transpose(1, 0))
        return x

    def forward(self, e1_emb, rel_emb, all_ent_emb):
        scores = self.score_computation(
            e1_emb=e1_emb, rel_emb=rel_emb, all_ent_emb=all_ent_emb)
        return scores
