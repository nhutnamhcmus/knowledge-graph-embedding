import torch
import torch.nn.functional as F

import math


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
