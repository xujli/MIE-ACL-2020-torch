from torch import nn
import torch.nn.functional as F
import torch
import itertools
import numpy as np
import os

from tqdm import tqdm
from math import ceil

INF = 1e5
DEFAULT_BATCH_SIZE = 1000
MAX_WINDOW_SIZE = 200


class bilstm(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, is_bidirectional, batch_first=True):
        super(bilstm, self).__init__()
        self.batch_first = batch_first
        self.layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, dropout=dropout, num_layers=1,
                             bidirectional=is_bidirectional, batch_first=batch_first).

    def forward(self, input, input_length):
        #input = torch.transpose(input, 1, 2)
        input = nn.utils.rnn.pack_padded_sequence(input, input_length.cpu(), batch_first=self.batch_first, enforce_sorted=False)
        output, (h_n, c_n) = self.layer(input)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, (h_n, c_n)

class gate(nn.Module):
    def __init__(self):
        super(gate, self).__init__()
        self.beta = torch.tensor(0.5, requires_grad=False)
    def forward(self, input1, input2):
        output = self.beta * input1 + (1-self.beta) * input2
        return output

class self_attention(nn.Module):
    def __init__(self, seq_length, dropout):
        super(self_attention, self).__init__()
        self.linear = nn.Linear(in_features=seq_length, out_features=1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input):
        x = self.linear(input)
        point = torch.unsqueeze(torch.sum(x, dim=-1), -1)
        mask = - torch.eq(point, torch.zeros(size=point.shape).cuda()).float() * INF
        x = x + mask
        p = F.softmax(x, dim=1)
        # [num, max_len, 1]
        c = torch.sum(p * input, dim=1)
        # [num, emb_size]
        c = self.dropout(c)
        return c

class feedforward(nn.Module):
    def __init__(self,
                    input_size,
                    num_layers,
                    num_units,
                    outputs_dim,
                    activation,
                    dropout):
        super(feedforward, self).__init__()
        if activation == 'relu':
            act_type = nn.ReLU()
        else:
            act_type = nn.Tanh()
        layers = []
        if num_layers != 0:
            layers += [nn.Linear(
                in_features=input_size,
                out_features=num_units)
            , act_type]
            layers += [nn.Dropout(p=dropout)]
            for i in range(num_layers - 1):
                layers += [nn.Linear(
                    num_units,
                    num_units
                ), act_type]
                layers += [nn.Dropout(p=dropout)]
            layers += [nn.Linear(num_units, outputs_dim)]
        else:
            layers += [nn.Linear(input_size, outputs_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        output = self.model(input)
        return output

class encoder(nn.Module):
    def __init__(self, is_global, seq_length, input_size, hidden_size, dropout, is_bidirectional, batch_first=True):
        super(encoder, self).__init__()
        self.is_global = is_global
        if is_global:
            self.h_s = bilstm(input_size, hidden_size, dropout, is_bidirectional, batch_first)
            self.h_g = bilstm(input_size, hidden_size, dropout, is_bidirectional, batch_first)
            self.h = gate()
            self.c_s = self_attention(seq_length, dropout)
            self.c_g = self_attention(seq_length, dropout)
            self.c = gate()
        else:
            self.h_g = bilstm(input_size, hidden_size, dropout, is_bidirectional, batch_first)
            self.c = self_attention(hidden_size*2, dropout)

    def forward(self, input, input_length):
        if self.is_global:
            h_s, _ = self.h_s(input, input_length)
            h_g, _ = self.h_g(input, input_length)
            h = self.h(h_s, h_g)
            c_s = self.c_s(h)
            c_g = self.c_g(h)
            c = self.c(c_s, c_g)
        else:
            h, _ = self.h_g(input, input_length)
            c = self.c(h)
        return h, c

class U(nn.Module):
    def __init__(self, is_global, seq_length, input_size, hidden_size, dropout, is_bidirectional=True, batch_first=True,
                 window_size=1, max_len=100, num_units=400):
        super(U, self).__init__()
        self.coder_u = encoder(is_global, seq_length, input_size, hidden_size, dropout, is_bidirectional, batch_first)
        self.coder_c = encoder(is_global, seq_length, input_size, hidden_size, dropout, is_bidirectional, batch_first)
        self.window_size = window_size
        self.max_len = max_len
        self.num_units = num_units

    def forward(self, uinput, uinput_length, cinput, cinput_length):
        utt_h, _ = self.coder_u(uinput, uinput_length)
        # [batch_size * window_size, max_len, num_units]
        utt_h = torch.reshape(utt_h, [-1, self.window_size, self.max_len, self.num_units])
        # [batch_size, window_size, max_len, num_units]
        _, candidate_c = self.coder_c(cinput, cinput_length)
        # [slot_value_num, num_units]
        return utt_h, candidate_c

class D(nn.Module):
    def __init__(self, num_units, num_layers,
                    outputs_dim,
                    activation,
                    dropout):
        super(D, self).__init__()
        self.num_units = num_units
        self.weight = torch.empty(size=[1, num_units, num_units],
                dtype=torch.float32, requires_grad=True)
        nn.init.xavier_normal_(self.weight)
        #self.weight = nn.parameter.Parameter(self.weight)
        self.feed_forward = feedforward(2 * num_units, num_layers,
                    num_units,
                    outputs_dim,
                    activation,
                    dropout)

    def _attention(self,
                  query,  # [slot_value_num, num_units]
                  keys,  # [batch_size, window_size, max_len, num_units]
                  values  # [batch_size, window_size, max_len, num_units]
                  ):
        self.batch_size, self.window_size = keys.shape[0], keys.shape[1]
        query = torch.unsqueeze(torch.unsqueeze(query, 0), 0).repeat(self.batch_size, self.window_size, 1, 1)

        # [batch_size, window_size, slot_value_num, num_units]
        p = torch.matmul(
            query,
            torch.transpose(keys, 2, 3)  # [batch_size, window_size, num_units, max_len]
        )  # [batch_size, window_size, slot_value_num, max_len]

        mask = - torch.mul(torch.eq(p, torch.zeros(p.shape).cuda()).float(), INF)
        p = F.softmax(p + mask, dim=-1)

        outputs = torch.matmul(p, values)
        # [batch_size, window_size, slot_value_num, num_units]
        return outputs

    def forward(self, slot_utt_h, slot_candidate_c, status_candidate_c, position_encoding, q_status, mask):
        slot_value_num = slot_candidate_c.shape[0]
        status_num = status_candidate_c.shape[0]

        q_slot = self._attention(slot_candidate_c, slot_utt_h, slot_utt_h) \
                 + position_encoding

        self.weight = self.weight.repeat([self.batch_size, 1, 1]) # [batch_size, num_units, num_units]

        co = torch.reshape(
            torch.reshape(
                torch.matmul(
                    torch.matmul(
                        torch.reshape(
                            q_slot,
                            [self.batch_size, self.window_size * slot_value_num, self.num_units]
                        ),
                        self.weight
                    ),
                    torch.reshape(
                        q_status,
                        [self.batch_size, self.window_size * status_num, self.num_units]
                    ).transpose(1, 2)
                ),  # [batcn_size, window_size * slot_value_num, window_size * status_num]
                [self.batch_size, self.window_size, slot_value_num, self.window_size * status_num]
            ),
            [self.batch_size, self.window_size, slot_value_num, self.window_size, status_num]
        )
        co_mask = - torch.mul(torch.eq(co, torch.zeros(co.shape).cuda()).float(), INF)
        p = co + co_mask

        p = F.softmax(p, 3)
        q_status_slot = torch.unsqueeze(torch.unsqueeze(q_status, -1), -1).repeat(
                [1, 1, 1, 1, self.window_size, slot_value_num])  # [batch_size, window_size, slot_value_num, window_size, status_num, num_units]
        q_status_slot = q_status_slot.transpose(1, 4).transpose(2, 5).transpose(3, 5).transpose(3, 4)

        q_status_slot = torch.sum(torch.mul(
            torch.unsqueeze(p, -1),
            q_status_slot
        ), 3)  # [batch_size, window_size, slot_value_num, status_num, num_units]

        q_slot = torch.unsqueeze(q_slot, 3).repeat([1, 1, 1, status_num, 1])
        features = torch.cat([q_slot, q_status_slot], -1)

        # aggregate
        # [batch_size, window_size, slot_value_num, status_num, 2 * num_units]
        logits = self.feed_forward(features)  # [batch_size, window_size, slot_value_num, status_num, 1]

        logits = torch.reshape(
            logits,
            [-1, self.window_size, slot_value_num * status_num]
        )
        # [batch_size, window_size, slot_value_num * status_num]
        logits = torch.squeeze(logits)
        logits = torch.add(logits, mask)
        #print(logits.shape)
        slot_pred_logits = torch.squeeze(logits, dim=1)
        # [batch_size, slot_value_num * status_num]

        #print(slot_pred_logits.shape)
        # 当前slot的输出标签
        slot_pred_labels = torch.gt(slot_pred_logits, torch.zeros(slot_pred_logits.shape, requires_grad=True).cuda()).float()
        slot_pred_labels.requires_grad = True
        print(slot_pred_labels.grad)
        return slot_pred_labels, slot_value_num * status_num


class MIE:
    def __init__(self, data, ontology, **kw):
        # 初始化data，ontology
        self.data = data
        self.ontology = ontology
        self.slots = [item[0] for item in self.ontology.ontology_list \
                      if item[0] != self.ontology.mutual_slot]
        # self.weights = [len(values) for _, values in self.ontology.ontology_list]

        self.max_len = self.data.max_len
        self.params = kw['params']
        self.device = torch.device('cuda:{}'.format(self.params['gpu_ids'])) if self.params['gpu_ids'] else torch.device('cpu')
        self.dropout = 1 - self.params['keep_p']
        self.window_size = 1
        self.batch_size = self.params['batch_size']
        self.Umodel_names = [item[0] for item in self.ontology.ontology_list] #['D_A', 'D_B', 'D_C', 'D_D']
        self.Dmodel_names = self.slots
        self.num_units = self.params['num_units']
        self.init()
        self.Umodels = {}
        self.Dmodels = {}
        self.opt = {}
        for item in self.Umodel_names:
            self.Umodels[item] = U(is_global=self.params['add_global'], seq_length=self.max_len, input_size=self.data.dictionary.emb_size,
                         hidden_size=int(self.num_units/2), dropout=self.dropout)
            self.Umodels[item].to(self.device)
        for item in self.slots:
            self.Dmodels[item] = D(num_units=self.num_units,
                                   num_layers=self.params['num_layers'], outputs_dim=1, activation='relu', dropout=self.dropout)
            self.Dmodels[item].to(self.device)
        for item in self.Dmodel_names:
            self.opt[item] = torch.optim.Adam(itertools.chain(self.Umodels[self.ontology.mutual_slot].parameters(),
                                                              self.Umodels[item].parameters(), self.Dmodels[item].parameters()), lr=self.params['lr'])

        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        self.criterion1 = nn.BCEWithLogitsLoss(reduction='none')

        for item in self.Dmodel_names:
            print(self.opt[item].param_groups)


    def init(self):
        self.windows_utts = torch.Tensor()
        # [batch_size * window_size]
        self.embedding_weight = torch.Tensor(self.data.dictionary.emb).cuda()
        self.slots_pred_labels = []

    def set_input(self, windows_utts, windows_utts_lens, labels):
        self.windows_utts = torch.tensor(windows_utts).long().to(self.device)
        self.windows_utts_lens = torch.tensor(windows_utts_lens).to(self.device)
        self.labels = labels
        self.slot_utt_hs_dict = dict()
        self.slot_candidate_cs_dict = dict()
        self.candidate_seqs_dict, self.candidate_seqs_lens_dict = self.ontology.onto2ids()
        # to tensor
        for slot in self.candidate_seqs_dict.keys():
            self.candidate_seqs_dict[slot] = torch.tensor(self.candidate_seqs_dict[slot]).long().to(self.device)
            self.candidate_seqs_lens_dict[slot] = torch.tensor(self.candidate_seqs_lens_dict[slot]).to(self.device)

    def _get_embedding(self):
        for slot in self.candidate_seqs_dict.keys():
            self.candidate_seqs_dict[slot] = F.embedding(
                weight=self.embedding_weight,
                input=self.candidate_seqs_dict[slot]
            )
        self.windows_utts_embedding = F.embedding(
            input=self.windows_utts,
            weight=self.embedding_weight
        ) # [batch_size, window_size, max_len, emb_size]
        # dim = tf.reduce_prod(tf.shape(windows_utts[:2]))
        self.utts = torch.reshape(self.windows_utts_embedding, [-1, self.max_len, self.data.dictionary.emb_size])
        self.utts_lens = []
        for i in range(self.utts.shape[0]):
            num = np.sum(np.equal(self.utts.cpu().detach().numpy()[i, :, i], 0))
            self.utts_lens.append(100-num)
        self.utts_lens = torch.tensor(self.utts_lens)


    def _attention(self,
                  query,  # [slot_value_num, num_units]
                  keys,  # [batch_size, window_size, max_len, num_units]
                  values  # [batch_size, window_size, max_len, num_units]
                  ):
        batch_size, window_size = keys.shape[0], keys.shape[1]
        query = torch.unsqueeze(torch.unsqueeze(query, 0), 0).repeat(batch_size, window_size, 1, 1)

        # [batch_size, window_size, slot_value_num, num_units]
        print(query.shape, keys.shape)
        p = torch.matmul(
            query,
            keys.transpose(2, 3)  # [batch_size, window_size, num_units, max_len]
        )  # [batch_size, window_size, slot_value_num, max_len]

        mask = -torch.eq(p, torch.zeros(p.shape).cuda()).float() * INF
        p = F.softmax(p + mask, dim=-1)

        outputs = torch.matmul(p, values)
        # [batch_size, window_size, slot_value_num, num_units]

        return outputs

    def _position_encoding(self):
        num_units = self.params['num_units']
        sin = lambda pos, i: np.sin(pos / (1000 ** (i / num_units)))  # i 是偶数
        cos = lambda pos, i: np.cos(pos / (1000 ** ((i - 1) / num_units)))  # i 是奇数
        PE = [[sin(pos, i) if i % 2 == 0 else cos(pos, i) for i in range(num_units)] \
              for pos in range(MAX_WINDOW_SIZE)]

        PE = torch.tensor(np.array(PE), dtype=torch.float32).cuda()  # [MAX_WINDOW_SIZE, num_units]
        return PE

    def mask_fn(self):
        tmp = torch.unsqueeze(self.windows_utts_lens.cuda(), -1)
        self.mask = -torch.eq(tmp, 0.).float() * INF

        self.position_encoding = self._position_encoding()
        self.position_encoding = torch.unsqueeze(
            torch.unsqueeze(
                self.position_encoding[:self.window_size],
                0
            ),
            2
        )
        self.q_status = self._attention(self.status_candidate_c, self.status_utt_h, self.status_utt_h) \
            + self.position_encoding
        # [batch_size, window_size, status_num, num_units]

    def reinit(self):
        self.infos = dict()
        for dataset in ('train', 'dev'):
            self.infos[dataset] = dict()
            for slot in self.slots:
                self.infos[dataset][slot] = {
                    'ps': [],
                    'rs': [],
                    'f1s' : [],
                    'losses': []
                }
            self.infos[dataset]['global'] = {
                'ps': [],
                'rs': [],
                'f1s' : [],
                'losses': []
            }

    def _evaluate(self, pred_labels, gold_labels):
        def _add_ex_col(x):
            col = 1 - np.sum(x, -1).astype(np.bool).astype(np.float32)
            col = np.expand_dims(col, -1)
            x = np.concatenate([x, col], -1)
            return x
        pred_labels = _add_ex_col(pred_labels)
        gold_labels = _add_ex_col(gold_labels)
        tp = np.sum((pred_labels == gold_labels).astype(np.float32) * pred_labels, -1)
        pred_pos_num = np.sum(pred_labels, -1)
        gold_pos_num = np.sum(gold_labels, -1)
        p = (tp / pred_pos_num)
        r = (tp / gold_pos_num)
        p_add_r = p + r
        p_add_r = p_add_r + (p_add_r == 0).astype(np.float32)
        f1 = 2 * p * r / p_add_r

        return p, r, f1

    def evaluate(self, name, num=-1, batch_size=DEFAULT_BATCH_SIZE):
        slots_pred_labels, slots_gold_labels = \
            self.inference(name, num, batch_size)

        info = dict()
        for slot in self.slots:
            info[slot] = {
                'p': None,
                'r': None,
                'f1': None
            }
        info['global'] = {
            'p': None,
            'r': None,
            'f1': None
        }

        for i, (slot_pred_labels, slot_gold_labels) in \
            enumerate(zip(slots_pred_labels, slots_gold_labels)):
            p, r, f1 = map(
                lambda x: float(np.mean(x)),
                self._evaluate(slot_pred_labels, slot_gold_labels)
            )
            slot = self.slots[i]
            info[slot]['p'] = p
            info[slot]['r'] = r
            info[slot]['f1'] = f1

        pred_labels = np.concatenate(slots_pred_labels, -1)
        gold_labels = np.concatenate(slots_gold_labels, -1)

        p, r, f1 = map(
            lambda x: float(np.mean(x)),
            self._evaluate(pred_labels, gold_labels)
        )
        info['global']['p'] = p
        info['global']['r'] = r
        info['global']['f1'] = f1

        return info

    def inference(self, name, num=-1, batch_size=DEFAULT_BATCH_SIZE):
        if num < 0:
            num = INF
        slots_pred_labels = [[] for i in range(len(self.slots_pred_labels))]
        slots_gold_labels = [[] for i in range(len(self.slots_pred_labels))]
        for i, batch in enumerate(self.data.batch(name, batch_size, False)):
            if (i + 1) * batch_size > num:
                break
            windows_utts_batch, windows_utts_lens_batch, labels_batch = batch
            print(windows_utts_batch.shape, windows_utts_lens_batch.shape, labels_batch.shape)

            self.set_input(windows_utts_batch, windows_utts_lens_batch, labels_batch)
            self._get_embedding()

            for slot in self.Umodel_names:
                utt_h, candidate_c = self.Umodels[slot](self.utts, self.utts_lens, self.candidate_seqs_dict[slot],
                                                        self.candidate_seqs_lens_dict[slot])
                if slot == self.ontology.mutual_slot:
                    self.status_utt_h = utt_h
                    self.status_candidate_c = candidate_c
                else:
                    self.slot_utt_hs_dict[slot] = utt_h
                    self.slot_candidate_cs_dict[slot] = candidate_c
            self.mask_fn()
            start = 0
            for slot in self.Dmodel_names:
                self.opt[slot].zero_grad()
                slot_pred_labels, num = self.Dmodels[slot](self.slot_utt_hs_dict[slot],
                                                           self.slot_candidate_cs_dict[slot],
                                                           self.status_candidate_c, self.position_encoding,
                                                           self.q_status, self.mask)

                end = start + slot_pred_labels.shape[1]
                slots_gold_labels[i].append(labels_batch[:, start: end])
                slots_pred_labels[i].append(slot_pred_labels)
                start = end
        # slots_pred_labels为一个num_slots个元素的列表，每个元素为[num, n * num_statues]
        for i in range(len(slots_gold_labels)):
            slots_gold_labels[i] = np.concatenate(slots_gold_labels[i], 0)
            slots_pred_labels[i] = np.concatenate(slots_pred_labels[i], 0)

        return slots_pred_labels, slots_gold_labels

    def compute_loss(self, name, num=-1, batch_size=DEFAULT_BATCH_SIZE):
        if num < 0:
            num = INF
        slots_loss = [[] for i in range(len(self.slots_loss))]
        for i, batch in enumerate(self.data.batch(name, batch_size, False)):
            if (i + 1) * batch_size > num:
                break
            windows_utts_batch, windows_utts_lens_batch, labels_batch = batch

            self.set_input(windows_utts_batch, windows_utts_lens_batch, labels_batch)
            self._get_embedding()

            for slot in self.Umodel_names:
                utt_h, candidate_c = self.Umodels[slot](self.utts, self.utts_lens, self.candidate_seqs_dict[slot],
                                                        self.candidate_seqs_lens_dict[slot])
                if slot == self.ontology.mutual_slot:
                    self.status_utt_h = utt_h
                    self.status_candidate_c = candidate_c
                else:
                    self.slot_utt_hs_dict[slot] = utt_h
                    self.slot_candidate_cs_dict[slot] = candidate_c
            self.mask_fn()
            start = 0
            for slot in self.Dmodel_names:
                slot_pred_labels, num = self.Dmodels[slot](self.slot_utt_hs_dict[slot],
                                                           self.slot_candidate_cs_dict[slot],
                                                           self.status_candidate_c, self.position_encoding,
                                                           self.q_status, self.mask)

                # print(slot_pred_labels.shape, self.labels.shape)
                slot_gold_labels = torch.tensor(self.labels[:, start: start + num]).float()
                # 当前slot的loss
                slot_loss = self.criterion1(slot_pred_labels,
                                           slot_gold_labels)  # [batch_size, status_num * slot_value_num]

                for i, slot_loss_batch in enumerate(slot_loss):
                    slots_loss[i].append(slot_loss_batch)

        for i in range(len(slots_loss)):
            slots_loss[i] = np.concatenate(slots_loss[i], 0)

        losses = dict([(slot, None) for slot in self.slots])
        losses['global'] = None

        for i, slot_loss in enumerate(slots_loss):
            slot = self.slots[i]
            loss = float(np.mean(slot_loss))
            losses[slot] = loss

        losses['global'] = float(np.mean(np.concatenate(slots_loss, -1)))

        return losses

    def set_requires_grad(self, requires_grad=True):
        for slot in self.Umodel_names:
            for param in self.Umodels[slot].parameters():
                param.requires_grad = requires_grad
        for slot in self.Dmodel_names:
            for param in self.Dmodels[slot].parameters():
                param.requires_grad = requires_grad

    def train(self,
            epoch_num,
            batch_size,
            tbatch_size,
            start_lr,
            end_lr,
            location=None):
        self.set_requires_grad(True)
        # 计算衰减率
        decay = (end_lr / start_lr) ** (1 / epoch_num)
        lr = start_lr

        for i in range(epoch_num):
            pbar = tqdm(
                self.data.batch('train', batch_size, False),
                desc='Epoch {}:'.format(i + 1),
                total=ceil(self.data.datasets['train']['num'] / batch_size)
            )

            for batch in pbar:
                windows_utts_batch, windows_utts_lens_batch, labels_batch = batch

                self.set_input(windows_utts_batch, windows_utts_lens_batch, labels_batch)
                self._get_embedding()

                for slot in self.Umodel_names:
                    utt_h, candidate_c = self.Umodels[slot](self.utts, self.utts_lens, self.candidate_seqs_dict[slot], self.candidate_seqs_lens_dict[slot])
                    if slot == self.ontology.mutual_slot:
                        self.status_utt_h = utt_h
                        self.status_candidate_c = candidate_c
                    else:
                        self.slot_utt_hs_dict[slot] = utt_h
                        self.slot_candidate_cs_dict[slot] = candidate_c
                self.mask_fn()
                start = 0
                self.slots_pred_labels = []
                for slot in self.Dmodel_names:
                    slot_pred_labels, num = self.Dmodels[slot](self.slot_utt_hs_dict[slot], self.slot_candidate_cs_dict[slot],
                                       self.status_candidate_c, self.position_encoding, self.q_status, self.mask)
                    #print(slot_pred_labels.requires_grad)
                    self.slots_pred_labels.append(slot_pred_labels)
                    #print(slot_pred_labels.shape, self.labels.shape)
                    slot_gold_labels = torch.tensor(self.labels[:, start: start + num]).float()
                    # 当前slot的loss
                    slot_loss = self.criterion(slot_pred_labels.cpu(), slot_gold_labels)  # [batch_size, status_num * slot_value_num]
                    #print(slot_loss.requires_grad)
                    #slot_loss = torch.mean(slot_loss)
                    print(slot_loss)
                    self.opt[slot].zero_grad()
                    slot_loss.backward()
                    print(self.opt[slot].param_groups)
                    self.opt[slot].step()
                    start += num

            pbar.close()
            lr *= decay

            train_prf = self.evaluate('train', tbatch_size, tbatch_size)
            train_loss = self.compute_loss('train', tbatch_size, tbatch_size)
            dev_prf = self.evaluate('dev', batch_size=tbatch_size)
            dev_loss = self.compute_loss('dev', batch_size=tbatch_size)

            self._add_infos('train', train_prf)
            self._add_infos('train', train_loss)
            self._add_infos('dev', dev_prf)
            self._add_infos('dev', dev_loss)

            # 打印信息
            print('Epoch {}: train_loss={:.4}, dev_loss={:.4}\
                train_p={:.4}, train_r={:.4}, train_f1={:.4}\
                dev_p={:.4}, dev_r={:.4}, dev_f1={:.4}'.
                format(i + 1, train_loss['global'], dev_loss['global'],
                    train_prf['global']['p'], train_prf['global']['r'],
                    train_prf['global']['f1'], dev_prf['global']['p'],
                    dev_prf['global']['r'], dev_prf['global']['f1']))


    def _add_infos(self, name, info):
        for slot in info.keys():
            if isinstance(info[slot], float):
                # 说明是loss
                self.infos[name][slot]['losses'].append(info[slot])
            elif isinstance(info[slot], dict):
                # 说明是p r f
                for key in info[slot].keys():
                    self.infos[name][slot][key + 's'].append(info[slot][key])

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)
                print('%s loaded' % name)

