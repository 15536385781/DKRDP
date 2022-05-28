# -*- coding: gbk -*-
import os
import time
from sklearn.metrics import roc_auc_score
from termcolor import colored
import mimic_data_split
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
use_cuda = torch.cuda.is_available()
import math
from sklearn.model_selection import train_test_split
import pandas as pd
import random
import numpy as np
import transformers.models.bert.modeling_bert
from transformers import BertForSequenceClassification,BertConfig,AdamW,get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score,average_precision_score,recall_score,f1_score

def read_feature_and_label(info_folder = os.path.abspath('./mimic_five_part_five_fold/')):

    all_treatment = list()
    all_label = list()
    all_disease = list()
    all_risk_factor = list()
    all_category = list()
    all_id = list()

    for i in range(5):
        label = np.load(os.path.join(info_folder, 'label_list_{}.npy'.format(i))).tolist()
        disease = np.load(os.path.join(info_folder, 'disease_list_{}.npy'.format(i))).tolist()
        risk_factor = np.load(os.path.join(info_folder, 'risk_factor_list_{}.npy'.format(i))).tolist()
        category = np.load(os.path.join(info_folder, 'disease_category_list_{}.npy'.format(i))).tolist()
        id = np.load(os.path.join(info_folder, 'pat_visit_list_{}.npy'.format(i))).tolist()
        treatment = np.load(os.path.join(info_folder, 'treatment_list_{}.npy'.format(i))).tolist()
        all_label += label
        all_disease += disease
        all_risk_factor += risk_factor
        all_category += category
        all_id += id
        all_treatment += treatment


    return all_label, all_disease, all_risk_factor, all_category, all_id, all_treatment

class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example




def convert_EHRexamples_to_features(examples, max_seq_length):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        feature = convert_singleEHR_example(ex_index, example, max_seq_length)
        features.append(feature)
    return features


### This is the EHR version

def convert_singleEHR_example(ex_index, example, max_seq_length):
    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    input_ids = example[2]
    segment_ids = example[3]
    label_id = example[1]

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # LR 5/13 Left Truncate longer sequence
    while len(input_ids) > max_seq_length:
        input_ids = input_ids[-max_seq_length:]
        input_mask = input_mask[-max_seq_length:]
        segment_ids = segment_ids[-max_seq_length:]

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    feature = [input_ids, input_mask, segment_ids, label_id, True]
    return feature

def train_valid_test_split(total_data):
    '''
    6:2:1:1
    :param total_data:
    :return:
    '''
    train_data = total_data[:int(len(total_data)*3/5)] #[:6416] 6416
    valid_data = total_data[int(len(total_data)*3/5):int(len(total_data)*4/5)] #[6416:8555] 2139
    test_data = total_data[int(len(total_data)*4/5):int(len(total_data)*9/10)] #[8555:9624] 1069
    test_data2 = total_data[int(len(total_data)*9/10):len(total_data)] #[9624:10694] 1070
    return train_data,valid_data,test_data,test_data2


class BERTdataEHR(Dataset):
    def __init__(self, Features):
        self.data = Features

    def __getitem__(self, idx, seeDescription=False):
        sample = self.data[idx]

        return sample

    def __len__(self):
        return len(self.data)


# customized parts for EHRdataloader
def my_collate(batch):
    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in batch:
        all_input_ids.append(feature[0])
        all_input_mask.append(feature[1])
        all_segment_ids.append(feature[2])
        all_label_ids.append(feature[3])
    return [all_input_ids, all_input_mask, all_segment_ids, all_label_ids]


class BERTdataEHRloader(DataLoader):
    def __init__(self, dataset, batch_size=128, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, collate_fn=my_collate, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        DataLoader.__init__(self, dataset, batch_size=batch_size, shuffle=False, sampler=None, batch_sampler=None,
                            num_workers=0, collate_fn=my_collate, pin_memory=False, drop_last=False,
                            timeout=0, worker_init_fn=None)
        self.collate_fn = collate_fn


# Model Definition
class EHR_BERT_LR(nn.Module):
    def __init__(self, input_size, embed_dim, hidden_size, config, n_layers=1, dropout_r=0.5, cell_type='LSTM', bi=False,
                 time=False, preTrainEmb=''):
        super(EHR_BERT_LR, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.dropout_r = dropout_r
        self.cell_type = cell_type
        self.preTrainEmb = preTrainEmb
        self.time = time
        self.config = config
        
        if bi:
            self.bi = 2
        else:
            self.bi = 1

        self.PreBERTmodel = BertForSequenceClassification.from_pretrained('/home/liangyz/pretrain model/bert-base-uncased/', )
        
        if use_cuda:
            self.PreBERTmodel.cuda()
        input_size = self.PreBERTmodel.bert.config.vocab_size
        self.in_size = self.PreBERTmodel.bert.config.hidden_size

        self.dropout = nn.Dropout(p=self.dropout_r)
        self.out = nn.Linear(self.in_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        if use_cuda:
            self.flt_typ = torch.cuda.FloatTensor
            self.lnt_typ = torch.cuda.LongTensor
        else:
            self.lnt_typ = torch.LongTensor
            self.flt_typ = torch.FloatTensor

    def forward(self, sequence):

        #for i in range(len(sequence[0])):
        #    for j in range(len(sequence[0][i])):
        #        print('sequence[0]'+str(i)+str(j)+'/'+str(sequence[0][i][j]))
        token_t = torch.from_numpy(np.asarray(sequence[0], dtype=float)).type(self.lnt_typ)
        seg_t = torch.from_numpy(np.asarray(sequence[2], dtype=int)).type(self.lnt_typ)

        # for i in range(len(sequence[3])):
        #     for j in range(len(sequence[3][i])):
        #         sequence[3][i][j] = float(sequence[3][i][j])

        Label_t = torch.from_numpy(np.asarray(sequence[3], dtype=float)).type(self.lnt_typ)
        # Bert_out = self.PreBERTmodel.bert(input_ids=token_t,
        #                                   attention_mask=torch.from_numpy(np.asarray(sequence[1], dtype=int)).type(
        #                                       self.lnt_typ),
        #                                   token_type_ids=seg_t)
        #print(token_t[0])
        Bert_out = self.PreBERTmodel.bert(input_ids=token_t,
                                          attention_mask=torch.from_numpy(np.asarray(sequence[1], dtype=int)).type(
                                              self.lnt_typ),
                                          token_type_ids=None)
        

        output = self.sigmoid(self.out(Bert_out[1]))
        return output.squeeze(), Label_t.type(self.flt_typ)


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# class FocalLoss(nn.Module):
#     def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
#         super(FocalLoss, self).__init__()
#         if alpha is None:
#             self.alpha = torch.Tensor(torch.ones(class_num, 1))
#         else:
#             self.alpha = torch.Tensor(alpha)
#         self.gamma = gamma
#         self.class_num = class_num
#         self.size_average = size_average
#
#     def forward(self, output, target):
#         output = output.float()
#         target = target.float()
#         output = torch.sigmoid(output)
#         y_t = output * target + (1 - output) * (1 - target)
#         ce = -torch.log(y_t)
#         weight = torch.pow((1 - y_t), self.gamma)
#
#         if output.is_cuda and not self.alpha.is_cuda:
#             self.alpha = self.alpha.cuda()
#         batch_loss = torch.matmul((weight * ce), self.alpha)
#         if self.size_average:
#             loss = batch_loss.mean()
#         else:
#             loss = batch_loss.sum()
#         return loss

torch.set_printoptions(precision=4, sci_mode=False, linewidth=150)

def focal_binary_cross_entropy(logits, targets, gamma=2):
    l = logits.reshape(-1)
    t = targets.reshape(-1)
    p = torch.sigmoid(l)
    p = torch.where(t >= 0.5, p, 1-p)
    logp = - torch.log(torch.clamp(p, 1e-4, 1-1e-4))
    loss = logp*((1-p)**gamma)
    loss = 1*loss.mean()
    return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

def trainsample(sample, model, optimizer,scheduler, criterion=nn.BCELoss()):
    model.train()
    model.zero_grad()
    output, label_tensor = model(sample)
    loss = criterion(output, label_tensor)
    loss.backward()
    optimizer.step()
    if scheduler:
        scheduler.step()
    return output, loss.item()


# train with loaders

def trainbatches(mbs_list, model, optimizer, scheduler, shuffle=True):
    current_loss = 0
    all_losses = []
    plot_every = 5
    n_iter = 0
    if shuffle:
        random.shuffle(mbs_list)
    for i, batch in enumerate(mbs_list):
        output, loss = trainsample(batch, model, optimizer,scheduler,)
        current_loss += loss
        n_iter += 1
        # print('current_loss:'+str(current_loss))
        if n_iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0
    return current_loss, all_losses


def calculate_auc(model, mbs_list, shuffle=True):
    model.eval()
    y_real = []
    y_hat = []
    if shuffle:
        random.shuffle(mbs_list)
    for i, batch in enumerate(mbs_list):
        output, label_tensor = model(batch)
        
        y_hat.extend(output.cpu().data.view(-1).numpy())
        y_real.extend(label_tensor.cpu().data.view(-1).numpy())
    # prob = np.array(y_hat).transpose()
    # pred = np.array(y_hat).transpose() > 0.5
    # true = np.array(y_real).transpose()
    y_pred = (np.array(y_hat) > 0.5).astype(int).tolist()
    # print(y_real, y_pred)
    auc = roc_auc_score(y_real, y_hat)
    acc = accuracy_score(y_real, y_pred)
    recall = recall_score(y_real, y_pred)
    f1 = f1_score(y_real, y_pred)
    return auc, acc, recall, f1

def get_ids(input_list):
    input_ids = []
    for i in range(len(input_list)):
        if int(float(input_list[i])) == 0:
            continue
        else:
            input_ids.append(i)
    return input_ids

# define the final epochs running, use the different names

def add_token(input_id, predict):
    new_input = []
    new_input += input_id
    k = 0
    for i in range(len(input_id)):
        for j in range(len(predict)):
            # if input_id[i] == predict[j] or input_id[i] == 57 or input_id[i] == 45:
            if input_id[i] == predict[j] or input_id[i] in [3,49,47,48,41,46,42,29,21]:
            # if input_id[i] in [1,6,4,5,22,23,25,45,46,41,43,52,3,28,33,27,30,24,26,29,40,55,34,53,17,19,21,54]:
                new_input.insert(k, 150)
                new_input.insert(k + 2, 151)
                k += 2
                break
        k += 1
    return new_input


def epochs_run(epochs, train, valid, test, test1, test2, model, optimizer, scheduler, shuffle=True, patience=20, output_dir='../models/',
               model_prefix='dhf.train', model_customed=''):
    bestValidAuc = 0.0
    bestTestAuc1 = 0.0
    bestTestAuc2 = 0.0
    bestValidEpoch = 0
    # header = 'BestValidAUC|TestAUC|atEpoch'
    # logFile = output_dir + model_prefix + model_customed +'EHRmodel.log'
    # print2file(header, logFile)
    # writer = SummaryWriter(output_dir+'/tsb_runs/') ## LR added 9/27 for tensorboard integration
    for ep in range(epochs):
        print(ep)
        start = time.time()
        current_loss, train_loss = trainbatches(mbs_list=train, model=model, optimizer=optimizer,scheduler =scheduler)
        train_time = timeSince(start)
        # epoch_loss.append(train_loss)
        avg_loss = np.mean(train_loss)
        # writer.add_scalar('Loss/train', avg_loss, ep) ## LR added 9/27
        valid_start = time.time()
        train_auc, _, _, _ = calculate_auc(model=model, mbs_list=train, shuffle=shuffle)
        valid_auc, _, _, _ = calculate_auc(model=model, mbs_list=valid, shuffle=shuffle)
        test_auc, acc, recall, f1 = calculate_auc(model=model, mbs_list=test2, shuffle=shuffle)
        valid_time = timeSince(valid_start)
        # writer.add_scalar('train_auc', train_auc, ep) ## LR added 9/27
        # writer.add_scalar('valid_auc', valid_auc, ep) ## LR added 9/27
        #print(colored(
        #    '\n Epoch (%s): Train_auc (%s), Valid_auc (%s) ,Training Average_loss (%s), Train_time (%s), Eval_time (%s)' % (
        #    ep, train_auc, valid_auc, avg_loss, train_time, valid_time), 'green'))
        print(colored(
            '\n Epoch (%s): Train_auc (%s), Valid_auc (%s) ,Training Average_loss (%s), Train_time (%s), Eval_time (%s),test Auc(%s) , test Acc (%s),recall (%s),test f1(%s)' % (
            ep, train_auc, valid_auc, avg_loss, train_time, valid_time,test_auc, acc, recall, f1), 'green'))
        if valid_auc > bestValidAuc:
            bestValidAuc = valid_auc
            bestValidEpoch = ep
            best_model = model
            bestTrainAuc = train_auc
            if test:
                testeval_start = time.time()
                bestTestAuc1, _, _, _ = calculate_auc(model=best_model, mbs_list=test1, shuffle=shuffle)
                bestTestAuc2, _, _, _ = calculate_auc(model=best_model, mbs_list=test2, shuffle=shuffle)

                # writer.add_scalar('test_auc', valid_auc, ep) ## LR added 9/27
                print(colored('\n Test_AUC1 (%s) ,Test_AUC2 (%s) , Test_eval_time (%s) ' % (
                bestTestAuc1, bestTestAuc2, timeSince(testeval_start)), 'yellow'))
                # print(best_model,model) ## to verify that the hyperparameters already impacting the model definition
                # print(optimizer)
        if ep - bestValidEpoch > patience:
            break

    # writer.close()
    # if not os.path.exists(output_dir):
    #    os.makedirs(output_dir)
    ###save model & parameters
    # torch.save(best_model, output_dir + model_prefix + model_customed + 'EHRmodel.pth')
    # torch.save(best_model.state_dict(), output_dir + model_prefix + model_customed + 'EHRmodel.st')

    if test:
        print(colored('BestValidAuc %f has a TestAuc of %f at epoch %d ' % (bestValidAuc, bestTestAuc1, bestValidEpoch),
                      'green'))
        return bestTrainAuc, bestValidAuc, bestTestAuc1, bestTestAuc2, bestValidEpoch
    else:
        print(colored('BestValidAuc %f at epoch %d ' % (bestValidAuc, bestValidEpoch), 'green'))
        print('No Test Accuracy')

    print(colored('Details see ../models/%sEHRmodel.log' % (model_prefix + model_customed), 'green'))


def main():

    MAX_SEQ_LENGTH = 256
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-5
    bert_config_file = "config.json"
    config = BertConfig.from_json_file(bert_config_file)
    # --------------------------------------load data start-------------------------------------------------------------------
    file_path = os.path.abspath('./mimic_imputed_data.csv')

    # feature_list:(10694,35), label_list:(10694,53), pat_visit_list:(10694,2)..
    all_label, all_disease, all_risk_factor, all_category, all_id, all_treatment = read_feature_and_label()
    predicts = np.load('./predicts7.npy', allow_pickle=True)
    predicts = predicts.tolist()


    total_data = []    # a list of single data [10694, single data]
    for i in range(len(predicts)):
        single_data = [] # [pt_id,label_list, seq_list , segment_list] [1, 53, 127, 127]
        single_data.append(all_id[i]) # pt_id(int)
        single_data.append(all_label[i][3]) # label_list(list)
        input_list = all_disease[i] + all_risk_factor[i] + all_category[i] + all_treatment[i]
        input_list = get_ids(input_list)
        predict_list = predicts[i]
        new_input_list = add_token(input_list,predict_list)
        single_data.append(new_input_list + predict_list) #seq_list(list)
        single_data.append([0] * len(new_input_list) +[1]*len(predict_list)) # segment_list(list)
        total_data.append(single_data)
    # total_data = total_data[:100]
    # print(len(total_data), len(total_data[0][1]), len(total_data[0][2]))
    # --------------------------------------load data end-------------------------------------------------------------------
    # --------------------------------------data transfer start-------------------------------------------------------------
    train_data,valid_data,test_data,test_data2 = train_valid_test_split(total_data)
    train_features = convert_EHRexamples_to_features(train_data, MAX_SEQ_LENGTH)
    test_features = convert_EHRexamples_to_features(test_data, MAX_SEQ_LENGTH)
    test_features2 = convert_EHRexamples_to_features(test_data2, MAX_SEQ_LENGTH)
    valid_features = convert_EHRexamples_to_features(valid_data, MAX_SEQ_LENGTH)
    train = BERTdataEHR(train_features)
    test = BERTdataEHR(test_features)
    test2 = BERTdataEHR(test_features2)
    valid = BERTdataEHR(valid_features)
    print(' creating the list of training minibatches')
    train_mbs = list(BERTdataEHRloader(train, batch_size=BATCH_SIZE))
    print(' creating the list of test minibatches')
    test_mbs = list(BERTdataEHRloader(test, batch_size=BATCH_SIZE))
    print(' creating the list of test2 minibatches')
    test_mbs2 = list(BERTdataEHRloader(test2, batch_size=BATCH_SIZE))
    print(' creating the list of valid minibatches')
    valid_mbs = list(BERTdataEHRloader(valid, batch_size=BATCH_SIZE))
    # --------------------------------------data transfer end-------------------------------------------------------------------


    results = []
    for run in range(10):  ### to average the results on 10 runs
        for model_type in ['Bert only']:
            ehr_model = EHR_BERT_LR(input_size=90000, embed_dim=192, hidden_size=192, config = config)
            if use_cuda:
                ehr_model.cuda()
            # optimizer = optim.Adam(ehr_model.parameters(), lr=LEARNING_RATE)
            optimizer = AdamW(ehr_model.parameters(),
                              lr=2e-5, # args.learning_rate - default is 5e-5
                              eps=1e-8  # args.adam_epsilon  - default is 1e-8
                              )
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=0,
                                                        num_training_steps=53450)
            out_dir_name = 'test_LR_Bert_BiGRU_FC'  # + str(i)
            trauc, vauc, testauc1, testauc2, bep = epochs_run(500, train=train_mbs,
                                                              valid=valid_mbs,
                                                              test=test,
                                                              test1=test_mbs, test2=test_mbs2,
                                                              model=ehr_model,
                                                              optimizer=optimizer,
                                                              scheduler=scheduler,
                                                              shuffle=True,
                                                              # batch_size = args.batch_size,
                                                              patience=5,
                                                              output_dir=out_dir_name,
                                                              model_prefix='first_run')
            results.append(
                [model_type, run, len(train_features), len(test_features), len(valid_features), trauc, vauc, testauc1,
                 testauc2, bep])



if __name__ == '__main__':
    print(use_cuda)
    main()