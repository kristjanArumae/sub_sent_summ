import torch
import json
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from pytorch_pretrained_bert import BertAdam
import train_args as parse
from m_io import create_output_name, create_valid_rouge, get_valid_evaluation, create_metric_figure, create_metric_eval
from model import CustomNetwork

from tqdm import tqdm, trange

import numpy as np
import os


def create_iterator(data_split='train', max_len=45, max_size=-1, batch_size=32, balance=None, bert_model='bert-large-uncased', use_posit=True):
    bal_str = ''

    if balance is not None and data_split == 'train':  # do not balance test or valid
        bal_str = '_balance_' + str(balance).replace('.', '_') + '_'

    ifp = open('data.nosync/' + data_split + '/' + bert_model + '_cnndm_labeled_tokenized' + bal_str + '.json', 'rb')

    data = json.load(ifp)

    ifp.close()

    x_ls, y_ls, s_idx_ls, b_id_ls, rouge_dict, x_for_rouge, x_align = data['x'], data['y'], data['s_id'], data['b_id'], data[
        'rouge'], data['x_orig'], data['x_align']

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_start_positions = []
    all_end_positions = []
    all_sent_labels = []
    all_sent_align = []
    batch_id_list = []

    num_t = 0
    for (x, _), (label, start, end), s_id, b_id, x_a in zip(x_ls, y_ls, s_idx_ls, b_id_ls, x_align):

        if start >= max_len or label == 0:
            label = 0
            start = max_len
            end = max_len

        if end > max_len:
            end = max_len - 1

        all_sent_labels.append(label)

        all_start_positions.append(start)
        all_end_positions.append(end)

        mask = [1] * len(x)
        padding_mask = [0] * (max_len - len(x))

        mask.extend(padding_mask)
        x.extend(padding_mask)

        all_input_ids.append(x[:max_len])
        all_input_mask.append(mask[:max_len])

        if use_posit:
            segment_id = [s_id] * max_len
        else:
            segment_id = [0] * max_len

        all_segment_ids.append(segment_id[:max_len])
        batch_id_list.append(b_id)
        all_sent_align.append(x_a)

        num_t += 1

        if num_t == max_size:
            break

    tensor_data = TensorDataset(torch.tensor(all_input_ids, dtype=torch.long),
                                torch.tensor(all_input_mask, dtype=torch.long),
                                torch.tensor(all_start_positions, dtype=torch.long),
                                torch.tensor(all_end_positions, dtype=torch.long),
                                torch.tensor(all_sent_labels, dtype=torch.long),
                                torch.tensor(all_segment_ids, dtype=torch.long))

    if data_split == 'train':
        sampler = RandomSampler(tensor_data)
    else:
        sampler = None

    data_loader = DataLoader(tensor_data, sampler=sampler, batch_size=batch_size)

    return data_loader, num_t, batch_id_list, x_for_rouge, all_sent_align


def train(model, loader_train, loader_valid, num_train_epochs=70, x_for_rouge=None, x_sent_align=None, optim='adam', learning_rate=3e-5, unchanged_limit=20, weights=None, ofp_fname='PLT', batch_ids=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    rouge_sys_sent_path = 'data.nosync/rouge_sent/' + ofp_fname + '/'
    rouge_sys_segs_path = 'data.nosync/rouge_segs/' + ofp_fname + '/'

    output_model_file = 'saved_models/' + ofp_fname
    output_config_file = 'saved_configs/' + ofp_fname

    if not os.path.exists(rouge_sys_sent_path):
        os.mkdir(rouge_sys_sent_path)
    if not os.path.exists(rouge_sys_segs_path):
        os.mkdir(rouge_sys_segs_path)

    if not os.path.exists('saved_models'):
        os.mkdir('saved_models')
    if not os.path.exists('saved_configs'):
        os.mkdir('saved_configs')

    if optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.01)
    else:
        optimizer = BertAdam(model.parameters(), lr=learning_rate)

    model.train()

    loss_ls, loss_ls_s, loss_ls_qa, loss_valid_ls = [], [], [], []
    qa_acc, qa_f1, sent_acc, sent_f1 = [], [], [], []

    acc_loss, acc_loss_s, acc_loss_qa = [], [], []
    best_qa_f1, best_sent_f1 = None, None

    best_valid = 1e3
    unchanged = 0

    if weights is not None:
        weights = torch.tensor([weights, 1.0], dtype=torch.float32).to(device)

    cur_used_ls_mean, total_used, total_s, mean_seg_len = None, None, None, None

    for _ in trange(num_train_epochs, desc="Epoch"):
        for step, batch in enumerate(tqdm(loader_train, desc="Iteration")):
            optimizer.zero_grad()

            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, start_positions, end_position, sent_labels, seg_ids = batch

            loss, loss_s, loss_q = model(input_ids, seg_ids, input_mask, sent_labels, start_positions, end_position, weights, train=True)

            loss.backward()
            optimizer.step()

            acc_loss.append(loss.cpu().data.numpy())
            acc_loss_s.append(loss_s.cpu().data.numpy())
            acc_loss_qa.append(loss_q.cpu().data.numpy())

            if (step + 1) % 10000 == 0:
                loss_ls.append(np.mean(acc_loss))
                loss_ls_s.append(np.mean(acc_loss_s))
                loss_ls_qa.append(np.mean(acc_loss_qa))

                acc_loss, acc_loss_s, acc_loss_qa = [], [], []

                with torch.no_grad():
                    eval_gt_start, eval_gt_end, eval_gt_sent = [], [], []
                    eval_sys_start, eval_sys_end, eval_sys_sent = [], [], []

                    valid_ls = []

                    for _, batch_valid in enumerate(tqdm(loader_valid, desc="Validation")):
                        batch_valid = tuple(t2.to(device) for t2 in batch_valid)

                        input_ids, input_mask, start_positions, end_position, sent_labels, seg_ids = batch_valid
                        start_l, end_l, sent_l, valid_l = model(input_ids, seg_ids, input_mask, sent_labels, start_positions, end_position, None)

                        eval_gt_start.extend(start_positions.cpu().data.numpy())
                        eval_gt_end.extend(end_position.cpu().data.numpy())
                        eval_gt_sent.extend(sent_labels.cpu().data.numpy())

                        eval_sys_start.extend(start_l.cpu().data.numpy())
                        eval_sys_end.extend(end_l.cpu().data.numpy())
                        eval_sys_sent.extend(sent_l.cpu().data.numpy())

                        valid_ls.append(valid_l.cpu().data.numpy())

                    qa_acc_val, qa_f1_val, sent_acc_val, sent_f1_val = get_valid_evaluation(eval_gt_start,
                                                                                            eval_gt_end,
                                                                                            eval_gt_sent,
                                                                                            eval_sys_start,
                                                                                            eval_sys_end,
                                                                                            eval_sys_sent)

                    avg_val_loss = np.mean(valid_ls)

                    qa_acc.append(qa_acc_val)
                    qa_f1.append(qa_f1_val)
                    sent_acc.append(sent_acc_val)
                    sent_f1.append(sent_f1_val)
                    loss_valid_ls.append(avg_val_loss)

                    if avg_val_loss < best_valid:
                        best_valid = avg_val_loss
                        unchanged = 0

                        best_qa_f1 = qa_f1_val
                        best_sent_f1 = sent_f1_val

                        cur_used_ls_mean, total_used, total_s, mean_seg_len = create_valid_rouge(x_for_rouge,
                                                                                                 eval_sys_sent,
                                                                                                 eval_sys_start,
                                                                                                 eval_sys_end,
                                                                                                 eval_gt_sent,
                                                                                                 eval_gt_start,
                                                                                                 eval_gt_end,
                                                                                                 batch_ids,
                                                                                                 x_sent_align,
                                                                                                 rouge_sys_sent_path,
                                                                                                 rouge_sys_segs_path,
                                                                                                 ofp_fname)

                        model_to_save = model.module if hasattr(model, 'module') else model
                        torch.save(model_to_save.state_dict(), output_model_file)

                        with open(output_config_file, 'w') as f:
                            f.write(model_to_save.config.to_json_string())

                    elif unchanged > unchanged_limit:
                        create_metric_figure(ofp_fname, loss_ls, loss_ls_s, loss_ls_qa, loss_valid_ls, qa_f1, sent_f1, cur_used_ls_mean, total_used, total_s, mean_seg_len, best_qa_f1, best_sent_f1)
                        return
                    else:
                        unchanged += 1

    create_metric_figure(ofp_fname, loss_ls, loss_ls_s, loss_ls_qa, loss_valid_ls, qa_f1, sent_f1, cur_used_ls_mean, total_used, total_s, mean_seg_len, best_qa_f1, best_sent_f1)


def evaluate(model, data_loader, x_for_rouge=None, x_sent_align=None, ofp_fname='PLT', batch_ids=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    rouge_sys_sent_path = 'data.nosync/rouge_sent_test/' + ofp_fname + '/'
    rouge_sys_segs_path = 'data.nosync/rouge_segs_test/' + ofp_fname + '/'

    if not os.path.exists(rouge_sys_sent_path):
        os.mkdir(rouge_sys_sent_path)
    if not os.path.exists(rouge_sys_segs_path):
        os.mkdir(rouge_sys_segs_path)

    eval_gt_start, eval_gt_end, eval_gt_sent = [], [], []
    eval_sys_start, eval_sys_end, eval_sys_sent = [], [], []

    for _, batch_valid in enumerate(tqdm(data_loader, desc="Evaluation")):
        batch_valid = tuple(t2.to(device) for t2 in batch_valid)

        input_ids, input_mask, start_positions, end_position, sent_labels, seg_ids = batch_valid
        start_l, end_l, sent_l, valid_l = model(input_ids, seg_ids, input_mask, sent_labels, start_positions,
                                                end_position, None)

        eval_gt_start.extend(start_positions.cpu().data.numpy())
        eval_gt_end.extend(end_position.cpu().data.numpy())
        eval_gt_sent.extend(sent_labels.cpu().data.numpy())

        eval_sys_start.extend(start_l.cpu().data.numpy())
        eval_sys_end.extend(end_l.cpu().data.numpy())
        eval_sys_sent.extend(sent_l.cpu().data.numpy())

    qa_acc_val, qa_f1_val, sent_acc_val, sent_f1_val = get_valid_evaluation(eval_gt_start,
                                                                            eval_gt_end,
                                                                            eval_gt_sent,
                                                                            eval_sys_start,
                                                                            eval_sys_end,
                                                                            eval_sys_sent)

    cur_used_ls_mean, total_used, total_s, mean_seg_len = create_valid_rouge(x_for_rouge,
                                                                             eval_sys_sent,
                                                                             eval_sys_start,
                                                                             eval_sys_end,
                                                                             eval_gt_sent,
                                                                             eval_gt_start,
                                                                             eval_gt_end,
                                                                             batch_ids,
                                                                             x_sent_align,
                                                                             rouge_sys_sent_path,
                                                                             rouge_sys_segs_path,
                                                                             ofp_fname)

    create_metric_eval(ofp_fname, cur_used_ls_mean, total_used, total_s, mean_seg_len, qa_f1_val, sent_f1_val)



args = parse.get_args()

batch_size = args.batch_size
sent_len = args.sent_len
ofp_fname = create_output_name(args)

if args.train:

    data_loader_valid, num_val, b_ls, x_for_rouge, all_sent_align = create_iterator(data_split='valid',
                                                                                         max_len=sent_len,
                                                                                         max_size=-1,
                                                                                         batch_size=batch_size,
                                                                                         balance=None,
                                                                                         bert_model=args.bert_model,
                                                                                         use_posit=args.use_positional)

    data_loader_train, num_train, _, _, _ = create_iterator(data_split='train',
                                                            max_len=sent_len,
                                                            max_size=-1,
                                                            batch_size=batch_size,
                                                            balance=args.balance,
                                                            bert_model=args.bert_model,
                                                            use_posit=args.use_positional)

    model = CustomNetwork.from_pretrained(args.bert_model, use_positional=args.use_positional, dropout=args.dropout)

    train(model=model,
          loader_train=data_loader_train,
          loader_valid=data_loader_valid,
          num_train_epochs=args.epochs,
          x_for_rouge=x_for_rouge,
          x_sent_align=all_sent_align,
          optim=args.optim,
          learning_rate=args.lr,
          unchanged_limit=args.unchanged_limit,
          weights=args.weights,
          ofp_fname=ofp_fname,
          batch_ids=b_ls)
else:
    output_model_file = 'saved_models/' + ofp_fname

    model = CustomNetwork.from_pretrained(args.bert_model, use_positional=args.use_positional, dropout=args.dropout)

    data_loader, num_val, b_ls, x_for_rouge, all_sent_align = create_iterator(data_split='test',
                                                                                    max_len=sent_len,
                                                                                    max_size=-1,
                                                                                    batch_size=batch_size,
                                                                                    balance=None,
                                                                                    bert_model=args.bert_model,
                                                                                    use_posit=args.use_positional)

    model.load_state_dict(torch.load(output_model_file))
    model.eval()

    evaluate(model=model,
             data_loader=data_loader,
             x_for_rouge=x_for_rouge,
             x_sent_align=all_sent_align,
             ofp_fname=ofp_fname,
             batch_ids=b_ls)




