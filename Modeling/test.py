import argparse

import sys
import numpy as np
import pandas as pd

import pickle
import datetime 
import os
import torch 
from Models import RNN, CARSI
from DatasetClass import DataSetWindows

from torchmetrics import Precision, F1Score, Recall, Accuracy, ConfusionMatrix
import warnings 

warnings.filterwarnings("ignore", category=UserWarning)
torch.manual_seed(0)

os.chdir('./Modeling')
######################## Args
def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--save', type=bool, default=True, help='save the model')
    parser.add_argument('--model', type=str, default='trans', help='rnn/lstm/gru/trans')
    parser.add_argument('--batch_size', type=int, default=4, help='batch Size during training')
    parser.add_argument('--epoch', default=5, type=int, help='epoch to run')
    parser.add_argument('--weather', type=bool, default=False, help='include weather variables')
    parser.add_argument('--last_interaction', type=bool, default=False, help='include weather variables')
    parser.add_argument('--car', type=str, default="SEB889", help='all / SEB880 / ...')
    parser.add_argument('--model_path', type=str, default="checkpoints/carsi/2023-11-20_20-24", help='path to best_model to load')
    return parser.parse_args()

args = parse_args()

######################## Checkpoint dir
if args.save:
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    path_to_checkpoint = f"checkpoints/test/{args.model}/{timestr}"
    os.makedirs(path_to_checkpoint)
    path_to_metrics = os.path.join(path_to_checkpoint, "metrics")
    os.makedirs(path_to_metrics)

######################## Param from data
with open('Dataset/data_all/param.pkl', 'rb') as f:
    _label_mapping = pickle.load(f)
    _class_w = pickle.load(f)
    _num_emb = pickle.load(f)
    _continuous_var = pickle.load(f)
    _static_var = pickle.load(f)
    _seq_len = pickle.load(f)

_num_cont_var = len(_continuous_var)
_num_static_var = len(_static_var)
_num_classes = len(_label_mapping.keys())

######################## Data Loading
training_set = DataSetWindows(split="train", weather=args.weather, car_id=args.car)
testing_set = DataSetWindows(split="test", weather=args.weather, car_id=args.car)

training_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=False)
testing_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=False)

######################## Helper Functions
def confusion_matrix_call(true_labels, predicted_labels):
    
    # metrics per class to dataframe
    accuracy = Accuracy(task="multiclass", average=None, num_classes=_num_classes) 
    f1 = F1Score(task="multiclass", average=None, num_classes=_num_classes) 
    precision = Precision(task="multiclass", average=None, num_classes=_num_classes)
    recall = Recall(task="multiclass", average=None, num_classes=_num_classes)
    conf_matrix = ConfusionMatrix(task="multiclass", num_classes=_num_classes)

    acc_list = accuracy(torch.tensor(true_labels), torch.tensor(predicted_labels)).tolist()
    f1_list = f1(torch.tensor(true_labels), torch.tensor(predicted_labels)).tolist()
    precision_list = precision(torch.tensor(true_labels), torch.tensor(predicted_labels)).tolist()
    recall_list = recall(torch.tensor(true_labels), torch.tensor(predicted_labels)).tolist()
    cm = conf_matrix(torch.tensor(true_labels), torch.tensor(predicted_labels)).tolist()

    metrics_data = [acc_list, f1_list, recall_list, precision_list]
    index_names = ['Accuracy', 'F1Score', 'Recall', "Precision"]
    metrics_df = pd.DataFrame(metrics_data, columns=_label_mapping.keys(), index=index_names)
    metrics_df.to_csv(os.path.join(path_to_metrics, "class_metrics.csv"))
    cm_df = pd.DataFrame(cm, columns=_label_mapping.keys(), index=_label_mapping.keys())
    cm_df.to_csv(os.path.join(path_to_metrics, "confusion_matrix.csv"))
    
######################## Model
NUM_HIDDEN = 256
NUM_LAYERS = 2

if args.model in ["rnn", "lstm", "gru"]:
    model = RNN(input_size_cont=_num_cont_var,
                input_size_cat= _num_static_var, 
                    hidden_size=NUM_HIDDEN, 
                    num_layers=NUM_LAYERS, 
                    num_classes=_num_classes, 
                    num_embedding=_num_emb,
                    model=args.model)
elif args.model == "trans":
    model = CARSI(input_size_cont=_num_cont_var,
                        input_size_cat= _num_static_var,
                    output_size=_num_classes,
                    seq_len=_seq_len,
                    num_embedding=_num_emb)

print("--Load Model--")
model.load_state_dict(torch.load(os.path.join(args.model_path, 'best_model.pth')))

######################## Testing Loop

for epoch in range(args.epoch):                                  

    with torch.no_grad():
        loss_counter = []
        accuracy_counter = []
        accuracy_top_three = []
        f1_score_epoch = []

        tm_f1 = []
        tm_precision = []
        tm_recall = []

        pred_total = []
        target_total = []
        
        # For confusion matrix
        predicted_labels = []
        true_labels = []

        test_metrics = {}


        best_accuracy = 0

        for inputs, labels in training_loader:           

            cont_inputs, static_inputs = inputs

            outputs = model(cont_inputs, static_inputs)
            
            
            pred = outputs.data.max(1)[1]
            top_three = np.argsort(-outputs.detach().numpy(), axis=1)[:, :3]

            # Use torchmetrics
            f1 = F1Score(task="multiclass", average="weighted", num_classes=_num_classes) 
            precision = Precision(task="multiclass", average="weighted", num_classes=_num_classes)
            recall = Recall(task="multiclass", average="weighted", num_classes=_num_classes)
            tm_f1.append(f1(pred, labels))
            tm_precision.append(precision(pred, labels))
            tm_recall.append(recall(pred, labels))
            
            # Accuracy based on top three
            correct_top = 0
            for i, top_batch in enumerate(top_three):
                if labels.data[i].item() in top_batch:
                    correct_top += 1
            accuracy_top_three.append(correct_top/len(labels))

            # Arcuracy based on prediction
            correct = pred.eq(labels.data).cpu().sum()
            accuracy_counter.append(correct.item()/len(labels))

            # Confusion matrix last epoch
            predicted_labels += pred.tolist()
            true_labels += labels.tolist()

        # Metrics mean per epoch
        test_metrics['Acc'] = np.mean(accuracy_counter)
        test_metrics['AccTopThree'] = np.mean(accuracy_top_three)
        test_metrics['Precision'] = np.mean(tm_precision)
        test_metrics['Recall'] = np.mean(tm_recall)
        test_metrics['F1Score'] = np.mean(tm_f1)

        output_message = (
            "Test  - Acc: {:.3f} - Acc Top: {:.3f} - Prec: {:.3f} - Recall: {:.3f}, F1: {:.3f}"
            .format(test_metrics['Acc'], test_metrics['AccTopThree'], 
                    test_metrics['Precision'], test_metrics['Recall'], test_metrics['F1Score'])
        )
        print(output_message)

        # Save model 
        if args.save & (test_metrics['Acc'] > best_accuracy):
            best_metrics = pd.DataFrame([test_metrics.values()], columns=test_metrics.keys(), index=['Train', 'Test'])
            best_metrics.to_csv(os.path.join(path_to_metrics, "best_metrics.csv"))
            confusion_matrix_call(true_labels, predicted_labels)



    


        


