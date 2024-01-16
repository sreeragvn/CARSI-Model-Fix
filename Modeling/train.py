import argparse

import sys
import numpy as np
import pandas as pd

import pickle
import datetime 
import os
import torch 
import torch.nn as nn
from Models import RNN, CARSI
from DatasetClass import DataSetWindows

from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Precision, F1Score, Recall, Accuracy, ConfusionMatrix
import warnings 

warnings.filterwarnings("ignore", category=UserWarning)
torch.manual_seed(0)

#print(os.getcwd())
#os.chdir('./Modeling')
######################## Args
def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--save', type=bool, default=True, help='save the model')
    parser.add_argument('--model', type=str, default='carsi', help='rnn/lstm/gru/carsi')
    parser.add_argument('--batch_size', type=int, default=4, help='batch Size during training')
    parser.add_argument('--epoch', default=200, type=int, help='epoch to run')
    parser.add_argument('--learning_rate', default=0.005, type=float, help='initial learning rate')
    parser.add_argument('--loss', type=str, default='NLLLoss', help='loss function if other implemented')
    parser.add_argument('--weather', type=bool, default=False, help='include weather variables')
    parser.add_argument('--last_interaction', type=bool, default=False, help='include interaction history')
    parser.add_argument('--car', type=str, default="all", help='all / SEB880 / ... (select one car or all)')
    parser.add_argument('--finetune', type=bool, default=False, help='finetune to a specific car')
    parser.add_argument('--model_path', type=str, default='', help='path to best_model to load')
    return parser.parse_args()

args = parse_args()

######################## Checkpoint dir
if args.save:
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    path_to_checkpoint = f"checkpoints/{args.model}/{timestr}"
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

print(training_set)

training_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=False)
testing_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=False)

######################## Helper Functions
def metric_call(true_labels, predicted_labels):
    
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
elif args.model == "carsi":
    model = CARSI(input_size_cont=_num_cont_var,
                    input_size_cat= _num_static_var,
                    output_size=_num_classes,
                    seq_len=_seq_len,
                    num_embedding=_num_emb)

if args.finetune:
    print("--Load Model for Fine-Tuning--")
    model.load_state_dict(torch.load(os.path.join(args.model_path, 'best_model.pth')))

######################## Training Loop
writer = SummaryWriter()

criterion = nn.NLLLoss(weight=_class_w) # CrossEntropy to remove softmax
optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

step = 0

for epoch in range(args.epoch):                                  

    loss_counter = []
    accuracy_counter = []
    accuracy_top_three = []
    f1_score_epoch = []

    tm_f1 = []
    tm_precision = []
    tm_recall = []

    # For confusion matrix
    predicted_labels = []
    true_labels = []

    train_metrics = {}

    for inputs, labels in training_loader:           

        cont_inputs, static_inputs = inputs

        outputs = model(cont_inputs, static_inputs)
        
        loss = criterion(outputs, labels)
        loss_counter.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
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

    # Metrics mean per epoch
    train_metrics['Acc'] = np.mean(accuracy_counter)
    train_metrics['AccTopThree'] = np.mean(accuracy_top_three)
    train_metrics['Precision'] = np.mean(tm_precision)
    train_metrics['Recall'] = np.mean(tm_recall)
    train_metrics['F1Score'] = np.mean(tm_f1)

    mean_loss = float(np.mean(loss_counter))

    writer.add_scalar("Training Loss", mean_loss, global_step=step)
    writer.add_scalar("Training Accuracy", train_metrics['Acc'], global_step=step)
    writer.add_scalar("Training Precision", train_metrics['Precision'], global_step=step)
    writer.add_scalar("Training Recall", train_metrics['Recall'], global_step=step)
    writer.add_scalar("Training F1-Score", train_metrics['F1Score'], global_step=step)    

    print(f"----- Epoch {epoch} -----")
    output_message = (
            "Train  - Loss: {:.3f} - Acc: {:.3f} - Acc Top: {:.3f} - Prec: {:.3f} - Recall: {:.3f}, F1: {:.3f}"
            .format(mean_loss, train_metrics['Acc'], train_metrics['AccTopThree'], 
                    train_metrics['Precision'], train_metrics['Recall'], train_metrics['F1Score'])
            )
    print(output_message)

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

        for inputs, labels in testing_loader:           

            cont_inputs, static_inputs = inputs

            outputs = model(cont_inputs, static_inputs)
            
            loss = criterion(outputs, labels)
            loss_counter.append(loss.item())
            
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

        mean_loss = float(np.mean(loss_counter))

        writer.add_scalar("Test Loss", mean_loss, global_step=step)
        writer.add_scalar("Test Accuracy", test_metrics['Acc'], global_step=step)
        writer.add_scalar("Test Precision", test_metrics['Precision'], global_step=step)
        writer.add_scalar("Test Recall", test_metrics['Recall'], global_step=step)
        writer.add_scalar("Test F1-Score", test_metrics['F1Score'], global_step=step)
        step += 1

        output_message = (
            "Test  - Loss: {:.3f} - Acc: {:.3f} - Acc Top: {:.3f} - Prec: {:.3f} - Recall: {:.3f}, F1: {:.3f}"
            .format(mean_loss, test_metrics['Acc'], test_metrics['AccTopThree'], 
                    test_metrics['Precision'], test_metrics['Recall'], test_metrics['F1Score'])
        )
        print(output_message)

        # Save model 
        if args.save & (test_metrics['Acc'] > best_accuracy):
            best_metrics = pd.DataFrame([train_metrics.values(), test_metrics.values()], columns=train_metrics.keys(), index=['Train', 'Test'])
            best_metrics.to_csv(os.path.join(path_to_metrics, "best_metrics.csv"))
            metric_call(true_labels, predicted_labels)
            model_path = os.path.join(path_to_checkpoint, 'best_model.pth')
            torch.save(model.state_dict(), model_path)