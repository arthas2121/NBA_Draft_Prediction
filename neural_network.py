import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import utils

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()
        # Number of input features is 48.
        self.layer_1 = nn.Linear(46, 128) 
        self.layer_2 = nn.Linear(128, 128)
        self.layer_out = nn.Linear(128, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        #x = self.dropout(x)
        x = self.relu(self.layer_2(x))
        x = self.dropout(x)
        x = self.layer_out(x)
        return x

#Dataset structure
class TrainData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

class ValidationData(Dataset):
    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)

class TestData(Dataset):
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)

def calc_train_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

def use_neural_network(X_train, y_train, X_test, y_test):
    X_new, y_new = utils.transform_pairwise(X_train, y_train)
    X_test_new, y_test_new = utils.transform_pairwise(X_test, y_test)

    #Split train/validation/test dataset
    X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(X_new, y_new, train_size=0.2)
    train_data = TrainData(torch.FloatTensor(X_train_nn), 
                           torch.FloatTensor(np.where(y_train_nn == -1, 0, y_train_nn)))
    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)

    val_data = ValidationData(torch.FloatTensor(X_test_nn))
    val_loader = DataLoader(dataset=val_data, batch_size=1)

    test_data = TestData(torch.FloatTensor(X_test_new))
    test_loader = DataLoader(dataset=test_data, batch_size=1)

    #Define model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BinaryClassification()
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    #Train
    print('Using Neural Network...')
    print('Start training ... It may takes a few minutes ...')
    model.train()
    best_accuracy = 0
    train_accuracy_list = []
    val_accuracy_list = []
    epochs = 100
    for e in range(epochs):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
        
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.unsqueeze(1))
            acc = calc_train_acc(y_pred, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
        
            epoch_loss += loss.item()
            epoch_acc += acc.item()

        #Validation
        if e % 10 == 0:
            y_pred_list = []
            model.eval()
            with torch.no_grad():
                for X_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_test_pred = model(X_batch)
                    y_test_pred = torch.sigmoid(y_test_pred)
                    y_pred_tag = torch.round(y_test_pred)
                    y_pred_list.append(y_pred_tag.cpu().numpy())

            y_pred_list = np.array([int(a.squeeze().tolist()) for a in y_pred_list])
            val_accuracy = accuracy_score(np.where(y_test_nn == -1, 0, y_test_nn), y_pred_list)
            val_accuracy_list.append(val_accuracy)
            train_accuracy_list.append(epoch_acc/len(train_loader))
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(model.state_dict(), 'rank_nn.pt') # save best model
            print(f"Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | train acc: {epoch_acc/len(train_loader):.3f} | val acc: {val_accuracy:.3f}")
    
    #Prediction
    print('Finished training. Start prediction ...')
    best_state_dict = torch.load('rank_nn.pt') #load best model
    model.load_state_dict(best_state_dict) 
    y_pred = []
    model.eval()
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred.append(y_pred_tag.cpu().numpy())

    y_pred = np.array([int(i.squeeze().tolist()) for i in y_pred])
    print('Pairwise accuracy of Neural Network: ', accuracy_score(y_test_new, np.where(y_pred == 0, -1, y_pred)))
    rank, score = utils.calc_ndcg(np.where(y_pred == 0, -1, y_pred), y_test)
    print("NDCG score of Neural Network: ", score)
    return rank