import os
import torch
from torch import nn
import numpy as np
import torch.optim as optim
from torch.distributions import Categorical

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device on NN")

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, output_size)
        
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.l4(out)
        out = self.relu(out)
        return out


    def save(self, FileName='reinforce5falcon.pth'):
        modelfolderpath = './model'
        if not os.path.exists(modelfolderpath):
            os.makedirs(modelfolderpath)

        FileName = os.path.join(modelfolderpath, FileName)
        torch.save(self.state_dict(), FileName)



class QTrain:
    def __init__(self, Model, LearningRate, Gamma):
        self.LearningRate = LearningRate
        self.Gamma = Gamma
        self.Model = Model
        self.Optimizer = optim.Adam(Model.parameters(), lr=self.LearningRate)
        self.Criterion = nn.MSELoss()

    def TrainStep(self, State, Action, Reward, NextState):
        State = torch.tensor(State, dtype=torch.float)
        NextState = torch.tensor(NextState, dtype=torch.float)
        Action = torch.tensor(Action, dtype=torch.int16)
        Reward = torch.tensor(Reward, dtype=torch.float) # Turn values into tensors
        if len(State.shape) == 1:
            State = torch.unsqueeze(State, 0) # Do some dimention trickery
            NextState = torch.unsqueeze(NextState, 0)
            Action = torch.unsqueeze(Action, 0)
            Reward = torch.unsqueeze(Reward, 0)
        Prediction = GetPredictions(State, self.Model) # R + y * NextPredict Q Value
        #print(State)
        Prediction = torch.tensor(Prediction, dtype=torch.float)
        target = Prediction.clone()
        for i in range(len(State)):
            #print(len(State.detach().numpy()))
            #print(State)
            QNew = Reward[i] + self.Gamma * torch.max(torch.tensor(GetPredictions(NextState[i], self.Model), dtype=torch.float))
            #print(torch.argmax(Action).item())
            target[i][torch.argmax(Action[i]).item()] = QNew

        self.Optimizer.zero_grad()
        Loss = self.Criterion(Prediction.detach(), target.detach())
        Loss.requires_grad = True
        Loss.backward()
        self.Optimizer.step()


class ReinforceTrainer:

    def __init__(self, Model, Gamma):
        self.Model = Model
        self.Gamma = Gamma
        self.Optimizer = optim.Adam(Model.parameters(), lr=0.1)

    def reinforce(self, Model, Rewards, Probabilities):
        Discounts = [self.Gamma ** i for i in range(len(Rewards) + 1)]
        R = sum(a*b for a,b in zip(Discounts, Rewards))
        #Probabilities = np.ndarray(Probabilities)
        Probabilities = torch.tensor(Probabilities, dtype=torch.float)
        
        LogProbs = []
        for i in range(1, len(Probabilities)):
            Categorize = Categorical(Probabilities[i][:])
            LogProbs.append(Categorize.log_prob(Categorize.sample()))

        ModelLoss = []
        for Probs in LogProbs:
            ModelLoss.append(-Probs * R)
        ModelLoss = torch.stack(ModelLoss).sum()

        self.Optimizer.zero_grad()
        ModelLoss.requires_grad = True
        ModelLoss.backward()
        self.Optimizer.step()
    


    
            



def LoadModel(Path, Model):
    Model.load_state_dict(torch.load(Path))
    Model.eval()


def GetPredictions(Inputs, model):
    Dims = np.ndim(Inputs)
    if Dims == 1:
        NewInputs = [np.float32(i) for i in Inputs]
        NewInputs = torch.from_numpy(np.array(NewInputs)).to(device)
        Predicts = model(NewInputs)
        Predicts = Predicts.cpu()
        Predicts = Predicts.detach().numpy()
    else:
        NewInputs = Inputs.clone()
        NewInputs = NewInputs.to(device)
        Predicts = model(NewInputs)
        Predicts.cpu()
    return Predicts

