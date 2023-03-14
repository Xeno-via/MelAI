import copy
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
        self.Model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
        
    
    def forward(self, input):
        #input = input.to(device)
        return self.Model(input)

class QTrain:
    def __init__(self,LearningRate, Gamma, FileName):
        self.LearningRate = LearningRate
        self.Gamma = Gamma
        self.FileName = FileName
        self.NN = NeuralNetwork(14, 50, 30).Model
        self.NN.to(device)
        self.Target = copy.deepcopy(self.NN)
        self.Target.to(device)
        for p in self.Target.parameters():
            p.requires_grad = False
        self.Optimizer = optim.Adam(self.NN.parameters(), lr=self.LearningRate)
        self.Criterion = nn.SmoothL1Loss()
        self.CurrentStep = 0
        self.SaveEvery = 1000
        self.SyncEvery = 1000
        self.BatchSize = 32



    def Converting(self, Tuple):
        new = torch.stack(Tuple)
        return new


    def save(self, FileName):
        modelfolderpath = './model'
        if not os.path.exists(modelfolderpath):
            os.makedirs(modelfolderpath)

        FilePath = os.path.join(modelfolderpath, FileName + '.pth')
        torch.save(self.NN.state_dict(), FilePath)


    def LoadModel(self, Path):
        self.NN.load_state_dict(torch.load(Path))
        self.NN.eval()
        self.Target = copy.deepcopy(self.NN).to(device)
        for p in self.Target.parameters():
            p.requires_grad = False

    def TDEstimate(self, State, Action):
        CurrentQ = self.NN(input=State.to(device))[np.arange(0, self.BatchSize), Action]
        return CurrentQ.cpu()
    
    @torch.no_grad()
    def TDTarget(self, Reward, NextState, Done):
        NextStateQ = self.NN(input=NextState.to(device))
        BestAction = torch.argmax(NextStateQ, axis=1)
        NextQ = self.Target(input=NextState.to(device))[np.arange(0, self.BatchSize), BestAction]
        return (Reward + (1-Done.float())*self.Gamma * NextQ.cpu()).float()

    def UpdateQModel(self, TDEstimate, TDTarget):
        Loss = self.Criterion(TDEstimate, TDTarget)
        self.Optimizer.zero_grad()
        Loss.backward()
        self.Optimizer.step()
        return Loss.item()
    
    def SyncTarget(self):
        self.Target.load_state_dict(self.NN.state_dict())


    def Train(self, State, NextState, Action, Reward, Done):
        State = self.Converting(State)
        NextState = self.Converting(NextState)
        Action = torch.from_numpy(np.asarray(Action))
        Reward = torch.from_numpy(np.asarray(Reward))
        Done = torch.from_numpy(np.asarray(Done))

        if self.CurrentStep % self.SyncEvery == 0:
            print("Synced")
            self.SyncTarget()
        
        if self.CurrentStep % self.SaveEvery ==0:
            print("Saved")
            self.save(self.FileName)

        
        TDEst = self.TDEstimate(State, Action)
        TDTarget = self.TDTarget(Reward, NextState, Done)
        Loss = self.UpdateQModel(TDEst, TDTarget)
        self.CurrentStep += 1


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
        Prediction = torch.tensor(Prediction, dtype=torch.float)
        Target = GetPredictions(NextState, self.Target)
        Target = torch.tensor(Target, dtype=torch.float)
        target = Prediction.clone()
        for i in range(len(State)):
            #print(len(State.detach().numpy()))
            #print(State)
            TDTarget = Reward[i] + self.Gamma * torch.max(torch.tensor(GetPredictions(NextState[i], self.Model), dtype=torch.float))
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
        self.Optimizer = optim.Adam(Model.parameters(), lr=0.01)

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
    


    



def GetPredictions(Inputs, model):
    Dims = np.ndim(Inputs)
    if Dims == 1:
        NewInputs = [np.float32(i) for i in Inputs]
        NewInputs = torch.from_numpy(np.array(NewInputs)).to(device)
        Predicts = model(NewInputs, model)
        Predicts = Predicts.cpu()
        Predicts = Predicts.detach().numpy()
    else:
        NewInputs = Inputs.clone()
        NewInputs = NewInputs.to(device)
        Predicts = model(NewInputs, model)
        Predicts.cpu()
    return Predicts

