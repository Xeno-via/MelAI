import os
import torch
from torch import nn
import numpy as np
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
#print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)
        
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

    def save(self, filename='model.pth'):
        modelfolderpath = './model'
        if not os.path.exists(modelfolderpath):
            os.makedirs(modelfolderpath)

        FileName = os.path.koin(modelfolderpath, FileName)
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
        Reward = torch.tensor(Reward, dtype=torch.float)
        if len(State.shape) == 1:
            State = torch.unsqueeze(State, 0)
            NextState = torch.unsqueeze(NextState, 0)
            Action = torch.unsqueeze(Action, 0)
            Reward = torch.unsqueeze(Reward, 0)

        Prediction = self.Model(State) # R + y * NextPredict Q Value
        target = Prediction.clone()

        for i in range(len(State)):
            QNew = Reward[i] + self.Gamma * torch.max(self.Model(NextState[i]))
            target[i][torch.argmax(Action).item()] = QNew

        self.Optimizer.zero_grad()
        Loss = self.Criterion(target, Prediction)
        Loss.backward()
        self.Optimizer.step()
            



def GetPredictions(Inputs, model):
    NewInputs = [np.float32(i) for i in Inputs]
    NewInputs = torch.from_numpy(np.array(NewInputs)).to(device)
    Predicts = model(NewInputs)
    Predicts = Predicts.cpu()
    Predicts = Predicts.detach().numpy()
    return Predicts

# model = NeuralNetwork(13, 39, 30).to(device)
# model = model.float()
# inputs = np.zeros(13)
# inputs = [np.float32(i) for i in inputs]
# inputs = torch.from_numpy(np.array(inputs)).to(device)
# print(torch.from_numpy(np.zeros(13)))
# print(model(inputs))



# def build_model(States, Actions):
#     model = tf.keras.Sequential([
#     tf.keras.layers.Dense((States)),
#     tf.keras.layers.Dense(24),
#     tf.keras.layers.Dense(24),
#     tf.keras.layers.Dense(Actions)])
#     model.compile(tf.keras.optimizers.RMSprop(0.001), loss='mse')
#     model.fit(np.zeros((10, States)),np.ones((10, Actions)))

#     return model


# def MakePrediction(Inputs, Model):
#     newTensor = tf.constant([Inputs], np.float64)
#     #print(newTensor)
#     return Model(newTensor, training=False)

#model = build_model(13, 14)
#print(MakePrediction([1,1,1,1,1,1,1,1,1,1,1,1,1], model))


