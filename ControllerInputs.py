from collections import deque
from dis import disco
import os
from time import sleep
import melee
import numpy as np
import argparse
import signal
import sys
import random
import torch
import PyTorchNN
from Plotting import plot
import csv
import json
BotStocks = 4 # To work out reward stuff
OppStocks = 4
BotPercent = 0
OppPercent = 0
Scores = []
MeanScores = []
TotalScore = 0
BestScore = 0
Score = 0
CharSelected = False
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device on Agent")
model = PyTorchNN.NeuralNetwork(13, 39, 30).to(device)
PyTorchNN.LoadModel("model/model.pth", model)
Trainer = PyTorchNN.QTrain(Model=model, LearningRate=0.001, Gamma=0.9) # Gamma < 1
State0 = [] # to hold oldstate
State1 = [] # To hold newstate
NumberOfGames = 0 # Counter
GamePlayed = False # For console.step stuff
MaxMem = 100000 # MaxMem size for training
BatchSize = 1000 # Max Batch size for longtermtrain
TradeoffNum = 400 # Number of games to stop trying some random inputs
Memory = deque(maxlen=MaxMem) # Pops Left if MaxMem reached

if os.path.isfile('Plotting/JsonPlot.json'):
  with open('Plotting/JsonPlot.json') as f:
    JsonObj = json.load(f)
  Scores = JsonObj["Scores"]
  MeanScores = JsonObj["MeanScores"]
  TotalScore = JsonObj["TotalScore"]
  NumberOfGames = JsonObj["NumberOfGames"]

def Normalize(Min, X, Max):
  Normal = (X - Min) / (Max-Min)
  return Normal

def Remember(State, Action, Reward, NextState):
  Memory.append((State, Action, Reward, NextState))

def LongMemTrain():
  if len(Memory) > BatchSize: # Gets selection of memory to use for training that is the batchsize
    SelectionSample = random.sample(Memory, BatchSize) # List of Tuples
  else:
    SelectionSample = Memory
  
  States, Actions, Rewards, NextStates = zip(*SelectionSample)
  Trainer.TrainStep(States, Actions, Rewards, NextStates)

def ShortMemTrain(State, Action, Reward, NextState):
  Trainer.TrainStep(State, Action, Reward, NextState)

def TradeOff(Predictions):
  Temp = [0] * 30 
  Action = 0
  if random.randint(0, 200) < (TradeoffNum - NumberOfGames):
    #print(random.randint(0,29))
    Temp[random.randint(0,29)] = 1
    Action = GetAction(Temp)
  else:
    Action = GetAction(Predictions)
    Temp[Action] = 1
  return Temp

def GetAction(Predictions): #Turns prediction into controller input
  Prediction = np.argmax(Predictions)
  Action = np.int16(Prediction) + 1

  match Action:
           case 1:
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.25, 0.5)
             #Left
           case 2:
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.75, 0.5)
             #Right
           case 3:
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.5, 0.75)
             #Up
           case 4:
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.5, 0.25)
             #Down
           case 5:
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.25, 0.75)
             #Left-Up
           case 6:
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.75, 0.75)
             #Right-Up
           case 7:
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.25, 0.25)
             #Left-Down
           case 8:
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.75, 0.25)
             #Right-Down
           case 9:
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.5, 0.5)
             #Neutral
           case 10:
             controller.press_button(melee.enums.Button.BUTTON_A)
           case 11:
             controller.press_button(melee.enums.Button.BUTTON_B)
           case 12:
             controller.press_button(melee.enums.Button.BUTTON_Y)
           case 13:
             controller.press_button(melee.enums.Button.BUTTON_L)
           case 14:
             controller.press_button(melee.enums.Button.BUTTON_R)
           case 15:
             controller.press_button(melee.enums.Button.BUTTON_Z)
           case 16:
             controller.press_button(melee.enums.Button.BUTTON_A)
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.25, 0.5)
             #L+A
           case 17:
             controller.press_button(melee.enums.Button.BUTTON_B)
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.25, 0.5)
             #L+B
           case 18:
             controller.press_button(melee.enums.Button.BUTTON_L)
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.25, 0.5)
             #L+R
           case 19:
             controller.press_button(melee.enums.Button.BUTTON_A)
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.75, 0.5)
             #R+A
           case 20:
             controller.press_button(melee.enums.Button.BUTTON_B)
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.75, 0.5)
             #R+B
           case 21:
             controller.press_button(melee.enums.Button.BUTTON_R)
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.75, 0.5)
             #R+R
           case 22:
             controller.press_button(melee.enums.Button.BUTTON_A)
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.5, 0.75)
             #U+A
           case 23:
             controller.press_button(melee.enums.Button.BUTTON_B)
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.5, 0.75)
             #U+B
           case 24:
             controller.press_button(melee.enums.Button.BUTTON_A)
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.5, 0.25)
             #D+A
           case 25:
             controller.press_button(melee.enums.Button.BUTTON_B)
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.5, 0.25)
             #D+B
           case 26:
             controller.press_button(melee.enums.Button.BUTTON_R)
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.5, 0)
             #D+R
           case 27:
             controller.press_button(melee.enums.Button.BUTTON_A)
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 1, 0.5)
             #SmashL
           case 28:
             controller.press_button(melee.enums.Button.BUTTON_A)
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0, 0.5)
             #SmashR
           case 29:
             controller.press_button(melee.enums.Button.BUTTON_A)
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.5, 1)
             #SmashU
           case 30:
             controller.press_button(melee.enums.Button.BUTTON_A)
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.5, 0)
             #SmashD
  return Action  

def GetStates():
     Inputs = [gamestate.players[1].facing, gamestate.players[2].facing,
     gamestate.players[1].hitlag_left, gamestate.players[2].hitlag_left, 
     gamestate.players[1].on_ground, gamestate.players[1].percent / 999, 
     gamestate.players[2].percent / 999, Normalize(-175.7,gamestate.players[1].position.x, 173.6), 
     Normalize(-91,gamestate.players[1].position.y, 168), Normalize(-175.7,gamestate.players[2].position.x, 173.6), # Need to normalise X and Y values
     Normalize(-91,gamestate.players[2].position.y, 168), gamestate.players[1].stock / 4, 
     gamestate.players[2].stock / 4]
     
     Inputs = [np.float32(i) for i in Inputs]
     return Inputs

def check_port(value):
    ivalue = int(value)
    if ivalue < 1 or ivalue > 4:
        raise argparse.ArgumentTypeError("%s is an invalid controller port. \
                                         Must be 1, 2, 3, or 4." % value)
    return ivalue

def CheckReward():
  Reward = 0
  global BotStocks
  global BotPercent
  global OppStocks
  global OppPercent

  if gamestate.players[2].hitlag_left == True:
      Reward += 0.01

  if gamestate.players[2].off_stage == True:
      Reward += 0.01
     
  if gamestate.players[2].percent > OppPercent:
      Reward += (0.01 * (gamestate.players[2].percent - OppPercent) / 10)
      OppPercent = gamestate.players[2].percent

  if gamestate.players[2].stock < OppStocks:
      Reward += 10
      OppStocks = gamestate.players[2].stock
      OppPercent = 0

  if gamestate.players[1].stock < BotStocks:
      Reward -= 10
      BotStocks = gamestate.players[1].stock
      BotPercent = 0

  if gamestate.players[1].percent > BotPercent:
      Reward -= (0.01 * (gamestate.players[1].percent - BotPercent) / 10)
      BotPercent = gamestate.players[1].percent

  return Reward
    


parser = argparse.ArgumentParser(description='Libmelee in action')
parser.add_argument('--dolphin_executable_path', '-e', default=None, help='The directory where dolphin is')
parser.add_argument('--connect_code', '-t', default="", help='Direct connect code to connect to in Slippi Online')
args = parser.parse_args()

console = melee.Console(path=args.dolphin_executable_path)
controller = melee.Controller(console=console, port=1, type=melee.ControllerType.STANDARD)
controller_opponent = melee.Controller(console=console, port=2, type=melee.ControllerType.STANDARD)
console.run()
console.connect()
controller.connect()
controller_opponent.connect()
costume = 1
framedata = melee.framedata.FrameData()

gamestate = console.step()
while True:
    #gamestate = console.step()
    if gamestate is None:
        gamestate = console.step()
        continue
    if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
        Reward = np.float32(CheckReward())
        controller.release_all()
        #Get Current State
        State0 = GetStates()
        #Get Prediction of action to take
        Predictions = PyTorchNN.GetPredictions(GetStates(), model)
        #Turn action from list into controller input
        Action = TradeOff(Predictions)
        #Push controller inputs to dolphin
        controller.flush()
        #Continue 1 frame
        gamestate = console.step()
        #Get New State
        State1 = GetStates()

        ShortMemTrain(State0, Action, Reward, State1)
        Remember(State0, Action, Reward, State1)
        GamePlayed = True
        Score += Reward
        CharSelected = True
    else:
        #print(Fitness)
        if GamePlayed == True:
          NumberOfGames += 1
          if Score > BestScore:
            BestScore = Score
          model.save()
          Scores.append(Score)
          TotalScore += Score
          MeanScores.append(TotalScore/NumberOfGames)
          #plot(Scores, MeanScores)
          PlotObj = {"Scores":Scores, "MeanScores":MeanScores, "TotalScore": TotalScore, "NumberOfGames": NumberOfGames}
          with open("Plotting/JsonPlot.json", 'w') as f:
            json.dump(PlotObj, f)
          Score = 0
          LongMemTrain()
          GamePlayed = False
        gamestate = console.step()
        if gamestate.menu_state in [melee.Menu.CHARACTER_SELECT]:
          if CharSelected == False:
            melee.MenuHelper.choose_character(gamestate=gamestate, controller=controller_opponent, cpu_level=9, start=False, character=melee.Character.FOX)
          if gamestate.players[2].character_selected == melee.Character.FOX:
            melee.MenuHelper.menu_helper_simple(gamestate, controller, melee.Character.FOX, melee.Stage.YOSHIS_STORY, costume=costume, autostart=True, swag=False, cpu_level=0)
        else:
          melee.MenuHelper.menu_helper_simple(gamestate, controller, melee.Character.FOX, melee.Stage.YOSHIS_STORY, costume=costume, autostart=True, swag=False, cpu_level=0)

          




