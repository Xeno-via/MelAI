from collections import deque
from dis import disco
import os
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
import time
StatesSetup = False
X = 0
SetupDone = False
CharSelected = False
GamePlayed = False # For console.step stuff
MaxMem = 100000 # MaxMem size for training
BatchSize = 32 # Max Batch size for longtermtrain
TradeoffNum = 5000 # Number of games to stop trying some random inputs
Memory = deque(maxlen=MaxMem) # Pops Left if MaxMem reached
FileName = "OnlyPunishOnDeath5k3Hidden40HiddenNode"



def StateSetup():
  global BotStocks
  global BotPercent
  global OppStocks
  global OppPercent
  global PrevX
  global PrevY
  global PrevAction
  global Score
  BotStocks = 4
  OppStocks = 4
  BotPercent = 0
  OppPercent = 0
  PrevX = gamestate.players[1].position.x
  PrevY = gamestate.players[1].position.y
  PrevAction = 0
  Score = 0

def NNSetup(FileName):
  global device
  global model
  global Trainer
  device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
  print(f"Using {device} device on Agent")
  #model = PyTorchNN.NeuralNetwork(13, 300, 30).to(device)
  Trainer = PyTorchNN.QTrain(LearningRate=0.00025, Gamma=0.9, FileName=FileName) # Gamma < 1
  if os.path.isfile('model/' + FileName + '.pth'):
    Trainer.LoadModel(Path="model/" + FileName + '.pth')
  #Trainer = PyTorchNN.ReinforceTrainer(Model=model, Gamma=0.9)

def JsonLoading(FileName):
  global Scores
  global MeanScores
  global TotalScore
  global NumberOfGames
  global BestScore
  global q
  global Loss
  global Tempq
  global TempLoss
  Tempq = []
  TempLoss = []
  if os.path.isfile('Plotting/' + FileName + '.json'): # Checks which JsonFile to use for data storage
    with open('Plotting/' + FileName + '.json') as f:
      JsonObj = json.load(f)
    Scores = JsonObj["Scores"]
    MeanScores = JsonObj["MeanScores"]
    TotalScore = JsonObj["TotalScore"]
    NumberOfGames = JsonObj["NumberOfGames"]
    BestScore = JsonObj["MaxScore"]
    q = JsonObj["AVGQ"]
    Loss = JsonObj["AVGLoss"]
  else:
    Scores = []
    MeanScores = []
    TotalScore = 0
    NumberOfGames = 0
    BestScore = -999999999999999999
    q = []
    Loss = []
    
def handler(signum, frame): # To handle Ctrl + C Exiting
  #model.save()
  exit(1)

def Normalize(Min, X, Max): # To Normalise Values
  Normal = (X - Min) / (Max-Min)
  return Normal

def Remember(State, Action, Reward, NextState, Done): # For putting things into Memory
  Memory.append((State, Action, Reward, NextState, Done))

def LongMemTrain(): # Uses Memory to keep the last 100000 states, then batches them into Samples to be learnt by the reinforcement agent, bit different now considering no longer using QTrain
  global Tempq
  global TempLoss
  if len(Memory) > BatchSize: # Gets selection of memory to use for training that is the batchsize
    SelectionSample = random.sample(Memory, BatchSize) # List of Tuples
    States, Actions, Rewards, NextStates, Done = zip(*SelectionSample)
    TempFrameQ, TempFrameLoss = Trainer.Train(States, NextStates, Actions, Rewards, Done)
    Tempq.append(TempFrameQ)
    TempLoss.append(TempFrameLoss)
  else:
   pass

def ShortMemTrain(State, Action, Reward, NextState): # Leftover from QTraining
  Trainer.Train(State, NextState, Action, Reward)

def TradeOff(State0): # Takes in the predictions and
  Temp = [0] * 30 
  Action = 0
  if random.randint(0, TradeoffNum) < (TradeoffNum - NumberOfGames): # Performs a random action ((TradeoffNum - NumberOfGames) / 100) % of the time, and the bot action (100 - (TradeoffNum - NumberOfGames) / 100)) %of the time
    Temp[random.randint(0,29)] = 1
    Action = GetAction(Temp)
  else:
    Predictions = Trainer.NN(input=State0.to(device)).cpu().detach() 
    Action = GetAction(Predictions)
    #print(Action)
  return Action

def GetAction(Predictions): #Turns prediction into controller input
  Prediction = np.argmax(Predictions)
  Action = np.int16(Prediction + 1)
  match Action:
           case 1:
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.25, 0.5)
             #print("Left")
             #Left
           case 2:
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.75, 0.5)
             #print("Right")
             #Right
           case 3:
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.5, 0.75)
             #print("Up")
             #Up
           case 4:
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.5, 0.25)
             #print("Down")
             #Down
           case 5:
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.25, 0.75)
             #print("Left-Up")
             #Left-Up
           case 6:
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.75, 0.75)
             #print("Right-Up")
             #Right-Up
           case 7:
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.25, 0.25)
             #print("Left-Down")
             #Left-Down
           case 8:
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.75, 0.25)
             #print("Right-Down")
             #Right-Down
           case 9:
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.5, 0.5)
             #print("Neutral")
             #Neutral
           case 10:
             controller.press_button(melee.enums.Button.BUTTON_A)
             #print("A")
           case 11:
             controller.press_button(melee.enums.Button.BUTTON_B)
             #print("B")
           case 12:
             controller.press_button(melee.enums.Button.BUTTON_Y)
             #print("Y")
           case 13:
             controller.press_button(melee.enums.Button.BUTTON_L)
             #print("L")
           case 14:
             controller.press_button(melee.enums.Button.BUTTON_R)#
             #print("R")
           case 15:
             controller.press_button(melee.enums.Button.BUTTON_Z)
             #print("Z")
           case 16:
             controller.press_button(melee.enums.Button.BUTTON_A)
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.25, 0.5)
             #print("Left + A")
             #L+A
           case 17:
             controller.press_button(melee.enums.Button.BUTTON_B)
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.25, 0.5)
             #print("Left + B")
             #L+B
           case 18:
             controller.press_button(melee.enums.Button.BUTTON_L)
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.25, 0.5)
             #print("Left + R")
             #L+R
           case 19:
             controller.press_button(melee.enums.Button.BUTTON_A)
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.75, 0.5)
             #print("Right + A")
             #R+A
           case 20:
             controller.press_button(melee.enums.Button.BUTTON_B)
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.75, 0.5)
             #print("Right + B")
             #R+B
           case 21:
             controller.press_button(melee.enums.Button.BUTTON_R)
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.75, 0.5)
             #print("Right + R")
             #R+R
           case 22:
             controller.press_button(melee.enums.Button.BUTTON_A)
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.5, 0.75)
             #print("Up + A")
             #U+A
           case 23:
             controller.press_button(melee.enums.Button.BUTTON_B)
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.5, 0.75)
             #print("Up + B")
             #U+B
           case 24:
             controller.press_button(melee.enums.Button.BUTTON_A)
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.5, 0.25)
             #print("Down + A")
             #D+A
           case 25:
             controller.press_button(melee.enums.Button.BUTTON_B)
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.5, 0.25)
             #print("Down + B")
             #D+B
           case 26:
             controller.press_button(melee.enums.Button.BUTTON_R)
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.5, 0)
             #print("Down + R")
             #D+R
           case 27:
             controller.press_button(melee.enums.Button.BUTTON_A)
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 1, 0.5)
             #print("Smash Left")
             #SmashL
           case 28:
             controller.press_button(melee.enums.Button.BUTTON_A)
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0, 0.5)
             #print("Smash Right")
             #SmashR
           case 29:
             controller.press_button(melee.enums.Button.BUTTON_A)
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.5, 1)
             #print("Smash Up")
             #SmashU
           case 30:
             controller.press_button(melee.enums.Button.BUTTON_A)
             controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.5, 0)
             #print("Smash Down")
             #SmashD
  return Action-1  

def GetStates():
     InHitStunBot = 0
     InHitStunOpp = 0
     Actionable = 0
     if gamestate.players[1].hitlag_left > 0: # Calculate if the bot or opponent is in hitlag, if so, it sets the value to true for the input into the NN
      InHitStunBot = 1
     if gamestate.players[1].hitlag_left > 0:
      InHitStunOpp = 1
     if gamestate.players[1].action == melee.enums.Action.STANDING or gamestate.players[1].action == melee.enums.Action.CROUCHING:
       Actionable = 1


     Inputs = [gamestate.players[1].facing, gamestate.players[2].facing, # NN Inputs, in the same order as on the Ins and Outs.txt
     InHitStunBot, InHitStunOpp, 
     gamestate.players[1].on_ground, gamestate.players[1].percent / 999, 
     gamestate.players[2].percent / 999, Normalize(-246,gamestate.players[1].position.x, 246), 
     Normalize(-140,gamestate.players[1].position.y, 118), Normalize(-246,gamestate.players[2].position.x, 246),
     Normalize(-140,gamestate.players[2].position.y, 118), gamestate.players[1].stock / 4, 
     gamestate.players[2].stock / 4, Actionable]
     
     Inputs = [np.float32(i) for i in Inputs]
     Inputs = torch.tensor(Inputs, dtype=torch.float)
     return Inputs

def check_port(value):
    ivalue = int(value)
    if ivalue < 1 or ivalue > 4:
        raise argparse.ArgumentTypeError("%s is an invalid controller port. \
                                         Must be 1, 2, 3, or 4." % value)
    return ivalue

def CheckDeath(): # Checks if either character has died and gives rewards / punishments depending
  NewReward = 0
  global BotStocks
  global BotPercent
  global OppStocks
  global OppPercent

  if (gamestate.players[2].stock < OppStocks) and not(gamestate.players[1].stock == 0 and gamestate.players[2].stock == 0): #At the end of the game, the system reports both having 0 stocks, giving the bot a +10 for losing, this stops that
      NewReward += 10
      OppStocks = gamestate.players[2].stock
      OppPercent = 0

  if (gamestate.players[1].stock < BotStocks) and not(gamestate.players[1].stock == 0 and gamestate.players[2].stock == 0): #At the end of the game, the system reports both having 0 stocks, giving the bot a +10 for losing, this stops that
      NewReward -= 10
      BotStocks = gamestate.players[1].stock
      BotPercent = 0
  # if (gamestate.players[1].stock < BotStocks) and gamestate.players[1].percent > 50: # If the bot dies with above 50% hp, don't punish as hard, as it "tried to live"
  #     NewReward -= 5
  #     BotStocks = gamestate.players[1].stock
  #     #print(gamestate.player[1].percent)
  #     BotPercent = 0
  # elif (gamestate.players[1].stock < BotStocks) and gamestate.players[1].percent <= 10:
  #     NewReward -= 10
  #     #print(gamestate.player[1].percent)
  #     BotStocks = gamestate.players[1].stock
  #     BotPercent = 0

  return NewReward

def CheckPercentChange(): # Detect if the bot has done any damage to the opponent, reward accordingly
  NewReward = 0
  global BotPercent
  global OppPercent
  if gamestate.players[2].percent > OppPercent:
      NewReward += (0.1 * (gamestate.players[2].percent - OppPercent))
      OppPercent = gamestate.players[2].percent

  # if gamestate.players[1].percent > BotPercent:
  #     NewReward -= (0.1 * (gamestate.players[1].percent - BotPercent))
  #     BotPercent = gamestate.players[1].percent

  return NewReward

def CheckReward(Action):
  global PrevAction
  global PrevX
  global PrevY
  NewReward = 0

  NewReward += CheckDeath()
  NewReward += CheckPercentChange()

  # if PrevAction == np.argmax(Action): # Check if bot performs the same action over and over again
  #   NewReward -= 0.0001
  # else:
  #   NewReward += 0.0001

  # PrevAction == np.argmax(Action)

  # if (PrevX == gamestate.players[1].position.x or PrevY == gamestate.players[1].position.y): # Check if the bot is staying still, punish it if so
  #   NewReward -= (0.001)
  # else:
  #   NewReward += 0.001

  # PrevX = gamestate.players[1].position.x
  # PrevY = gamestate.players[1].position.y

  return NewReward

def LibmeleeSetup():
  global console
  global controller
  global controller_opponent
  global costume
  parser = argparse.ArgumentParser(description='Libmelee in action')
  parser.add_argument('--dolphin_executable_path', '-e', default=None, help='The directory where dolphin is')
  parser.add_argument('--connect_code', '-t', default="", help='Direct connect code to connect to in Slippi Online')
  args = parser.parse_args()

  console = melee.Console(path=args.dolphin_executable_path, blocking_input=True, online_delay=0)
  controller = melee.Controller(console=console, port=1, type=melee.ControllerType.STANDARD)
  controller_opponent = melee.Controller(console=console, port=2, type=melee.ControllerType.STANDARD)
  console.run()
  console.connect()
  controller.connect()
  controller_opponent.connect()
  costume = 1

def SetupVariables():
  global GamePlayed
  global CharSelected
  CharSelected = True
  GamePlayed = True

def JsonWrite(FileName):
  global OppPercent
  global BotPercent
  global OppStocks
  global BotStocks
  global NumberOfGames
  global Score
  global BestScore
  global TotalScore
  global Scores
  global MeanScores
  global q
  global Loss
  global Tempq
  global TempLoss
  NumberOfGames += 1
  OppPercent = 0
  BotPercent = 0
  OppStocks = 4
  BotStocks = 4
  q.append(np.average(Tempq))
  Loss.append(np.average(TempLoss))
  Tempq = []
  TempLoss = []
  if Score > BestScore:
    BestScore = Score
  Trainer.save(FileName=(FileName))
  Scores.append(Score)
  TotalScore += Score
  MeanScores.append(TotalScore/NumberOfGames)
  #ActionTemp = RecordedActions.tolist()
  PlotObj = {"Scores": Scores, "MeanScores": MeanScores, "AVGQ": q, "AVGLoss": Loss,"TotalScore": TotalScore, "NumberOfGames": NumberOfGames, "MaxScore": BestScore}
  with open("Plotting/"+ FileName + ".json", 'w') as f:
    json.dump(PlotObj, f)
  Score = 0
    
# def EvolutionTrainer(): # Carry over from Evolutionary Approach
#   pass
#   Counter = 0
#   Generating = 1
#   Population = []
#   NewPopulation = []
#   while Counter <= 999:
#     Population.append(PyTorchNN.NeuralNetwork(14, 30, 30))
#   Population.sort()
#   Parents = Population[0:99]
#   NewPopulation.append(Parents)
#   while Generating <= 10: #Generate OffSpring
#     for Index, Parent in enumerate(Parents[0:98]):
#       TempWeights = Parent.fc.weight[2]
#       Parent.fc.weight[2] = Parents[Index + 1].fc.weight[2]
#       Parents[Index + 1].fc.weight[2] = TempWeights
#       NewPopulation.append(Parent)
#       NewPopulation.append(Parents[Index + 1])
#     Generating += 1
#   #Mutate
#   for Child in NewPopulation[:100]:
#     if random.randint(0, 100) < 10:
#       Node = random.randint(0,29)
#       Layer = random.randint(0,2)
#       Child.fc.weight[Layer,Node] = Child.fc.weight[Layer,Node] * (1 + (random.random(-1,1) * 0.2))

JsonLoading(FileName=FileName)
NNSetup(FileName=FileName)
LibmeleeSetup()
signal.signal(signal.SIGINT, handler)
gamestate = console.step()
while True:
    #gamestate = console.step()
  if gamestate is None:
        gamestate = console.step()
        continue
  if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
        if StatesSetup == False:
          StateSetup()
          StatesSetup = True    
        controller.release_all()
        #melee.techskill.multishine(ai_state=gamestate.players[1], controller=controller)
        State0 = GetStates() #Get Current State #Get Prediction of action to take
        Action = TradeOff(State0)#Turn action from list into controller input and Calculate exploration vs exploitation
        gamestate = console.step() #Continue 1 frame#
        Reward = np.float32(CheckReward(Action)) # Calculate Reward
        State1 = GetStates()#Get New State  
        #print(Reward)
        Done = (gamestate.players[1].stock == 0 and gamestate.players[2].stock == 0)
        Remember(State0, Action, Reward, State1, Done)
        #ShortMemTrain(State0, Action, Reward, State1)
        Score += Reward
        #Temp = np.append(Temp, np.argmax(Action))
        SetupVariables()
        if gamestate.frame % 4 == 0:
          LongMemTrain()
  else:
        #RecordedActions = np.append(RecordedActions, Temp, 0)
        Temp = np.array([])
        if GamePlayed == True:
          JsonWrite(FileName=FileName)
          GamePlayed = False
          if gamestate.players[2].character_selected != melee.Character.MARTH or gamestate.players[2].cpu_level !=3:
            CharSelected = False
        gamestate = console.step()
        if SetupDone:
          if gamestate.menu_state in [melee.Menu.CHARACTER_SELECT]:
            if CharSelected == False:
              #melee.MenuHelper.menu_helper_simple(gamestate, controller_opponent, melee.Character.MARTH, melee.Stage.FINAL_DESTINATION, costume=costume, autostart=True, swag=False, cpu_level=9)
              pass
            if (gamestate.players[2].character_selected == melee.Character.MARTH) and (gamestate.players[2].cpu_level == 3):
              melee.MenuHelper.menu_helper_simple(gamestate, controller, melee.Character.FOX, melee.Stage.FINAL_DESTINATION, costume=costume, autostart=True, swag=False, cpu_level=0)
              #print("Yee")
          else:
            melee.MenuHelper.menu_helper_simple(gamestate, controller_opponent, melee.Character.MARTH, melee.Stage.FINAL_DESTINATION, costume=costume, autostart=True, swag=False, cpu_level=3)
        else:
          if gamestate.menu_state == melee.Menu.CHARACTER_SELECT:
            SetupDone = True
          

          




