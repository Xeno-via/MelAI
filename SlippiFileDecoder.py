from time import sleep
from slippi import Game
import tkinter as tk
from tkinter import Button, filedialog

root = tk.Tk()
root.withdraw()

def OpenFile():
    Counter = 59
    #FilePath = filedialog.askopenfilename()
    #SlpFile = open("C:\\Users\\Finn_\\Documents\\New folder\\2022-04\\Game_20220406T232304")
    SlpObject = Game('C:\\Users\\Finn_\\Documents\\New folder\\2022-04\\Game_20220406T232304.slp')
    
    while Counter < len(SlpObject.frames):
        Button = str(SlpObject.frames[Counter].ports[0].leader.pre.buttons.logical.name)
        if Button != "NONE":
            hi = SlpObject.frames[Counter].ports[0].leader.pre.buttons.physical.pressed
        Counter = Counter + 1

OpenFile()