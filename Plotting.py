import matplotlib.pyplot as plt
from IPython import display

#plt.ion()

def plot(Scores, MeanScores):
    plt.plot(Scores)
    plt.plot(MeanScores)
    #plt.ylim(ymin=0)
    #plt.text(len(Scores)-1,Scores[-1], str(Scores[-1]))
    #plt.text(len(MeanScores)-1, MeanScores[-1], str(MeanScores[-1]))
    plt.show()
