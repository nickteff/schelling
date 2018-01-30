import matplotlib as mpl
import matplotlib.colors as col
import matplotlib.pyplot as plt
import itertools
import numpy as np
import pandas as pd
import copy

mpl.rcParams.update(mpl.rcParamsDefault)
e = ['#ffffff',
     '#009FD7',
     '#E30010',
     '#f39200',
     '#528EA5',
     '#004C64',
     '#E84932',
     '#74BCBF',
     '#fed630',
     '#75D0F4',
     '#EBA289']

cmap = col.LinearSegmentedColormap.from_list('', [col.hex2color(color) for color in e], N=len(e))

class Schelling:
    def __init__(self, N, empty_ratio, similarity_threshold, races=2):
        self.N = N
        self.races = races
        self.empty_ratio = empty_ratio
        self.similarity_threshold = similarity_threshold
        self.empty_houses = []
        self.households = {}
        self.map = []
        self.houses_by_race = {}
        self.satisfied = []

    def is_satisfied(self, x):

        if x not in self.households.keys():
            satisfied = True

        elif x[0] in [0, self.N+1] or x[1] in [0, self.N+1]:
            satisfied = True

        else:
            same = len(np.where(np.isin(self.map[x[0]-1:x[0]+2, x[1]-1:x[1]+2],
                                [self.map[x[0],x[1]], 0]))[0])-1

            if same >= 8*self.similarity_threshold:
                satisfied = True
            else:
                satisfied = False

        return satisfied

    def update(self):

        if len(self.map) == 0:
            prob = [self.empty_ratio] + [(1 - self.empty_ratio)/(self.races) for i in range(self.races)]

            self.map = np.random.choice(self.races+1, size=(self.N+1)**2, replace=True, p=prob).reshape(N+1, N+1)

            self.empty_houses = np.argwhere(self.map==0)

            self.houses_by_race = {i: np.argwhere(self.map==i) for i in np.arange(1, self.races+1)}

            self.households = {tuple(k) : self.map[k[0], k[1]] for k in np.argwhere(self.map>0)}

        else:
            self.satisfied = np.array([self.is_satisfied(tuple(x)) for x in np.argwhere(self.map>-1)]).reshape(self.N+1, self.N+1)

            unsatisfied = np.random.permutation(np.argwhere(self.satisfied==False))

            available = np.random.permutation(np.concatenate([self.empty_houses, unsatisfied]))

            for k in range(len(unsatisfied)):
                avail = tuple(available[k])
                self.map[avail[0], avail[1]] = self.households[tuple(unsatisfied[k])]

            now_empty = np.delete(available, np.s_[:len(unsatisfied)], 0)

            for k in now_empty:
                self.map[k[0], k[1]] = 0

            self.empty_houses = np.argwhere(self.map==0)

            self.houses_by_race = {i: np.argwhere(self.map==i) for i in np.arange(1, self.races+1)}

            self.households = {tuple(k) : self.map[k[0], k[1]] for k in np.argwhere(self.map>0)}

            self.satisfied = np.array([self.is_satisfied(tuple(x)) for x in np.argwhere(self.map>-1)]).reshape(self.N+1, self.N+1)

            if len(np.argwhere(self.satisfied==False)) == 0:
                print("Simulation stable")



from matplotlib import animation


def animate_schelling(n):
    pass

N = 100
races = 2
empty_ratio = .1
similarity_threshold = .75


schelling = Schelling(N, empty_ratio, similarity_threshold, races)

schelling.update()

fig, ax = plt.subplots(figsize=(6,6))
#fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
#ax.axis('off')
#ax.set(xlim=(-1, 1), ylim=(-1, 1))
ax.set(title="hi")


animate = []


for i in range(200):
    if i % 5 == 4:
        schelling.update()
        animate.append([plt.imshow(schelling.map[1:schelling.N, 1:schelling.N], cmap=cmap)])

anim = animation.ArtistAnimation(fig, animate, interval=250, repeat_delay=500,
                                   blit=True)
plt.show()
