
from __future__ import division

from multiprocessing import Process
import os
import pandas as pd
import re
import numpy as np
import random
import scipy.stats as ss
import matplotlib.pyplot as plt
from datetime import timedelta
import datetime as dt
import timeit
#from pulp import *
from collections import namedtuple
from datetime import datetime
from pytz import timezone
from typing import List, Optional
from pydfs_lineup_optimizer.utils import process_percents
from pydfs_lineup_optimizer.tz import get_timezone


boom = pd.read_csv('boom_bust.csv', thousands=',')

GameInfo = namedtuple('GameInfo', ['home_team', 'away_team', 'starts_at', 'game_started'])

class Player:
    def __init__(self,
                 player_id: str,
                 first_name: str,
                 last_name: str,
                 positions: List[str],
                 team: str,
                 salary: float,
                 proj: float,
                 sd: float,
                 is_injured: bool = False,
                 max_exposure: Optional[float] = None,
                 min_exposure: Optional[float] = None,
                 projected_ownership: Optional[float] = None,
                 game_info: Optional[GameInfo] = None,
                 roster_order: Optional[int] = None,
                 min_deviation: Optional[float] = None,
                 max_deviation: Optional[float] = None,
                 is_confirmed_starter: Optional[bool] = None
                 ):
        self.id = player_id
        self.first_name = first_name
        self.last_name = last_name
        self.positions = positions
        self.team = team
        self.salary = salary
        self.fppg = 0
        self.proj = proj
        self.sd = sd
        self.in_optimal = 0
        self.in_optimal_proj = []
        self.is_injured = is_injured
        self.game_info = game_info
        self.roster_order = roster_order
        self.is_mvp = False  # type: bool
        self.is_star = False  # type: bool
        self.is_pro = False  # type: bool
        self._min_exposure = None  # type: Optional[float]
        self._max_exposure = None  # type: Optional[float]
        self._min_deviation = None  # type: Optional[float]
        self._max_deviation = None  # type: Optional[float]
        self._projected_ownership = None  # type: Optional[float]
        self.min_exposure = min_exposure
        self.max_exposure = max_exposure
        self.min_deviation = min_deviation
        self.max_deviation = max_deviation
        self.projected_ownership = projected_ownership
        self.is_confirmed_starter = is_confirmed_starter
        self.in_optimal_lineup = 0
        self.in_lineup = 0
        self.in_winning_lineup = 0
        self.calcPos()
#
#    def __repr__(self):
#        return '%s %s (%s)' % (self.full_name, '/'.join(self.positions), self.team)

    def __hash__(self):
        return hash(self.id)
    
    def randomizeFPPG(self):
        self.fppg = np.random.normal(loc=self.proj, scale=self.sd)
        
    def calcPos(self):
        if "PG" in self.positions:
            self.positions.append("G")
        elif "SG" in self.positions:
            self.positions.append("G")
        elif "SF" in self.positions:
            self.positions.append("F")
        elif "PF" in self.positions:
            self.positions.append("F")
        self.positions.append("UTIL")

    @property
    def max_exposure(self) -> Optional[float]:
        return self._max_exposure

    @max_exposure.setter
    def max_exposure(self, max_exposure: Optional[float]):
        self._max_exposure = process_percents(max_exposure)

    @property
    def min_exposure(self) -> Optional[float]:
        return self._min_exposure

    @min_exposure.setter
    def min_exposure(self, min_exposure: Optional[float]):
        self._min_exposure = process_percents(min_exposure)

    @property
    def min_deviation(self) -> Optional[float]:
        return self._min_deviation

    @min_deviation.setter
    def min_deviation(self, min_deviation: Optional[float]):
        self._min_deviation = process_percents(min_deviation)

    @property
    def max_deviation(self) -> Optional[float]:
        return self._max_deviation

    @max_deviation.setter
    def max_deviation(self, max_deviation: Optional[float]):
        self._max_deviation = process_percents(max_deviation)

    @property
    def projected_ownership(self) -> Optional[float]:
        return self._projected_ownership

    @projected_ownership.setter
    def projected_ownership(self, projected_ownership: Optional[float]):
        self._projected_ownership = process_percents(projected_ownership)

    @property
    def full_name(self) -> str:
        return '{} {}'.format(self.first_name, self.last_name)

    @property
    def efficiency(self) -> float:
        return round(self.fppg / self.salary, 6)

plyrs = []
for index,row in boom.iterrows():
    p = Player(positions = row['Position'].split('/'), first_name = '', last_name= row['Name'], player_id = index, team=row['Team'], salary = row['Salary'], proj = row['Projection'], sd=row['Std Dev'], projected_ownership = row['Ownership%'])
    plyrs.append(p)


def select_random_player(pos):
    plyr_list = []
    prob_list =[]
    for p in plyrs:
        if p.in_lineup == 0:
            if pos in p.positions:
                plyr_list.append(p)
                prob_list.append(p.projected_ownership)
    prob_list = [float(i)/sum(prob_list) for i in prob_list]
    return np.random.choice(a=plyr_list, p=prob_list)    
    
pos_list = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F']        

def generateLineups(plyrs, pos_list, size):
    all_lus = []
    for i in range(size):
        reject = True
        while reject == True:
            salary = 0
            lineup = []
            for p in plyrs:
                p.in_lineup = 0
            for pos in pos_list:
                x = select_random_player(pos)
                x.in_lineup = 1  
                lineup.append(x)
                salary += x.salary 
            if (47000 <= salary <= 50000):
                reject= False
        all_lus.append(lineup)
    return all_lus

#def simulateGPP(all_lus):    

all_lus = generateLineups(plyrs, pos_list, 100000)

winning_scores = []
for i in range(10000):
    lu_fps = []
    for p in plyrs:
        p.randomizeFPPG()
    for l in all_lus:
        fp = 0
        for p in l:
            fp += p.fppg
        lu_fps.append(fp)    
    for p in all_lus[np.argmax(lu_fps)]:
        p.in_winning_lineup += 1
        winning_scores.append(max(lu_fps))