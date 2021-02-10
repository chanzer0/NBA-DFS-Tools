import json
import csv
import os
import numpy as np
import pandas as pd
from pulp import *


class TennisOptimizer:
    site = None
    config = None
    problem = None
    output_dir = None
    num_lineups = None
    num_uniques = None
    pickle = None
    use_randomness = None
    lineups = {}
    player_dict = {}
    limit_rules = []

    def __init__(self, site=None, num_lineups=0, use_randomness=False, num_uniques=1):
        self.site = site
        self.num_lineups = int(num_lineups)
        self.num_uniques = int(num_uniques)
        self.use_randomness = use_randomness == 'rand'
        self.load_config()
        self.problem = LpProblem('Tennis', LpMaximize)

        projection_path = os.path.join(os.path.dirname( __file__), '../dk_data/tennis_projections.csv')
        self.load_projections(projection_path)

        pickle_path = os.path.join(os.path.dirname(__file__), '../dk_data/players.pickle')
        self.load_pickle(pickle_path)

        # ownership_path = os.path.join(os.path.dirname(__file__), '../{}_data/{}'.format(site, self.config['ownership_path']))
        # self.load_ownership(ownership_path)

        # boom_bust_path = os.path.join(os.path.dirname(__file__), '../{}_data/{}'.format(site, self.config['boom_bust_path']))
        # self.load_boom_bust(boom_bust_path)

        # self.load_limit_players(self.config["limit"])

    # Load config from file
    def load_config(self):
        with open(os.path.join(os.path.dirname(__file__), '../config.json')) as json_file:
            self.config = json.load(json_file)

    def load_pickle(self, path):
        self.pickle = pd.read_pickle(path)
        print(self.pickle)

    # Load player IDs for exporting
    def load_player_ids(self, path):
        with open(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                name_key = 'Name' if self.site == 'dk' else 'Nickname'
                player_name = row[name_key].replace('-', '#')
                if player_name in self.player_dict:
                    if self.site == 'dk':
                        self.player_dict[player_name]['ID'] = int(row['ID'])
                    else:
                        self.player_dict[player_name]['ID'] = row['Id']

    # Load projections from file
    def load_projections(self, path):
        # Read projections into a dictionary
        with open(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                self.player_dict[row['Name']] = {'Fpts': 0, 'ID': 0, 'Salary': 0, 'Name': '', 'StdDev': 0, 'Ownership': 0.1, 'Optimal': 0, 'Boom': 0, 'Bust': 0, 'Ceiling': 0}
                self.player_dict[row['Name']]['Fpts'] = float(row['DK Pts'])
                self.player_dict[row['Name']]['Salary'] = int(row['DK Salary'].replace(',', ''))
                self.player_dict[row['Name']]['Name'] = row['Name']

    # Load ownership from file
    def load_ownership(self, path):
        # Read ownership into a dictionary
        with open(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                player_name = row['Name'].replace('-', '#')
                if player_name in self.player_dict:
                    self.player_dict[player_name]['Ownership'] = float(row['Ownership %'])

    def optimize(self):
        # Setup our linear programming equation - https://en.wikipedia.org/wiki/Linear_programming
        # We will use PuLP as our solver - https://coin-or.github.io/pulp/

        # We want to create a variable for each roster slot.
        # There will be an index for each player and the variable will be binary (0 or 1) representing whether the player is included or excluded from the roster.
        lp_variables = {player: LpVariable(player, cat='Binary') for player, _ in self.player_dict.items()}

        # set the objective - maximize fpts
        self.problem += lpSum(self.player_dict[player]['Fpts'] * lp_variables[player] for player in self.player_dict), 'Objective'

        # Set the salary constraints
        self.problem += lpSum(self.player_dict[player]['Salary'] * lp_variables[player] for player in self.player_dict) <= 50000

        # 6 players
        self.problem += lpSum(lp_variables[player] for player in self.player_dict) == 6
        
        # Crunch!
        for i in range(self.num_lineups):
            try:
                self.problem.solve(PULP_CBC_CMD(msg=0))
            except PulpSolverError:
                print('Infeasibility reached - only generated {} lineups out of {}. Continuing with export.'.format(
                    len(self.num_lineups), self.num_lineups))

            score = str(self.problem.objective)
            for v in self.problem.variables():
                score = score.replace(v.name, str(v.varValue))

            if i % 100 == 0:
                print(i)

            player_names = [v.name for v in self.problem.variables() if v.varValue != 0]
            fpts = eval(score)
            self.lineups[fpts] = player_names

            # Dont generate the same lineup twice
            if self.use_randomness:
                # Enforce this by lowering the objective i.e. producing sub-optimal results
                self.problem += lpSum(np.random.normal(
                    self.player_dict[player]['Fpts'], self.player_dict[player]['StdDev']) * lp_variables[player] for player in self.player_dict)
            else:
                # Set a new random fpts projection within their distribution
                self.problem += lpSum(self.player_dict[player]['Fpts'] * lp_variables[player] for player in self.player_dict) <= (fpts - 0.01)

    def output(self):
        print('Lineups done generating. Outputting.')
        unique = {}
        for fpts, lineup in self.lineups.items():
            if lineup not in unique.values():
                unique[fpts] = lineup

        self.lineups = unique
        if self.num_uniques != 1:
            num_uniq_lineups = OrderedDict(sorted(self.lineups.items(), reverse=False, key=lambda t: t[0]))
            self.lineups = {}
            for fpts, lineup in num_uniq_lineups.copy().items():
                temp_lineups = list(num_uniq_lineups.values())
                temp_lineups.remove(lineup)
                use_lineup = True
                for x in temp_lineups:
                    if (6 - len(common_players)) < self.num_uniques:
                        use_lineup = False
                        del num_uniq_lineups[fpts]
                        break

                if use_lineup:
                    self.lineups[fpts] = lineup

        out_path = os.path.join(os.path.dirname(__file__), '../output/{}_optimal_tennis_lineups.csv'.format(self.site))
        with open(out_path, 'w') as f:
            f.write('P,P,P,P,P,P,Salary,Projection\n')
            for fpts, x in self.lineups.items():
                salary = sum(self.player_dict[player.replace('_',' ')]['Salary'] for player in x)
                fpts_p = sum(self.player_dict[player.replace('_',' ')]['Fpts'] for player in x)
                own_p = np.prod([self.player_dict[player.replace('_',' ')]['Ownership']/100.0 for player in x])
                lineup_str = '{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{},{},{}'.format(
                    x[0].replace('_',' '), self.player_dict[x[0].replace('_',' ')]['ID'],
                    x[1].replace('_',' '), self.player_dict[x[1].replace('_',' ')]['ID'],
                    x[2].replace('_',' '), self.player_dict[x[2].replace('_',' ')]['ID'],
                    x[3].replace('_',' '), self.player_dict[x[3].replace('_',' ')]['ID'],
                    x[4].replace('_',' '), self.player_dict[x[4].replace('_',' ')]['ID'],
                    x[5].replace('_',' '), self.player_dict[x[5].replace('_',' ')]['ID'],
                    salary, round(fpts_p, 2), own_p)
                f.write('%s\n' % lineup_str)
        print('Output done.')


class Player:
    def __init__(self, name, surface):
        self.name = name
        self.AcePct = stat_dict[name, surface]['AcePct']
        self.OppAcePct = stat_dict[name, surface]['OppAcePct']
        self.Opp1stWinPct = stat_dict[name, surface]['Opp1stWinPct']
        self.FirstWonPct = stat_dict[name, surface]['1stWinPct']
        self.SecondWonPct = stat_dict[name, surface]['2ndWinPct']
        self.Opp2ndWonPct = stat_dict[name, surface]['Opp2ndWinPct']
        self.FirstServePct = stat_dict[name, surface]['1stServePct']
        self.DFPct = stat_dict[name, surface]['DFPct']
        self.aces = 0
        self.games_won = 0
        self.ptswon = 0
        self.setswon = 0
        self.setsweep = 0
        self.matchsweep = 0
        self.matchwinner = 0
        self.breaks = 0
        self.svpt = 0
        self.SecondWon = 0
        self.FirstIn = 0
        self.FirstWon = 0
        self.df = 0
        self.server = 0
        self.dkfp = 0
        self.fdfp = 0
        self.total_aces = 0
        self.total_dfs = 0
        self.dkfp_list = []
        self.fdfp_list = []
        self.games_lost = 0
        self.sets_lost = 0
        self.matches_lost = 0
        self.acebonus = 0
        self.nodfbonus = 0
        self.total_games_won = 0
        self.total_sets_won = 0
        self.total_aces = 0
        self.total_first_in = 0
        self.total_first_won = 0
        self.total_second_won = 0
        self.total_df = 0
        self.total_pts_won = 0
        self.total_clean_sets = 0
        self.total_sweeps = 0
        self.total_breaks = 0
        self.total_svpt = 0
        self.total_ace_bonus = 0
        self.total_nodf_bonus = 0
        self.total_dkfp = 0
        self.total_fdfp = 0
        self.total_matches_won = 0
        self.total_games_lost = 0
        self.total_sets_lost = 0

    def to_dict(self, sims, opp):
        return {
            'name': self.name,
            'opp': opp.name,
            'aces': self.total_aces / sims,
            'games_won': self.total_games_won / sims,
            'ptswon': self.total_pts_won / sims,
            'setswon': self.total_sets_won / sims,
            'setsweep': self.total_clean_sets / sims,
            'matchsweep': self.total_sweeps / sims,
            'matchwinner': self.total_matches_won / sims,
            'breaks': self.total_breaks / sims,
            'svpt': self.total_svpt / sims,
            'SecondWon': self.total_second_won / sims,
            'FirstIn': self.total_first_in / sims,
            'FirstWon': self.total_first_won / sims,
            'df': self.total_df / sims,
            'nodfbonus': self.total_nodf_bonus / sims,
            'acebonus': self.total_ace_bonus / sims,
            'dkfp': self.total_dkfp / sims,
            'fdfp': self.total_fdfp/sims,
            'games_lost': self.total_games_lost / sims,
            'sets_lost': self.total_sets_lost / sims

        }

    def clear_counts(self):
        self.total_aces += self.aces
        self.aces = 0
        self.total_games_won += self.games_won
        self.games_won = 0
        self.total_pts_won += self.ptswon
        self.ptswon = 0
        self.total_sets_won += self.setswon
        self.setswon = 0
        self.total_clean_sets += self.setsweep
        self.setsweep = 0
        self.total_sweeps += self.matchsweep
        self.matchsweep = 0
        self.total_breaks += self.breaks
        self.breaks = 0
        self.total_svpt += self.svpt
        self.svpt = 0
        self.total_second_won += self.SecondWon
        self.SecondWon = 0
        self.total_first_in += self.FirstIn
        self.FirstIn = 0
        self.total_first_won += self.FirstWon
        self.FirstWon = 0
        self.total_df += self.df
        self.df = 0
        self.server = 0
        self.total_ace_bonus += self.acebonus
        self.acebonus = 0
        self.total_nodf_bonus += self.nodfbonus
        self.nodfbonus = 0
        self.fdfp_list.append(self.fdfp)
        self.total_fdfp += self.fdfp
        self.fdfp = 0
        self.total_matches_won += self.matchwinner
        self.matchwinner = 0
        self.dkfp_list.append(self.dkfp)
        self.total_dkfp += self.dkfp
        self.dkfp = 0
        self.total_games_lost += self.games_lost
        self.total_sets_lost += self.sets_lost
        self.games_lost = 0
        self.sets_lost = 0
