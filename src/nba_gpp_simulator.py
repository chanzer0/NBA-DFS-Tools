import json, os, csv, math, random, heapq
import numpy as np
from pulp import *


class NBA_GPP_Simulator:
    config = None
    player_dict = {}
    field_lineups = {}
    roster_construction = []
    salary = None
    optimal_score = None
    field_size = None
    num_iterations = None
    site = None
    payout_structure = {}
    use_contest_data = False
    entry_fee = None

    def __init__(self, site, field_size, num_iterations, use_contest_data):
        self.site = site
        self.load_config()
        projection_path = os.path.join(os.path.dirname(__file__), '../{}_data/{}'.format(site, self.config['projection_path']))
        self.load_projections(projection_path)

        ownership_path = os.path.join(os.path.dirname(__file__), '../{}_data/{}'.format(site, self.config['ownership_path']))
        self.load_ownership(ownership_path)

        boom_bust_path = os.path.join(os.path.dirname(__file__), '../{}_data/{}'.format(site, self.config['boom_bust_path']))
        self.load_boom_bust(boom_bust_path)

        player_path = os.path.join(os.path.dirname(__file__), '../{}_data/{}'.format(site, self.config['player_path']))
        self.load_player_ids(player_path)

        if site == 'dk':
            self.roster_construction = ['PG', 'SG', 'SF', 'PF', 'C', 'F', 'G', 'UTIL']
            self.salary = 50000
        elif site == 'fd':
            self.roster_construction = ['PG', 'PG', 'SG', 'SG', 'SF', 'SF', 'PF', 'PF', 'C']
            self.salary = 60000

        self.use_contest_data = use_contest_data
        if use_contest_data:
            contest_path = os.path.join(os.path.dirname(__file__), '../{}_data/{}'.format(site, self.config['contest_structure_path']))
            self.load_contest_data(contest_path)
            print('Contest payout structure loaded.')
        else:
            self.field_size = int(field_size)
            
        self.num_iterations = int(num_iterations)
        self.get_optimal()

    # In order to make reasonable tournament lineups, we want to be close enough to the optimal that 
    # a person could realistically land on this lineup. Skeleton here is taken from base `nba_optimizer.py`
    def get_optimal(self):
        problem = LpProblem('NBA', LpMaximize)
        lp_variables = {player: LpVariable(player, cat='Binary') for player, _ in self.player_dict.items()}

        # set the objective - maximize fpts
        problem += lpSum(self.player_dict[player]['Fpts'] * lp_variables[player] for player in self.player_dict), 'Objective'

        # Set the salary constraints
        problem += lpSum(self.player_dict[player]['Salary'] * lp_variables[player] for player in self.player_dict) <= self.salary

        if self.site == 'dk':
            # Need at least 1 point guard, can have up to 3 if utilizing G and UTIL slots
            problem += lpSum(lp_variables[player] for player in self.player_dict if 'PG' in self.player_dict[player]['Position']) >= 1
            problem += lpSum(lp_variables[player] for player in self.player_dict if 'PG' in self.player_dict[player]['Position']) <= 3
            # Need at least 1 shooting guard, can have up to 3 if utilizing G and UTIL slots
            problem += lpSum(lp_variables[player] for player in self.player_dict if 'SG' in self.player_dict[player]['Position']) >= 1
            problem += lpSum(lp_variables[player] for player in self.player_dict if 'SG' in self.player_dict[player]['Position']) <= 3
            # Need at least 1 small forward, can have up to 3 if utilizing F and UTIL slots
            problem += lpSum(lp_variables[player] for player in self.player_dict if 'SF' in self.player_dict[player]['Position']) >= 1
            problem += lpSum(lp_variables[player] for player in self.player_dict if 'SF' in self.player_dict[player]['Position']) <= 3
            # Need at least 1 power forward, can have up to 3 if utilizing F and UTIL slots
            problem += lpSum(lp_variables[player] for player in self.player_dict if 'PF' in self.player_dict[player]['Position']) >= 1
            problem += lpSum(lp_variables[player] for player in self.player_dict if 'PF' in self.player_dict[player]['Position']) <= 3
            # Need at least 1 center, can have up to 2 if utilizing C and UTIL slots
            problem += lpSum(lp_variables[player] for player in self.player_dict if 'C' in self.player_dict[player]['Position']) >= 1
            problem += lpSum(lp_variables[player] for player in self.player_dict if 'C' in self.player_dict[player]['Position']) <= 2
            # Need at least 3 guards (PG,SG,G)
            problem += lpSum(lp_variables[player] for player in self.player_dict if 'PG' in self.player_dict[player]['Position'] or 'SG' in self.player_dict[player]['Position']) >= 3
            # Need at least 3 forwards (SF,PF,F)
            problem += lpSum(lp_variables[player] for player in self.player_dict if 'SF' in self.player_dict[player]['Position'] or 'PF' in self.player_dict[player]['Position']) >= 3
            # Can only roster 8 total players
            problem += lpSum(lp_variables[player] for player in self.player_dict) == 8
        else:
            # Need 2 PG
            problem += lpSum(lp_variables[player] for player in self.player_dict if 'PG' == self.player_dict[player]['Position']) == 2
            # Need 2 SG
            problem += lpSum(lp_variables[player] for player in self.player_dict if 'SG' == self.player_dict[player]['Position']) == 2
            # Need 2 SF
            problem += lpSum(lp_variables[player] for player in self.player_dict if 'SF' == self.player_dict[player]['Position']) == 2
            # Need 2 PF
            problem += lpSum(lp_variables[player] for player in self.player_dict if 'PF' == self.player_dict[player]['Position']) == 2
            # Need 1 center
            problem += lpSum(lp_variables[player] for player in self.player_dict if 'C' == self.player_dict[player]['Position']) == 1
            # Can only roster 9 total players
            problem += lpSum(lp_variables[player] for player in self.player_dict) == 9
        
        # Crunch!
        try:
            problem.solve(PULP_CBC_CMD(msg=0)) 
        except PulpSolverError:
            print('Infeasibility reached - only generated {} lineups out of {}. Continuing with export.'.format(len(self.num_lineups), self.num_lineups))

        score = str(problem.objective)
        for v in problem.variables():
            score = score.replace(v.name, str(v.varValue))
    
        self.optimal_score = eval(score)
        

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

    def load_contest_data(self, path):
        with open(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                if self.field_size is None:
                    self.field_size = int(row['Field Size'])
                if self.entry_fee is None:
                    self.entry_fee = float(row['Entry Fee'])
            
                # multi-position payouts
                if '-' in row['Place']:
                    indices = row['Place'].split('-')
                    for i in range(int(indices[0]), int(indices[1])):
                        # Where I'm from, we 0 index things. Thus, -1 since Payout starts at 1st place
                        self.payout_structure[i - 1] = float(row['Payout'].split('.')[0].replace(',',''))
                # single-position payouts
                else:
                    self.payout_structure[int(row['Place']) - 1] = float(row['Payout'].split('.')[0].replace(',',''))

    # Load config from file
    def load_config(self):
        with open(os.path.join(os.path.dirname(__file__), '../config.json')) as json_file: 
            self.config = json.load(json_file)
         
    # Load projections from file
    def load_projections(self, path):
        # Read projections into a dictionary
        with open(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                player_name = row['Name']
                self.player_dict[player_name] = {'Fpts': 0, 'Position': [], 'ID': 0, 'Salary': 0, 'StdDev': 0, 'Ownership': 0.1, 'In Lineup': False}
                self.player_dict[player_name]['Fpts'] = float(row['Fpts'])
                self.player_dict[player_name]['Salary'] = int(row['Salary'].replace(',',''))

                #some players have 2 positions - will be listed like 'PG/SF' or 'PF/C'
                if self.site == 'dk':
                    self.player_dict[player_name]['Position'] = [pos for pos in row['Position'].split('/')]

                    if 'PG' in self.player_dict[player_name]['Position'] or 'SG' in self.player_dict[player_name]['Position']:
                        self.player_dict[player_name]['Position'].append('G')

                    if 'SF' in self.player_dict[player_name]['Position'] or 'PF' in self.player_dict[player_name]['Position']:
                        self.player_dict[player_name]['Position'].append('F')

                    self.player_dict[player_name]['Position'].append('UTIL')
                elif self.site == 'fd':
                    self.player_dict[player_name]['Position'] = row['Position']

   
    # Load ownership from file
    def load_ownership(self, path):
        # Read ownership into a dictionary
        with open(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                player_name = row['Name']
                if player_name in self.player_dict:
                    self.player_dict[player_name]['Ownership'] = float(row['Ownership %'])
    
    # Load standard deviations
    def load_boom_bust(self, path):
        with open(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                player_name = row['Name']
                if player_name in self.player_dict:
                    self.player_dict[player_name]['StdDev'] = float(row['Std Dev'])

    def select_random_player(self, position):
        plyr_list = []
        prob_list = []
        for player in self.player_dict:
            if self.player_dict[player]['In Lineup'] == False:
                should_use = False
                if self.site == 'dk':
                    if position in self.player_dict[player]['Position']:
                        should_use = True

                elif self.site ==  'fd':
                    if position == self.player_dict[player]['Position']:
                        should_use = True
                
                if should_use:
                    plyr_list.append(player)
                    prob_list.append(self.player_dict[player]['Ownership'])        
                    
        prob_list = [float(i)/sum(prob_list) for i in prob_list]
        return np.random.choice(a=plyr_list, p=prob_list)

    def generate_field_lineups(self):
        print('Generating ' + str(self.field_size) + ' lineups.')
        for i in range(self.field_size):
            reject = True
            while reject:
                salary = 0
                lineup = []
                for player in self.player_dict:
                    self.player_dict[player]['In Lineup'] = False
                for pos in self.roster_construction:
                    x = self.select_random_player(pos)
                    self.player_dict[x]['In Lineup'] = True
                    lineup.append(x)
                    salary += self.player_dict[x]['Salary']
                # Must have a reasonable salary
                reasonable_salary = self.salary - 1000 if self.site == 'dk' else self.salary - 1500
                if (salary >= reasonable_salary and salary <= self.salary):
                    # Must have a reasonable projection (within 10% of optimal)
                    reasonable_projection = self.optimal_score - (0.10*self.optimal_score)
                    if (sum(self.player_dict[player]['Fpts'] for player in lineup) >= reasonable_projection):
                        reject = False
                        if i % 1000 == 0:
                            print(i)
            self.field_lineups[i] = {'Lineup': lineup, 'Wins': 0, 'Top10': 0, 'ROI': 0}
        print(str(self.field_size) + ' field lineups successfully generated')

    def run_tournament_simulation(self):
        print('Running ' + str(self.num_iterations) + ' simulations')
        for i in range(self.num_iterations):
            temp_fpts_dict = {p: round((np.random.normal(stats['Fpts'], stats['StdDev'])), 2) for p,stats in self.player_dict.items()}
            field_score = {}

            for index,values in self.field_lineups.items():
                fpts_sim = sum(temp_fpts_dict[player] for player in values['Lineup'])
                field_score[fpts_sim] = {'Lineup': values['Lineup'], 'Fpts': fpts_sim, 'Index': index}

            # If we're using contest data, we need to calculate ROI for the lineups - sort them descending and assign payouts
            if self.use_contest_data:
                sorted_dict = dict(sorted(field_score.items(), reverse=True))
                for i, (k,values) in enumerate(sorted_dict.items()):
                    idx = values['Index']
                    # If this lineup "placed"
                    if i in self.payout_structure:
                        self.field_lineups[idx]['ROI'] += (self.payout_structure[i] - self.entry_fee)
                    # Else, this lineup lost money (entry fee)
                    else:
                        self.field_lineups[idx]['ROI'] = self.field_lineups[idx]['ROI'] - self.entry_fee

                    # Winning
                    if i == 0:
                        self.field_lineups[idx]['Wins'] += 1
                    # Top 10
                    if i < 10:
                        self.field_lineups[idx]['Top10'] +=  1
            else:
                # Get the top 10 scores for the sim
                top_10 = heapq.nlargest(10, field_score.values(), key=lambda x: x['Fpts'])
                for lineup in top_10:
                    if lineup == top_10[0]:
                        self.field_lineups[lineup['Index']]['Wins'] += 1

                    self.field_lineups[lineup['Index']]['Top10'] += 1

        print(str(self.num_iterations) + ' tournament simulations finished. Outputting.')

    def output(self):
        unique = {}
        for index, x in self.field_lineups.items():
            salary = sum(self.player_dict[player]['Salary'] for player in x['Lineup'])
            fpts_p = sum(self.player_dict[player]['Fpts'] for player in x['Lineup'])
            own_p = np.prod([self.player_dict[player]['Ownership']/100.0 for player in x['Lineup']])
            win_p = round(x['Wins']/self.num_iterations * 100, 2)
            top10_p = round(x['Top10']/self.num_iterations * 100, 2)
            if self.site == 'dk':
                if self.use_contest_data:
                    roi_p = round(x['ROI']/self.entry_fee/self.num_iterations * 100, 2)
                    lineup_str = '{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{},{},{}%,{}%,{}%,{}'.format(
                        x['Lineup'][0], self.player_dict[x['Lineup'][0]]['ID'],
                        x['Lineup'][1], self.player_dict[x['Lineup'][1]]['ID'],
                        x['Lineup'][2], self.player_dict[x['Lineup'][2]]['ID'],
                        x['Lineup'][3], self.player_dict[x['Lineup'][3]]['ID'],
                        x['Lineup'][4], self.player_dict[x['Lineup'][4]]['ID'],
                        x['Lineup'][5], self.player_dict[x['Lineup'][5]]['ID'],
                        x['Lineup'][6], self.player_dict[x['Lineup'][6]]['ID'],
                        x['Lineup'][7], self.player_dict[x['Lineup'][7]]['ID'],
                        fpts_p,salary,win_p,top10_p,roi_p,own_p
                    )
                else:
                    lineup_str = '{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{},{},{}%,{}%,{}'.format(
                        x['Lineup'][0], self.player_dict[x['Lineup'][0]]['ID'],
                        x['Lineup'][1], self.player_dict[x['Lineup'][1]]['ID'],
                        x['Lineup'][2], self.player_dict[x['Lineup'][2]]['ID'],
                        x['Lineup'][3], self.player_dict[x['Lineup'][3]]['ID'],
                        x['Lineup'][4], self.player_dict[x['Lineup'][4]]['ID'],
                        x['Lineup'][5], self.player_dict[x['Lineup'][5]]['ID'],
                        x['Lineup'][6], self.player_dict[x['Lineup'][6]]['ID'],
                        x['Lineup'][7], self.player_dict[x['Lineup'][7]]['ID'],
                        fpts_p,salary,win_p,top10_p,own_p
                    )
            elif self.site == 'fd':
                if self.use_contest_data:
                    roi_p = round(x['ROI']/self.entry_fee/self.num_iterations * 100, 2)
                    lineup_str = '{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{},{},{}%,{}%,{}%,{}'.format(
                        self.player_dict[x['Lineup'][0]]['ID'], x['Lineup'][0],
                        self.player_dict[x['Lineup'][1]]['ID'], x['Lineup'][1],
                        self.player_dict[x['Lineup'][2]]['ID'], x['Lineup'][2],
                        self.player_dict[x['Lineup'][3]]['ID'], x['Lineup'][3],
                        self.player_dict[x['Lineup'][4]]['ID'], x['Lineup'][4],
                        self.player_dict[x['Lineup'][5]]['ID'], x['Lineup'][5],
                        self.player_dict[x['Lineup'][6]]['ID'], x['Lineup'][6],
                        self.player_dict[x['Lineup'][7]]['ID'], x['Lineup'][7],
                        self.player_dict[x['Lineup'][8]]['ID'], x['Lineup'][8],
                        fpts_p,salary,win_p,top10_p,roi_p,own_p
                    )
                else:
                    lineup_str = '{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{},{},{}%,{}%,{}'.format(
                        self.player_dict[x['Lineup'][0]]['ID'], x['Lineup'][0],
                        self.player_dict[x['Lineup'][1]]['ID'], x['Lineup'][1],
                        self.player_dict[x['Lineup'][2]]['ID'], x['Lineup'][2],
                        self.player_dict[x['Lineup'][3]]['ID'], x['Lineup'][3],
                        self.player_dict[x['Lineup'][4]]['ID'], x['Lineup'][4],
                        self.player_dict[x['Lineup'][5]]['ID'], x['Lineup'][5],
                        self.player_dict[x['Lineup'][6]]['ID'], x['Lineup'][6],
                        self.player_dict[x['Lineup'][7]]['ID'], x['Lineup'][7],
                        self.player_dict[x['Lineup'][8]]['ID'], x['Lineup'][8],
                        fpts_p,salary,win_p,top10_p,own_p
                    )
            unique[index] = lineup_str

        out_path = os.path.join(os.path.dirname(__file__), '../output/{}_gpp_sim_lineups_{}_{}.csv'.format(self.site, self.field_size, self.num_iterations))
        with open(out_path, 'w') as f:
            if self.site == 'dk':
                if self.use_contest_data:
                    f.write('PG,SG,SF,PF,C,G,F,UTIL,Fpts Proj,Salary,Win %,Top 10%,ROI%,Proj. Own. Product\n')
                else:
                    f.write('PG,SG,SF,PF,C,G,F,UTIL,Fpts Proj,Salary,Win %,Top 10%,Proj. Own. Product\n')
            elif self.site == 'fd':
                if self.use_contest_data:
                    f.write('PG,PG,SG,SG,SF,SF,PF,PF,C,Fpts Proj,Salary,Win %,Top 10%,ROI%,Proj. Own. Product\n')
                else:
                    f.write('PG,PG,SG,SG,SF,SF,PF,PF,C,Fpts Proj,Salary,Win %,Top 10%,Proj. Own. Product\n')

            for fpts, lineup_str in unique.items():
                f.write('%s\n' % lineup_str)
    
        out_path = os.path.join(os.path.dirname(__file__), '../output/{}_gpp_sim_player_exposure_{}_{}.csv'.format(self.site, self.field_size, self.num_iterations))
        with open(out_path, 'w') as f:
            f.write('Player,Win%,Top10%,Sim. Own%,Proj. Own%\n')
            unique_players = {}
            for val in self.field_lineups.values():
                for player in val['Lineup']:
                    if player not in unique_players:
                        unique_players[player] = {'Wins': val['Wins'], 'Top10':val['Top10'], 'In': 1}
                    else:
                        unique_players[player]['Wins'] = unique_players[player]['Wins'] + val['Wins']
                        unique_players[player]['Top10'] = unique_players[player]['Top10'] + val['Top10']
                        unique_players[player]['In'] = unique_players[player]['In'] + 1

            for player,data in unique_players.items():
                field_p = round(data['In']/self.field_size * 100, 2)
                win_p = round(data['Wins']/self.num_iterations * 100, 2)
                top10_p = round(data['Top10']/self.num_iterations / 10 * 100, 2)
                proj_own = self.player_dict[player]['Ownership']
                f.write('{},{}%,{}%,{}%,{}%\n'.format(player, win_p, top10_p, field_p, proj_own))
        
