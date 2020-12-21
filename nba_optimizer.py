import json
import csv
from pulp import *

class NBA_Optimizer:
    problem = None
    config = None
    output_filepath = None
    num_lineups = None
    player_projections = {'PG': {}, 'SG': {}, 'SF': {}, 'PF': {}, 'C': {}, 'F': {}, 'G': {}, 'UTIL': {}}
    player_salaries = {'PG': {}, 'SG': {}, 'SF': {}, 'PF': {}, 'C': {}, 'F': {}, 'G': {}, 'UTIL': {}}
    player_ownership = {'PG': {}, 'SG': {}, 'SF': {}, 'PF': {}, 'C': {}, 'F': {}, 'G': {}, 'UTIL': {}}
    roster_construction = {'PG': 1, 'SG': 1, 'SF': 1, 'PF': 1, 'C': 1, 'F': 1, 'G': 1, 'UTIL': 1}
    player_ids = {}
    max_salary = 50000
    lineups = []

    def __init__(self):
        self.problem = LpProblem('NBA', LpMaximize)
        self.load_config()
        self.load_projections(self.config['projection_path'])
        self.load_ownership(self.config['ownership_path'])
        self.load_player_ids(self.config['player_path'])
        self.output_filepath = self.config['output_path']
        self.num_lineups = self.config['num_lineups']

    # Load config from file
    def load_config(self):
        with open('config.json') as json_file: 
            self.config = json.load(json_file) 

    def load_player_ids(self, path):
        with open(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                self.player_ids[row['Name']] = row['ID']

    # Load projections from file
    def load_projections(self, path):
        # Read projections into a dictionary
        with open(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Handle multi-position eligibility 
                if '/' in row['Position']:
                    pos_1 = row['Position'].split('/')[0]
                    pos_2 = row['Position'].split('/')[1]
                    # projection
                    self.player_projections[pos_1][row['Name']] = float(row['Fpts'])
                    self.player_projections[pos_2][row['Name']] = float(row['Fpts'])

                    # salary 
                    self.player_salaries[pos_1][row['Name']] = int(row['Salary'].replace(',',''))
                    self.player_salaries[pos_2][row['Name']] = int(row['Salary'].replace(',',''))

                else:
                    # projection
                    self.player_projections[row['Position']][row['Name']] = float(row['Fpts'])

                    # salary 
                    self.player_salaries[row['Position']][row['Name']] = int(row['Salary'].replace(',',''))

            # Add SF and PF to F position
            sf_players = self.player_projections['SF']
            pf_players = self.player_projections['PF']
            self.player_projections['F'] = {**sf_players, **pf_players}
            self.player_salaries['F'] = {**sf_players, **pf_players}
            
            # Add PG and SG to G position
            pg_players = self.player_projections['PG'] 
            sg_players = self.player_projections['SG'] 
            self.player_projections['G'] = {**pg_players, **sg_players}
            self.player_salaries['G'] = {**pg_players, **sg_players}
            
            # Add all players to UTIL position
            c_players = self.player_projections['C']
            self.player_projections['UTIL'] = {**pg_players, **sg_players, **sf_players, **pf_players, **c_players}
            self.player_salaries['UTIL'] = {**pg_players, **sg_players, **sf_players, **pf_players, **c_players}

            print(self.player_projections)

    # Load ownership from file
    def load_ownership(self, path):
        # Read ownership into a dictionary
        with open(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                 # Handle multi-position eligibility 
                if '/' in row['Position']:
                    pos_1 = row['Position'].split('/')[0]
                    pos_2 = row['Position'].split('/')[1]
                    self.player_ownership[pos_1][row['Name']] = row['Ownership %']
                    self.player_ownership[pos_2][row['Name']] = row['Ownership %']
                else:
                    self.player_ownership[row['Position']][row['Name']] = row['Ownership %']

    def optimize(self):
        # Setup our linear programming equation - https://en.wikipedia.org/wiki/Linear_programming
        # We will use PuLP as our solver - https://coin-or.github.io/pulp/

        # We want to create a variable for each roster slot. 
        # There will be an index for each player and the variable will be binary (0 or 1) representing whether the player is included or excluded from the roster.
        lp_variables = {pos: LpVariable.dict(pos, players, cat='Binary') for pos, players in self.player_projections.items()}

        projection_constraints = []
        salary_constraints = []
        position_constraints = []

        for position, players in lp_variables.items():
            # Set salary constraints
            salary_constraints += lpSum([self.player_salaries[position][player] * lp_variables[position][player] for player in players])
            # Set projections to maximize
            projection_constraints += lpSum([self.player_projections[position][player] * lp_variables[position][player] for player in players])
            # Set positional constraints
            self.problem += lpSum([lp_variables[position][player] for player in players]) == self.roster_construction[position]

        self.problem += lpSum(projection_constraints)
        self.problem += lpSum(salary_constraints) <= self.max_salary
        self.problem.solve()

    def output(self):
        div = '---------------------------------------\n'
        print('Variables:\n')
        score = str(self.problem.objective)
        constraints = [str(const) for const in self.problem.constraints.values()]
        for v in self.problem.variables():
            score = score.replace(v.name, str(v.varValue))
            constraints = [const.replace(v.name, str(v.varValue)) for const in constraints]
            if v.varValue != 0:
                print(v.name, '=', v.varValue)
        print(div)
        print('Constraints:')
        for constraint in constraints:
            constraint_pretty = ' + '.join(re.findall('[0-9\.]*\*1.0', constraint))
            if constraint_pretty != '':
                print('{} = {}'.format(constraint_pretty, eval(constraint_pretty)))
        print(div)
        print('Score:')
        score_pretty = ' + '.join(re.findall('[0-9\.]+\*1.0', score))
        print('{} = {}'.format(score_pretty, eval(score)))
        # with open(self.output_filepath, 'w') as f:
        #     f.write('QB,RB,RB,WR,WR,WR,TE,FLEX,DST,Fpts,Salary,Ownership\n')
        #     for x in final:
        #         fpts = sum(float(projection_dict[player]['Fpts']) for player in x)
        #         salary = sum(int(projection_dict[player]['Salary'].replace(',','')) for player in x)
        #         own = sum(float(ownership_dict[player]) for player in x)
        #         lineup_str = '{},{},{},{},{},{},{},{},{},{},{},{}'.format(
        #             x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],fpts,salary,own
        #         )
        #         f.write('%s\n' % lineup_str)