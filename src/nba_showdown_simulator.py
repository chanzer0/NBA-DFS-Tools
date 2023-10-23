import csv
import json
import math
import os
import random
import time, datetime
import numpy as np
import pulp as plp
import multiprocessing as mp
import pandas as pd
import statistics

# import fuzzywuzzy
import itertools
import collections
import re
from scipy.stats import norm, kendalltau, multivariate_normal, gamma
import matplotlib.pyplot as plt
import seaborn as sns
from numba import njit, jit
import sys

@jit(nopython=True)  
def salary_boost(salary, max_salary):
    return (salary / max_salary) ** 2

class nba_showdown_simulator:
    config = None
    player_dict = {}
    field_lineups = {}
    stacks_dict = {}
    gen_lineup_list = []
    roster_construction = []
    id_name_dict = {}
    salary = None
    optimal_score = None
    field_size = None
    team_list = []
    num_iterations = None
    site = None
    payout_structure = {}
    use_contest_data = False
    entry_fee = None
    use_lineup_input = None
    matchups = set()
    projection_minimum = 15
    randomness_amount = 100
    min_lineup_salary = 48000
    max_pct_off_optimal = 0.4
    teams_dict = collections.defaultdict(list)  # Initialize teams_dict
    correlation_rules = {}
    game_info = {}
    overlap_limit = 69
    team_replacement_dict = {
        'PHO': 'PHX',
        'GS': 'GSW',
        'SA': 'SAS',
        'NO': 'NOP',
        'NY': 'NYK',
    }

    def __init__(
        self,
        site,
        field_size,
        num_iterations,
        use_contest_data,
        use_lineup_input,
    ):
        self.site = site
        self.use_lineup_input = use_lineup_input
        self.load_config()
        self.load_rules()

        projection_path = os.path.join(
            os.path.dirname(__file__),
            "../{}_data/{}".format(site, self.config["projection_path"]),
        )
        self.load_projections(projection_path)

        player_path = os.path.join(
            os.path.dirname(__file__),
            "../{}_data/{}".format(site, self.config["player_path"]),
        )
        self.load_player_ids(player_path)
        self.seen_lineups = {}
        self.seen_lineups_ix = {}

        # ownership_path = os.path.join(
        #    os.path.dirname(__file__),
        #    "../{}_data/{}".format(site, self.config["ownership_path"]),
        # )
        # self.load_ownership(ownership_path)

        # boom_bust_path = os.path.join(
        #    os.path.dirname(__file__),
        #    "../{}_data/{}".format(site, self.config["boom_bust_path"]),
        # )
        # self.load_boom_bust(boom_bust_path)

        #       batting_order_path = os.path.join(
        #           os.path.dirname(__file__),
        #            "../{}_data/{}".format(site, self.config["batting_order_path"]),
        #        )
        #        self.load_batting_order(batting_order_path)

        if site == "dk":
            self.salary = 50000
            self.roster_construction = ["CPT", "UTIL", "UTIL", "UTIL", "UTIL", "UTIL"]

        elif site == "fd":
            self.salary = 60000
            self.roster_construction = ["MVP", "STAR", "PRO", "UTIL", "UTIL"]

        self.use_contest_data = use_contest_data
        if use_contest_data:
            contest_path = os.path.join(
                os.path.dirname(__file__),
                "../{}_data/{}".format(site, self.config["contest_structure_path"]),
            )
            self.load_contest_data(contest_path)
            print("Contest payout structure loaded.")
        else:
            self.field_size = int(field_size)
            self.payout_structure = {0: 0.0}
            self.entry_fee = 0

        # self.adjust_default_stdev()
        self.assertPlayerDict()
        self.num_iterations = int(num_iterations)
        self.get_optimal()
        if self.use_lineup_input:
            self.load_lineups_from_file()
        # if self.match_lineup_input_to_field_size or len(self.field_lineups) == 0:
        # self.generate_field_lineups()
        self.load_correlation_rules()

    # make column lookups on datafiles case insensitive
    def lower_first(self, iterator):
        return itertools.chain([next(iterator).lower()], iterator)
    
    @staticmethod
    def extract_matchup_time(game_string):
        # Extract the matchup, date, and time
        match = re.match(r"(\w{2,4}@\w{2,4}) (\d{2}/\d{2}/\d{4}) (\d{2}:\d{2}[APM]{2} ET)", game_string)

        if match:
            matchup, date, time = match.groups()
            # Convert 12-hour time format to 24-hour format
            time_obj = datetime.datetime.strptime(time, "%I:%M%p ET")
            # Convert the date string to datetime.date
            date_obj = datetime.datetime.strptime(date, "%m/%d/%Y").date()
            # Combine date and time to get a full datetime object
            datetime_obj = datetime.datetime.combine(date_obj, time_obj.time())
            return matchup, datetime_obj
        return None
    
    def load_rules(self):
        self.projection_minimum = int(self.config["projection_minimum"])
        self.randomness_amount = float(self.config["randomness"])
        self.min_lineup_salary = int(self.config["min_lineup_salary"])
        self.max_pct_off_optimal = float(self.config["max_pct_off_optimal"])
        self.default_var = float(self.config["default_var"])
        self.correlation_rules = self.config["custom_correlations"]

    def assertPlayerDict(self):
        for p, s in list(self.player_dict.items()):
            if s["ID"] == 0 or s["ID"] == "" or s["ID"] is None:
                print(
                    s["Name"]
                    + " name mismatch between projections and player ids, excluding from player_dict"
                )
                self.player_dict.pop(p)

    # In order to make reasonable tournament lineups, we want to be close enough to the optimal that
    # a person could realistically land on this lineup. Skeleton here is taken from base `mlb_optimizer.py`
    def get_optimal(self):
        problem = plp.LpProblem("NBA", plp.LpMaximize)
        
        lp_variables = {}
        for player, attributes in self.player_dict.items():
            player_id = attributes['ID']
            lp_variables[(player, player_id)] = plp.LpVariable(name=f"{player}_{player_id}", cat=plp.LpBinary)

        # set the objective - maximize fpts & set randomness amount from config
        problem += (
            plp.lpSum(
                self.player_dict[player]['fieldFpts']
                * lp_variables[(player, attributes['ID'])]
                for player, attributes in self.player_dict.items()
            ),
            "Objective",
        )
        
        # Set the salary constraints
        max_salary = 50000 if self.site == 'dk' else 60000
        min_salary = 48000 if self.site == 'dk' else 0

        if self.projection_minimum is not None:
            min_salary = self.min_lineup_salary
        
        # Maximum Salary Constraint
        problem += (
            plp.lpSum(
                self.player_dict[player]["Salary"]
                * lp_variables[(player, attributes['ID'])]
                for player, attributes in self.player_dict.items()
            )
            <= max_salary,
            "Max Salary",
        )

        # Minimum Salary Constraint
        problem += (
            plp.lpSum(
                self.player_dict[player]["Salary"]
                * lp_variables[(player, attributes['ID'])]
                for player, attributes in self.player_dict.items()
            )
            >= min_salary,
            "Min Salary",
        )

        if self.site == 'dk':
            # 1 CPT
            cpt_players = [(player, attributes['ID']) for player, attributes in self.player_dict.items() if player[1] == "CPT"]
            problem += plp.lpSum(lp_variables[player] for player in cpt_players) == 1, "Must have 1 CPT"
            
            # 5 UTIL
            util_players = [(player, attributes['ID']) for player, attributes in self.player_dict.items() if player[1] == "UTIL"]
            problem += plp.lpSum(lp_variables[player] for player in util_players) == 5, "Must have 5 UTIL"
            
            # Constraint to ensure each player is only selected once
            players_grouped_by_name = {v['Name']: [] for v in self.player_dict.values()}
            for key, value in self.player_dict.items():
                players_grouped_by_name[value['Name']].append(key)

            player_key_groups = [group for group in players_grouped_by_name.values()]
                    
            for player_key_list in player_key_groups:
                problem += (
                    plp.lpSum(lp_variables[(pk, self.player_dict[pk]['ID'])] for pk in player_key_list) <= 1,
                    f"Can only select {player_key_list} once",
                )
            
            # Max 5 players from one team 
            for team in self.team_list:
                problem += plp.lpSum(
                    lp_variables[(player, attributes['ID'])]
                    for player, attributes in self.player_dict.items()
                    if team in attributes['Team']
                ) <= 5, f"Max 5 players from {team}"

                
        else:
            # 1 MVP
            mvp_players = [(player,attributes['ID']) for player, attributes in self.player_dict.items() if player[1] == "MVP"]
            problem += plp.lpSum(lp_variables[player] for player in mvp_players) == 1, "Must have 1 MVP"
            
            # 1 STAR
            star_players = [(player, attributes['ID']) for player, attributes in self.player_dict.items() if player[1] == "STAR"]
            problem += plp.lpSum(lp_variables[player] for player in star_players) == 1, "Must have 1 STAR"
            
            # 1 PRO
            pro_players = [(player, attributes['ID']) for player, attributes in self.player_dict.items() if player[1] == "PRO"]
            problem += plp.lpSum(lp_variables[player] for player in pro_players) == 1, "Must have 1 PRO"
            
            # 2 UTIL
            util_players = [(player, attributes['ID']) for player, attributes in self.player_dict.items() if player[1] == "UTIL"]
            problem += plp.lpSum(lp_variables[player] for player in util_players) == 2, "Must have 2 UTIL"

            # Max 4 players from one team 
            for team in self.team_list:
                problem += plp.lpSum(
                    lp_variables[(player, attributes['ID'])]
                    for player, attributes in self.player_dict.items()
                    if team in attributes['Team']
                ) <= 4, f"Max 4 players from {team}"

            # Constraint to ensure each player is only selected once
            players_grouped_by_name = {v['Name']: [] for v in self.player_dict.values()}
            for key, value in self.player_dict.items():
                players_grouped_by_name[value['Name']].append(key)

            player_key_groups = [group for group in players_grouped_by_name.values()]
                    
            for player_key_list in player_key_groups:
                problem += (
                    plp.lpSum(lp_variables[(pk, self.player_dict[pk]['ID'])] for pk in player_key_list) <= 1,
                    f"Can only select {player_key_list} once",
                )

        # Crunch!
        problem.writeLP("problem.lp")
        try:
            problem.solve(plp.PULP_CBC_CMD(msg=0))
        except plp.PulpSolverError:
            print('Infeasibility reached - only generated {} lineups out of {}')

        # Get the lineup and add it to our list
        selected_vars = [player for player in lp_variables if lp_variables[player].varValue != 0]

        #print(selected_vars)
        fpts_proj = sum(self.player_dict[player[0]]["fieldFpts"] for player in selected_vars)
        # sal_used = sum(self.player_dict[player]["Salary"] for player in players)

        var_values = [
            value.varValue for value in problem.variables() if value.varValue != 0
        ]

        #print(fpts_proj, selected_vars, var_values)
        self.optimal_score = float(fpts_proj)

    def load_player_ids(self, path):
        with open(path, encoding='utf-8-sig') as file:
            reader = csv.DictReader(self.lower_first(file))
            for row in reader:
                name_key = 'name' if self.site == 'dk' else 'nickname'
                player_name = row[name_key].replace('-', '#')
                team = row['teamabbrev'] if self.site == 'dk' else row['team']
                game_info_str = row["game info"] if self.site == "dk" else row["game"]
                game_info = "game info" if self.site == "dk" else "game"
                result = self.extract_matchup_time(game_info_str)
                if result:
                    matchup, game_time = result
                    self.game_info[matchup] = game_time
                match = re.search(pattern="(\w{2,4}@\w{2,4})", string=row[game_info])
                if match:
                    opp = match.groups()[0].split("@")
                    
                    # Replace team abbreviations using a list comprehension
                    opp = [self.team_replacement_dict[o] if o in self.team_replacement_dict else o for o in opp]

                    self.matchups.add(tuple(opp))
                    
                    for m in opp:
                        if m != team:
                            team_opp = m

                    opp = tuple(opp)
                position = row['roster position']
                
                if team in self.team_replacement_dict:
                    team = self.team_replacement_dict[team]
                    
                if self.site == 'dk':
                    if (player_name, position, team) in self.player_dict:
                        self.player_dict[(player_name, position, team)]['ID'] = row['id']
                        self.player_dict[(player_name, position, team)]["Opp"] = team_opp
                        self.player_dict[(player_name, position, team)]["Matchup"] = opp
                        self.player_dict[(player_name, position, team)]['GameTime'] = game_time
                        self.player_dict[(player_name, position, team)]['UniqueKey'] = row['id']
                else:
                    for pos in ["MVP", "STAR", "PRO", "UTIL"]:
                            if (player_name, pos, team) in self.player_dict:
                                self.player_dict[(player_name, pos, team)]['ID'] = row['id'].replace('-', '#')
                                self.player_dict[(player_name, pos, team)]["Opp"] = team_opp
                                self.player_dict[(player_name, pos, team)]["Matchup"] = opp
                                self.player_dict[(player_name, pos, team)]['UniqueKey'] =  f"{pos}:{row['id'].replace('-', '#')}"
                                    
    def load_contest_data(self, path):
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(self.lower_first(file))
            for row in reader:
                if self.field_size is None:
                    self.field_size = int(row["field size"])
                if self.entry_fee is None:
                    self.entry_fee = float(row["entry fee"])
                # multi-position payouts
                if "-" in row["place"]:
                    indices = row["place"].split("-")
                    # print(indices)
                    # have to add 1 to range to get it to generate value for everything
                    for i in range(int(indices[0]), int(indices[1]) + 1):
                        # print(i)
                        # Where I'm from, we 0 index things. Thus, -1 since Payout starts at 1st place
                        if i >= self.field_size:
                            break
                        self.payout_structure[i - 1] = float(
                            row["payout"].split(".")[0].replace(",", "")
                        )
                # single-position payouts
                else:
                    if int(row["place"]) >= self.field_size:
                        break
                    self.payout_structure[int(row["place"]) - 1] = float(
                        row["payout"].split(".")[0].replace(",", "")
                    )
        # print(self.payout_structure)

    def load_correlation_rules(self):
        if len(self.correlation_rules.keys()) > 0:
            for primary_player in self.correlation_rules.keys():
                # Convert primary_player to the consistent format
                formatted_primary_player = primary_player.replace("-", "#").lower().strip()
                for (player_name, pos_str, team), player_data in self.player_dict.items():
                    if formatted_primary_player == player_name:
                        for second_entity, correlation_value in self.correlation_rules[primary_player].items():
                            # Convert second_entity to the consistent format
                            formatted_second_entity = second_entity.replace("-", "#").lower().strip()

                            # Check if the formatted_second_entity is a player name
                            found_second_entity = False
                            for (se_name, se_pos_str, se_team), se_data in self.player_dict.items():
                                if formatted_second_entity == se_name:
                                    player_data["Player Correlations"][formatted_second_entity] = correlation_value
                                    se_data["Player Correlations"][formatted_primary_player] = correlation_value
                                    found_second_entity = True
                                    break

                            # If the second_entity is not found as a player, assume it's a position and update 'Correlations'
                            if not found_second_entity:
                                player_data["Correlations"][second_entity] = correlation_value



    # Load config from file
    def load_config(self):
        with open(
            os.path.join(os.path.dirname(__file__), "../config.json"),
            encoding="utf-8-sig",
        ) as json_file:
            self.config = json.load(json_file)
    
        # Load projections from file
    def load_projections(self, path):
        # Read projections into a dictionary
        with open(path, encoding='utf-8-sig') as file:
            reader = csv.DictReader(self.lower_first(file))
            for row in reader:
                player_name = row['name'].replace('-', '#')
                try:
                    fpts = float(row["fpts"])
                except:
                    fpts = 0
                    print(
                        "unable to load player fpts: "
                        + player_name
                        + ", fpts:"
                        + row["fpts"]
                    )
                if fpts < self.projection_minimum:
                    continue
                if "fieldfpts" in row:
                    if row["fieldfpts"] == "":
                        fieldFpts = fpts
                    else:
                        fieldFpts = float(row["fieldfpts"])
                else:
                    fieldFpts = fpts
                position = [pos for pos in row["position"].split("/")]
                pos = position[0]
                if "stddev" in row:
                    if row["stddev"] == "" or float(row["stddev"]) == 0:
                            stddev = fpts * self.default_var
                    else:
                        stddev = float(row["stddev"])
                else:
                    stddev = fpts * self.default_var
                # check if ceiling exists in row columns
                if "ceiling" in row:
                    if row["ceiling"] == "" or float(row["ceiling"]) == 0:
                        ceil = fpts + stddev
                    else:
                        ceil = float(row["ceiling"])
                else:
                    ceil = fpts + stddev
                if row["salary"]:
                    sal = int(row["salary"].replace(",", ""))
                if "minutes" in row:
                    mins = row['minutes']
                else:
                    mins = 0
                if pos == "PG":
                    corr = {
                        "PG": 1,
                        "SG": -0.066989,
                        "SF": -0.066989,
                        "PF": -0.066989,
                        "C": -0.043954,
                        "Opp PG": 0.020682,
                        "Opp SG": 0.020682,
                        "Opp SF": 0.015477,
                        "Opp PF": 0.015477,
                        "Opp C": 0.000866,
                    }
                elif pos == "SG":
                    corr = {
                        "PG": -0.066989,
                        "SG": 1,
                        "SF": -0.066989,
                        "PF": -0.066989,
                        "C": -0.043954,
                        "Opp PG": 0.020682,
                        "Opp SG": 0.020682,
                        "Opp SF": 0.015477,
                        "Opp PF": 0.015477,
                        "Opp C": 0.000866,
                    }
                elif pos == "SF":
                    corr = {
                        "PG": -0.066989,
                        "SG": -0.066989,
                        "SF": 1,
                        "PF": -0.002143,
                        "C": -0.082331,
                        "Opp PG": 0.015477,
                        "Opp SG": 0.015477,
                        "Opp SF": 0.015477,
                        "Opp PF": 0.015477,
                        "Opp C": -0.012331,
                    }
                elif pos == "PF":
                    corr = {
                        "PG": -0.066989,
                        "SG": -0.066989,
                        "SF": -0.002143,
                        "PF": 1,
                        "C": -0.082331,
                        "Opp PG": 0.015477,
                        "Opp SG": 0.015477,
                        "Opp SF": 0.015477,
                        "Opp PF": 0.015477,
                        "Opp C": -0.012331,
                    }
                elif pos == "C":
                    corr = {
                        "PG": -0.043954,
                        "SG": -0.043954,
                        "SF": -0.082331,
                        "PF": -0.082331,
                        "C": 1,
                        "Opp PG": 0.000866,
                        "Opp SG": 0.000866,
                        "Opp SF": -0.012331,
                        "Opp PF": -0.012331,
                        "Opp C": -0.073081,
                    }
                team = row["team"]
                if team in self.team_replacement_dict:
                    team = self.team_replacement_dict[team]
                own = float(row["own%"].replace("%", ""))
                if own == 0:
                    own = 0.1
                pos_str = str(position)
                
                team = row['team']
                if team in self.team_replacement_dict:
                    team = self.team_replacement_dict[team]
                
                if self.site == 'dk':
                    self.player_dict[(player_name, "CPT", team)] = {
                        'Fpts': 1.5 * fpts,
                        'Salary': 1.5 * sal,
                        'Minutes': mins,
                        'Name': row['name'],
                        'Team': team,
                        'Ownership': float(row['cptown%']),
                        'StdDev': 1.5 * stddev,
                        'Position': pos,
                        'fieldFpts' : 1.5*fieldFpts,
                        'rosterPosition' : 'CPT',
                    }
                    player_data = {
                        'Fpts': fpts,
                        'Salary': sal,
                        'Minutes': mins,
                        'Name': row['name'],
                        'Team': team,
                        'Ownership': float(row['own%']),
                        'StdDev': stddev,
                        'Position': pos,
                        'fieldFpts' : fieldFpts,
                        'rosterPosition' : 'UTIL',
                        "Correlations": corr,
                        "Player Correlations": {},
                    }
                    self.player_dict[(player_name, "UTIL", team)] = player_data
                    self.teams_dict[team].append(player_data)
                else:
                    self.player_dict[(player_name, "MVP", team)] = {
                        'Fpts': 2 * fpts,
                        'Salary': sal,
                        'Minutes': mins,
                        'Name': row['name'],
                        'Team': team,
                        'Ownership': float(row['mvpown%']),
                        'StdDev': 2 * stddev,
                        'Position': pos,
                        'fieldFpts': 2*fieldFpts,
                        'rosterPosition' : 'MVP'
                    }
                    self.player_dict[(player_name, "STAR", team)] = {
                        'Fpts': 1.5 * fpts,
                        'Salary': sal,
                        'Minutes': mins,
                        'Name': row['name'],
                        'Team': team,
                        'Ownership': float(row['starown%']),
                        'StdDev': 1.5 * stddev,
                        'Position': pos,
                        'fieldFpts': 1.5*fieldFpts,
                        'rosterPosition' : 'STAR'
                    }
                    self.player_dict[(player_name, "PRO", team)] = {
                        'Fpts': 1.2 * fpts,
                        'Salary': sal,
                        'Minutes': mins,
                        'Name': row['name'],
                        'Team': team,
                        'Ownership': float(row['proown%']),
                        'StdDev': 1.2 * stddev,
                        'Position': pos,
                        'fieldFpts': 1.2*fieldFpts,
                        'rosterPosition' : 'PRO'
                    }
                    player_data = {
                        'Fpts': fpts,
                        'Salary': sal,
                        'Minutes': mins,
                        'Name': row['name'],
                        'Team': team,
                        'Ownership': own,
                        'StdDev': stddev,
                        'Position': pos,
                        'fieldFpts': fieldFpts,
                        'rosterPosition' : 'UTIL',
                        "Correlations": corr,
                        "Player Correlations": {},
                    }
                    self.player_dict[(player_name, "UTIL", team)] = player_data
                    self.teams_dict[team].append(player_data)
                            
                if team not in self.team_list:
                    self.team_list.append(team)

    def extract_id(self, cell_value):
        if "(" in cell_value and ")" in cell_value:
            return cell_value.split("(")[1].replace(")", "")
        elif ":" in cell_value:
            return cell_value.split(":")[0].replace("-", "#")
        else:
            return cell_value

    def load_lineups_from_file(self):
        print("loading lineups")
        i = 0
        path = os.path.join(
            os.path.dirname(__file__),
            "../{}_data/{}".format(self.site, "tournament_lineups.csv"),
        )
        with open(path) as file:
            reader = pd.read_csv(file)
            lineup = []
            j = 0
            for i, row in reader.iterrows():
                # print(row)
                if i == self.field_size:
                    break
                lineup = [
                    self.extract_id(str(row[j]))
                    for j in range(len(self.roster_construction))
                ]
                # storing if this lineup was made by an optimizer or with the generation process in this script
                error = False
                for l in lineup:
                    ids = [self.player_dict[k]["ID"] for k in self.player_dict]
                    if l not in ids:
                        print("lineup {} is missing players {}".format(i, l))
                        if l in self.id_name_dict:
                            print(self.id_name_dict[l])
                        error = True
                if len(lineup) < len(self.roster_construction):
                    print("lineup {} is missing players".format(i))
                    continue
                # storing if this lineup was made by an optimizer or with the generation process in this script
                error = False
                if self.site == "fd":
                    un_key_lu = []
                    i = 0
                    for l in lineup:
                        l = l.replace("-", "#")
                        ids = [self.player_dict[k]["ID"] for k in self.player_dict]
                        if l not in ids:
                            print("lineup {} is missing players {}".format(i, l))
                            if l in self.id_name_dict:
                                print(self.id_name_dict[l])
                            error = True
                        else:
                            for k in self.player_dict:
                                if self.player_dict[k]["ID"] == l:
                                    if i == 0:
                                        if (
                                            self.player_dict[k]["rosterPosition"]
                                            == "MVP"
                                        ):
                                            un_key_lu.append(
                                                self.player_dict[k]["UniqueKey"]
                                            )
                                        else:
                                            pass
                                    elif i == 1:
                                        if (
                                            self.player_dict[k]["rosterPosition"]
                                            == "STAR"
                                        ):
                                            un_key_lu.append(
                                                self.player_dict[k]["UniqueKey"]
                                            )
                                        else:
                                            pass
                                    elif i == 2:
                                        if (
                                            self.player_dict[k]["rosterPosition"]
                                            == "PRO"
                                        ):
                                            un_key_lu.append(
                                                self.player_dict[k]["UniqueKey"]
                                            )
                                        else:
                                            pass
                                    else:
                                        if (
                                            self.player_dict[k]["rosterPosition"]
                                            == "UTIL"
                                        ):
                                            un_key_lu.append(
                                                self.player_dict[k]["UniqueKey"]
                                            )
                                        else:
                                            pass
                        i += 1
                if len(lineup) < len(self.roster_construction):
                    print("lineup {} is missing players".format(i))
                    continue
                lu = lineup if self.site == "dk" else un_key_lu
                if not error:
                    self.field_lineups[j] = {
                        "Lineup": {
                            "Lineup": lu,
                            "Wins": 0,
                            "Top10": 0,
                            "ROI": 0,
                            "Cashes": 0,
                            "Type": "input",
                        },
                        "count": 1,
                    }
                    j += 1
        print("loaded {} lineups".format(j))
        # print(self.field_lineups)

    @staticmethod
    def select_player(
        pos,
        in_lineup,
        ownership,
        ids,
        salaries,
        current_salary,
        remaining_salary,
        k,
        rng,
        salary_ceiling=None,
        salary_floor=None,
        def_opp=None,
        teams=None,
    ):
        valid_players = np.nonzero(
            (pos > 0)
            & (in_lineup == 0)
            & (salaries <= remaining_salary)
            & (
                (current_salary + salaries >= salary_floor)
                if salary_floor is not None
                else True
            )
            # & ((teams != def_opp) if def_opp is not None else True)
        )[0]
        if len(valid_players) == 0:
            # common_indices = set(np.where(pos > 0)[0]) & \
            #     set(np.where(in_lineup == 0)[0]) & \
            #     set(np.where(salaries <= remaining_salary)[0]) & \
            #     set(np.where((current_salary + salaries >= salary_floor) if salary_floor is not None else True)[0]) & \
            #     set(np.where((teams != def_opp) if def_opp is not None else True)[0])
            # print(common_indices)
            # print(current_salary, salary_floor, remaining_salary, k, np.where((current_salary + salaries >= salary_floor)), np.where(pos>0), np.where(salaries <= remaining_salary), np.where(in_lineup == 0), np.where(teams != def_opp) if def_opp is not None else True)
            return None, None
        plyr_list = ids[valid_players]
        prob_list = ownership[valid_players] / ownership[valid_players].sum()
        if salary_ceiling:
            boosted_salaries = np.array([salary_boost(s, salary_ceiling) for s in salaries[valid_players]])
            boosted_probabilities = prob_list * boosted_salaries
            boosted_probabilities /= boosted_probabilities.sum()  # normalize to ensure it sums to 1
            choice = rng.choice(plyr_list, p=boosted_probabilities)
        else:
            choice = rng.choice(plyr_list,p=prob_list)
        return np.where(ids == choice)[0], choice

    @staticmethod
    def validate_lineup(
        salary,
        salary_floor,
        salary_ceiling,
        proj,
        optimal_score,
        max_pct_off_optimal,
        player_teams,
    ):
        reasonable_projection = optimal_score - (max_pct_off_optimal * optimal_score)

        if (
            salary_floor <= salary <= salary_ceiling
            and proj >= reasonable_projection
            and len(set(player_teams)) > 1
        ):
            return True
        return False

    @staticmethod
    def generate_lineups(
        lu_num,
        ids,
        in_lineup,
        pos_matrix,
        ownership,
        salary_floor,
        salary_ceiling,
        optimal_score,
        salaries,
        projections,
        max_pct_off_optimal,
        teams,
        opponents,
        overlap_limit,
        matchups,
        new_player_dict,
        num_players_in_roster,
        site
    ):
        rng = np.random.Generator(np.random.PCG64())
        lus = {}
        in_lineup.fill(0)
        iteration_count = 0
        while True:
            iteration_count += 1
            salary, proj = 0, 0
            lineup, player_teams, lineup_matchups = [], [], []
            def_opp, players_opposing_def, cpt_selected = None, 0, False
            in_lineup.fill(0)
            cpt_name = None
            remaining_salary = salary_ceiling

            for k, pos in enumerate(pos_matrix.T):
                position_constraint = k >= 1 and players_opposing_def < overlap_limit
                choice_idx, choice = nba_showdown_simulator.select_player(
                    pos,
                    in_lineup,
                    ownership,
                    ids,
                    salaries,
                    salary,
                    remaining_salary,
                    k,
                    rng,
                    salary_ceiling if k == num_players_in_roster - 1 else None,
                    salary_floor if k == num_players_in_roster - 1 else None,
                    def_opp if position_constraint else None,
                    teams if position_constraint else None,
                )
                if choice is None:
                    iteration_count += 1
                    salary, proj = 0, 0
                    lineup, player_teams, lineup_matchups = [], [], []
                    def_opp, players_opposing_def, cpt_selected = None, 0, False
                    in_lineup.fill(0)
                    cpt_name = None
                    remaining_salary = salary_ceiling
                    continue
                if site == 'dk':
                    if k == 0:
                        cpt_player_info = new_player_dict[choice]
                        UTIL_choice_idx = next(
                            (
                                i
                                for i, v in enumerate(new_player_dict.values())
                                if v["Name"] == cpt_player_info["Name"]
                                and v["Team"] == cpt_player_info["Team"]
                                and v["Position"] == cpt_player_info["Position"]
                                and v["rosterPosition"] == "UTIL"
                            ),
                            None,
                        )
                        if UTIL_choice_idx is not None:
                            in_lineup[UTIL_choice_idx] = 1
                        def_opp = opponents[choice_idx][0]
                        cpt_selected = True
                if site == 'fd':
                    if k == 0:
                        mvp_player_info = new_player_dict[choice]
                        STAR_choice_idx = next(
                            (
                                i
                                for i, v in enumerate(new_player_dict.values())
                                if v["Name"] == mvp_player_info["Name"]
                                and v["Team"] == mvp_player_info["Team"]
                                and v["Position"] == mvp_player_info["Position"]
                                and v["rosterPosition"] == "STAR"
                            ),
                            None,
                        )
                        if STAR_choice_idx is not None:
                            in_lineup[STAR_choice_idx] = 1
                        PRO_choice_idx = next(
                            (
                                i
                                for i, v in enumerate(new_player_dict.values())
                                if v["Name"] == mvp_player_info["Name"]
                                and v["Team"] == mvp_player_info["Team"]
                                and v["Position"] == mvp_player_info["Position"]
                                and v["rosterPosition"] == "PRO"
                            ),
                            None,
                        )
                        if PRO_choice_idx is not None:
                            in_lineup[PRO_choice_idx] = 1                     
                        UTIL_choice_idx = next(
                            (
                                i
                                for i, v in enumerate(new_player_dict.values())
                                if v["Name"] == mvp_player_info["Name"]
                                and v["Team"] == mvp_player_info["Team"]
                                and v["Position"] == mvp_player_info["Position"]
                                and v["rosterPosition"] == "UTIL"
                            ),
                            None,
                        )
                        if UTIL_choice_idx is not None:
                            in_lineup[UTIL_choice_idx] = 1
                        def_opp = opponents[choice_idx][0]
                    elif k == 1:
                        star_player_info = new_player_dict[choice]
                        PRO_choice_idx = next(
                            (
                                i
                                for i, v in enumerate(new_player_dict.values())
                                if v["Name"] == star_player_info["Name"]
                                and v["Team"] == star_player_info["Team"]
                                and v["Position"] == star_player_info["Position"]
                                and v["rosterPosition"] == "PRO"
                            ),
                            None,
                        )
                        if PRO_choice_idx is not None:
                            in_lineup[PRO_choice_idx] = 1
                        UTIL_choice_idx = next(
                            (
                                i
                                for i, v in enumerate(new_player_dict.values())
                                if v["Name"] == star_player_info["Name"]
                                and v["Team"] == star_player_info["Team"]
                                and v["Position"] == star_player_info["Position"]
                                and v["rosterPosition"] == "UTIL"
                            ),
                            None,
                        )
                        if UTIL_choice_idx is not None:
                            in_lineup[UTIL_choice_idx] = 1
                    elif k == 2:
                        pro_player_info = new_player_dict[choice]
                        UTIL_choice_idx = next(
                            (
                                i
                                for i, v in enumerate(new_player_dict.values())
                                if v["Name"] == pro_player_info["Name"]
                                and v["Team"] == pro_player_info["Team"]
                                and v["Position"] == pro_player_info["Position"]
                                and v["rosterPosition"] == "UTIL"
                            ),
                            None,
                        )
                        if UTIL_choice_idx is not None:
                            in_lineup[UTIL_choice_idx] = 1
                         
                        
                lineup.append(str(choice))
                in_lineup[choice_idx] = 1
                salary += salaries[choice_idx]
                proj += projections[choice_idx]
                remaining_salary = salary_ceiling - salary

                # lineup_matchups.append(matchups[choice_idx[0]])
                player_teams.append(teams[choice_idx][0])

                if teams[choice_idx][0] == def_opp:
                    players_opposing_def += 1

            if nba_showdown_simulator.validate_lineup(
                salary,
                salary_floor,
                salary_ceiling,
                proj,
                optimal_score,
                max_pct_off_optimal,
                player_teams,
            ):
                lus[lu_num] = {
                    "Lineup": lineup,
                    "Wins": 0,
                    "Top10": 0,
                    "ROI": 0,
                    "Cashes": 0,
                    "Type": "generated",
                }
                break
        return lus

    def remap_player_dict(self, player_dict):
        remapped_dict = {}
        for key, value in player_dict.items():
            if "UniqueKey" in value:
                player_id = value["UniqueKey"]
                remapped_dict[player_id] = value
            else:
                raise KeyError(f"Player details for {key} does not contain an 'ID' key")
        return remapped_dict

    def generate_field_lineups(self):
        diff = self.field_size - len(self.field_lineups)
        if diff <= 0:
            print(
                f"supplied lineups >= contest field size. only retrieving the first {self.field_size} lineups"
            )
            return

        print(f"Generating {diff} lineups.")
        player_data = self.extract_player_data()

        # Initialize problem list
        problems = self.initialize_problems_list(diff, player_data)

        # Handle stacks logic
        # stacks = self.handle_stacks_logic(diff)
        # print(problems)
        # print(self.player_dict)

        start_time = time.time()

        # Parallel processing for generating lineups
        with mp.Pool() as pool:
            output = pool.starmap(self.generate_lineups, problems)
            pool.close()
            pool.join()

        print("pool closed")

        # Update field lineups
        self.update_field_lineups(output, diff)

        end_time = time.time()
        print(f"lineups took {end_time - start_time} seconds")
        print(f"{diff} field lineups successfully generated")

    def extract_player_data(self):
        ids, ownership, salaries, projections, teams, opponents, matchups, positions = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for player_info in self.player_dict.values():
            if "Team" not in player_info:
                print(
                    f"{player_info['Name']} name mismatch between projections and player ids!"
                )
            ids.append(player_info["UniqueKey"])
            ownership.append(player_info["Ownership"])
            salaries.append(player_info["Salary"])
            projections.append(max(0, player_info.get("fieldFpts", 0)))
            teams.append(player_info["Team"])
            opponents.append(player_info["Opp"])
            matchups.append(player_info["Matchup"])
            pos_list = [
                1 if pos in player_info["rosterPosition"] else 0
                for pos in self.roster_construction
            ]
            positions.append(np.array(pos_list))
        return (
            ids,
            ownership,
            salaries,
            projections,
            teams,
            opponents,
            matchups,
            positions,
        )

    def initialize_problems_list(self, diff, player_data):
        (
            ids,
            ownership,
            salaries,
            projections,
            teams,
            opponents,
            matchups,
            positions,
        ) = player_data
        in_lineup = np.zeros(shape=len(ids))
        ownership, salaries, projections, pos_matrix = map(
            np.array, [ownership, salaries, projections, positions]
        )
        teams, opponents, ids = map(np.array, [teams, opponents, ids])
        new_player_dict = self.remap_player_dict(self.player_dict)
        num_players_in_roster = len(self.roster_construction)
        problems = []
        for i in range(diff):
            lu_tuple = (
                i,
                ids,
                in_lineup,
                pos_matrix,
                ownership,
                self.min_lineup_salary,
                self.salary,
                self.optimal_score,
                salaries,
                projections,
                self.max_pct_off_optimal,
                teams,
                opponents,
                self.overlap_limit,
                matchups,
                new_player_dict,
                num_players_in_roster,
                self.site
            )
            problems.append(lu_tuple)
        # print(self.player_dict.keys())
        return problems

    def handle_stacks_logic(self, diff):
        stacks = np.random.binomial(
            n=1, p=self.pct_field_using_stacks, size=diff
        ).astype(str)
        stack_len = np.random.choice(
            a=[1, 2],
            p=[1 - self.pct_field_double_stacks, self.pct_field_double_stacks],
            size=diff,
        )
        a = list(self.stacks_dict.keys())
        p = np.array(list(self.stacks_dict.values()))
        probs = p / sum(p)
        for i in range(len(stacks)):
            if stacks[i] == "1":
                choice = random.choices(a, weights=probs, k=1)
                stacks[i] = choice[0]
            else:
                stacks[i] = ""
        return stacks

    def update_field_lineups(self, output, diff):
        if len(self.field_lineups) == 0:
            new_keys = list(range(0, self.field_size))
        else:
            new_keys = list(
                range(
                    max(self.field_lineups.keys()) + 1,
                    max(self.field_lineups.keys()) + 1 + diff,
                )
            )

        nk = new_keys[0]
        for i, o in enumerate(output):
            lineup_list = sorted(next(iter(o.values()))["Lineup"])
            lineup_set = frozenset(lineup_list)  # Convert the list to a frozenset

            # Keeping track of lineup duplication counts
            if lineup_set in self.seen_lineups:
                self.seen_lineups[lineup_set] += 1

                # Increase the count in field_lineups using the index stored in seen_lineups_ix
                self.field_lineups[self.seen_lineups_ix[lineup_set]]["count"] += 1
            else:
                self.seen_lineups[lineup_set] = 1

                # Updating the field lineups dictionary
                if nk in self.field_lineups.keys():
                    print("bad lineups dict, please check dk_data files")
                else:
                    self.field_lineups[nk] = {
                        "Lineup": next(iter(o.values())),
                        "count": self.seen_lineups[lineup_set],
                    }

                    # Store the new nk in seen_lineups_ix for quick access in the future
                    self.seen_lineups_ix[lineup_set] = nk
                    nk += 1

    def calc_gamma(self, mean, sd):
        alpha = (mean / sd) ** 2
        beta = sd**2 / mean
        return alpha, beta

    def run_simulation_for_game(self, team1_id, team1, team2_id, team2, num_iterations):
        def get_corr_value(player1, player2):
            # First, check for specific player-to-player correlations
            if player2["Name"] in player1.get("Player Correlations", {}):
                return player1["Player Correlations"][player2["Name"]]
            
            # If no specific correlation is found, proceed with the general logic
            position_correlations = {
                "PG": -0.1324,
                "SG": -0.1324,
                "SF": -0.0812,
                "PF": -0.0812,
                "C": -0.1231
            }

            if player1["Team"] == player2["Team"] and player1["Position"][0] in ["PG", "SG", "SF", "PF", "C"]:
                primary_position = player1["Position"][0]
                return position_correlations[primary_position]

            if player1["Team"] != player2["Team"]:
                player_2_pos = "Opp " + str(player2["Position"][0])
            else:
                player_2_pos = player2["Position"][0]

            return player1["Correlations"].get(player_2_pos, 0)  # Default to 0 if no correlation is found

        def build_covariance_matrix(players):
            N = len(players)
            matrix = [[0 for _ in range(N)] for _ in range(N)]
            corr_matrix = [[0 for _ in range(N)] for _ in range(N)]

            for i in range(N):
                for j in range(N):
                    if i == j:
                        matrix[i][j] = (
                            players[i]["StdDev"] ** 2
                        )  # Variance on the diagonal
                        corr_matrix[i][j] = 1
                    else:
                        matrix[i][j] = (
                            get_corr_value(players[i], players[j])
                            * players[i]["StdDev"]
                            * players[j]["StdDev"]
                        )
                        corr_matrix[i][j] = get_corr_value(players[i], players[j])
            return matrix, corr_matrix

        def ensure_positive_semidefinite(matrix):
            eigs = np.linalg.eigvals(matrix)
            if np.any(eigs < 0):
                jitter = abs(min(eigs)) + 1e-6  # a small value
                matrix += np.eye(len(matrix)) * jitter

            # Given eigenvalues and eigenvectors from previous code
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)

            # Set negative eigenvalues to zero
            eigenvalues[eigenvalues < 0] = 0

            # Reconstruct the matrix
            matrix = eigenvectors.dot(np.diag(eigenvalues)).dot(eigenvectors.T)
            return matrix

        # Filter out players with projections less than or equal to 0
        team1 = [
            player
            for player in team1
            if player["Fpts"] > 0 and player["rosterPosition"] == "UTIL"
        ]
        team2 = [
            player
            for player in team2
            if player["Fpts"] > 0 and player["rosterPosition"] == "UTIL"
        ]

        game = team1 + team2
        covariance_matrix, corr_matrix = build_covariance_matrix(game)
        covariance_matrix = ensure_positive_semidefinite(covariance_matrix)

        try:
            samples = multivariate_normal.rvs(
                mean=[player["Fpts"] for player in game],
                cov=covariance_matrix,
                size=num_iterations,
            )
        except Exception as e:
            print(f"{team1_id}, {team2_id}, bad matrix: {str(e)}")
            return {}

        player_samples = [samples[:, i] for i in range(len(game))]

        temp_fpts_dict = {}
        for i, player in enumerate(game):
            temp_fpts_dict[player["UniqueKey"]] = player_samples[i]
            
        # fig, (ax1, ax2, ax3,ax4) = plt.subplots(4, figsize=(15, 25))
        # fig.tight_layout(pad=5.0)

        # for i, player in enumerate(game):
        #     sns.kdeplot(player_samples[i], ax=ax1, label=player['Name'])

        # ax1.legend(loc='upper right', fontsize=14)
        # ax1.set_xlabel('Fpts', fontsize=14)
        # ax1.set_ylabel('Density', fontsize=14)
        # ax1.set_title(f'Team {team1_id}{team2_id} Distributions', fontsize=14)
        # ax1.tick_params(axis='both', which='both', labelsize=14)

        # y_min, y_max = ax1.get_ylim()
        # ax1.set_ylim(y_min, y_max*1.1)

        # ax1.set_xlim(-5, 50)

        # # # Sorting players and correlating their data
        # player_names = [f"{player['Name']} ({player['Position']})" if player['Position'] is not None else f"{player['Name']} (P)" for player in game]

        # # # Ensuring the data is correctly structured as a 2D array
        # sorted_samples_array = np.array(player_samples)
        # if sorted_samples_array.shape[0] < sorted_samples_array.shape[1]:
        #     sorted_samples_array = sorted_samples_array.T

        # correlation_matrix = pd.DataFrame(np.corrcoef(sorted_samples_array.T), columns=player_names, index=player_names)

        # sns.heatmap(correlation_matrix, annot=True, ax=ax2, cmap='YlGnBu', cbar_kws={"shrink": .5})
        # ax2.set_title(f'Correlation Matrix for Game {team1_id}{team2_id}', fontsize=14)

        # original_corr_matrix = pd.DataFrame(corr_matrix, columns=player_names, index=player_names)
        # sns.heatmap(original_corr_matrix, annot=True, ax=ax3, cmap='YlGnBu', cbar_kws={"shrink": .5})
        # ax3.set_title(f'Original Correlation Matrix for Game {team1_id}{team2_id}', fontsize=14)

        # mean_values = [np.mean(samples) for samples in player_samples]
        # variance_values = [np.var(samples) for samples in player_samples]
        # min_values = [np.min(samples) for samples in player_samples]
        # max_values = [np.max(samples) for samples in player_samples]

        # # Create a DataFrame for the mean and variance values
        # mean_variance_df = pd.DataFrame({
        #     'Player': player_names,
        #     'Mean': mean_values,
        #     'Variance': variance_values,
        #     'Min' : min_values,
        #     'Max' :max_values
        # }).set_index('Player')

        # # Plot the mean and variance table
        # ax4.axis('tight')
        # ax4.axis('off')
        # ax4.table(cellText=mean_variance_df.values, colLabels=mean_variance_df.columns, rowLabels=mean_variance_df.index, cellLoc='center', loc='center')
        # ax4.set_title(f'Mean and Variance for Game {team1_id}{team2_id}', fontsize=14)

        # plt.savefig(f'output/Team_{team1_id}{team2_id}_Distributions_Correlation.png', bbox_inches='tight')
        # plt.close()

        return temp_fpts_dict

    @staticmethod
    @jit(nopython=True)
    def calculate_payouts(args):
        (
            ranks,
            payout_array,
            entry_fee,
            field_lineup_keys,
            use_contest_data,
            field_lineups_count,
        ) = args
        num_lineups = len(field_lineup_keys)
        combined_result_array = np.zeros(num_lineups)

        payout_cumsum = np.cumsum(payout_array)

        for r in range(ranks.shape[1]):
            ranks_in_sim = ranks[:, r]
            payout_index = 0
            for lineup_index in ranks_in_sim:
                lineup_count = field_lineups_count[lineup_index]
                prize_for_lineup = (
                    (
                        payout_cumsum[payout_index + lineup_count - 1]
                        - payout_cumsum[payout_index - 1]
                    )
                    / lineup_count
                    if payout_index != 0
                    else payout_cumsum[payout_index + lineup_count - 1] / lineup_count
                )
                combined_result_array[lineup_index] += prize_for_lineup
                payout_index += lineup_count
        return combined_result_array

    def run_tournament_simulation(self):
        print(f"Running {self.num_iterations} simulations")
        print(f"Number of unique field lineups: {len(self.field_lineups.keys())}")

        def generate_cpt_outcomes(UTIL_dict):
            cpt_dict = {}
            for player_id, UTIL_outcomes in UTIL_dict.items():
                # Fetch team information using the player_id
                # Assuming self.player_dict uses a structure like {(player_name, position, team): player_data}
                player_data_UTIL = [
                    data
                    for (name, pos, team), data in self.player_dict.items()
                    if data["UniqueKey"] == player_id and pos == "UTIL"
                ]
                if player_data_UTIL:
                    player_data_UTIL = player_data_UTIL[
                        0
                    ]  # Get the first match (there should only be one)
                    team = player_data_UTIL["Team"]

                    # Fetch the CPT data using the player_name and team fetched from the above step
                    player_data_cpt = self.player_dict.get(
                        (player_data_UTIL["Name"], "CPT", team)
                    )
                    if player_data_cpt:
                        cpt_outcomes = UTIL_outcomes * 1.5
                        cpt_dict[player_data_cpt["UniqueKey"]] = cpt_outcomes
            return cpt_dict

        def generate_fd_outcomes(UTIL_dict):
            mvp_dict = {}
            for player_id, UTIL_outcomes in UTIL_dict.items():
                # Fetch team information using the player_id
                # Assuming self.player_dict uses a structure like {(player_name, position, team): player_data}
                player_data_UTIL = [
                    data
                    for (name, pos, team), data in self.player_dict.items()
                    if data["UniqueKey"] == player_id and pos == "UTIL"
                ]
                if player_data_UTIL:
                    player_data_UTIL = player_data_UTIL[
                        0
                    ]  # Get the first match (there should only be one)
                    team = player_data_UTIL["Team"]

                    # Fetch the CPT data using the player_name and team fetched from the above step
                    player_data_mvp = self.player_dict.get(
                        (player_data_UTIL["Name"], "MVP", team)
                    )
                    if player_data_mvp:
                        mvp_outcomes = UTIL_outcomes * 2
                        mvp_dict[player_data_mvp["UniqueKey"]] = mvp_outcomes
                    player_data_star = self.player_dict.get(
                        (player_data_UTIL["Name"], "STAR", team)
                    )
                    if player_data_star:
                        star_outcomes = UTIL_outcomes * 1.5
                        mvp_dict[player_data_star["UniqueKey"]] = star_outcomes
                    player_data_pro = self.player_dict.get(
                        (player_data_UTIL["Name"], "PRO", team)
                    )
                    if player_data_pro:
                        pro_outcomes = UTIL_outcomes * 1.2
                        mvp_dict[player_data_pro["UniqueKey"]] = pro_outcomes
            return mvp_dict

        start_time = time.time()
        temp_fpts_dict = {}

        # Get the only matchup since it's a showdown
        matchup = list(self.matchups)[0]

        # Prepare the arguments for the simulation function
        game_simulation_params = (
            matchup[0],
            self.teams_dict[matchup[0]],
            matchup[1],
            self.teams_dict[matchup[1]],
            self.num_iterations,
        )

        # Run the simulation for the single game
        temp_fpts_dict.update(self.run_simulation_for_game(*game_simulation_params))
        if self.site == 'dk':
            cpt_outcomes_dict = generate_cpt_outcomes(temp_fpts_dict)
            temp_fpts_dict.update(cpt_outcomes_dict)
        elif self.site == 'fd':
            cpt_outcomes_dict = generate_fd_outcomes(temp_fpts_dict)
            temp_fpts_dict.update(cpt_outcomes_dict)            
        
        # generate arrays for every sim result for each player in the lineup and sum
        fpts_array = np.zeros(shape=(len(self.field_lineups), self.num_iterations))
        # converting payout structure into an np friendly format, could probably just do this in the load contest function
        # print(self.field_lineups)
        # print(temp_fpts_dict)
        # print(payout_array)
        # print(self.player_dict[('patrick mahomes', 'UTIL', 'KC')])
        field_lineups_count = np.array(
            [self.field_lineups[idx]["count"] for idx in self.field_lineups.keys()]
        )

        #print(self.player_dict)
        for index, values in self.field_lineups.items():
            try:
                fpts_sim = sum(
                    [temp_fpts_dict[player] for player in values["Lineup"]["Lineup"]]
                )
            except KeyError:
                for player in values["Lineup"]["Lineup"]:
                    if player not in temp_fpts_dict.keys():
                        print(player)
                        # for k,v in self.player_dict.items():
                        # if v['ID'] == player:
                        #        print(k,v)
                # print('cant find player in sim dict', values["Lineup"], temp_fpts_dict.keys())
            # store lineup fpts sum in 2d np array where index (row) corresponds to index of field_lineups and columns are the fpts from each sim
            fpts_array[index] = fpts_sim

        fpts_array = fpts_array.astype(np.float16)
        # ranks = np.argsort(fpts_array, axis=0)[::-1].astype(np.uint16)
        ranks = np.argsort(-fpts_array, axis=0).astype(np.uint32)

        # count wins, top 10s vectorized
        wins, win_counts = np.unique(ranks[0, :], return_counts=True)
        t10, t10_counts = np.unique(ranks[0:9], return_counts=True)
        payout_array = np.array(list(self.payout_structure.values()))
        # subtract entry fee
        payout_array = payout_array - self.entry_fee
        l_array = np.full(
            shape=self.field_size - len(payout_array), fill_value=-self.entry_fee
        )
        payout_array = np.concatenate((payout_array, l_array))
        field_lineups_keys_array = np.array(list(self.field_lineups.keys()))

        # Adjusted ROI calculation
        # print(field_lineups_count.shape, payout_array.shape, ranks.shape, fpts_array.shape)

        # Split the simulation indices into chunks
        field_lineups_keys_array = np.array(list(self.field_lineups.keys()))

        chunk_size = self.num_iterations // 16  # Adjust chunk size as needed
        simulation_chunks = [
            (
                ranks[:, i : min(i + chunk_size, self.num_iterations)].copy(),
                payout_array,
                self.entry_fee,
                field_lineups_keys_array,
                self.use_contest_data,
                field_lineups_count,
            )  # Adding field_lineups_count here
            for i in range(0, self.num_iterations, chunk_size)
        ]

        # Use the pool to process the chunks in parallel
        with mp.Pool() as pool:
            results = pool.map(self.calculate_payouts, simulation_chunks)

        combined_result_array = np.sum(results, axis=0)
        total_sum = 0
        index_to_key = list(self.field_lineups.keys())
        for idx, roi in enumerate(combined_result_array):
            lineup_key = index_to_key[idx]
            lineup_count = self.field_lineups[lineup_key][
                "count"
            ]  # Assuming "Count" holds the count of the lineups
            total_sum += roi * lineup_count
            self.field_lineups[lineup_key]["Lineup"]["ROI"] += roi

        for idx in self.field_lineups.keys():
            if idx in wins:
                self.field_lineups[idx]["Lineup"]["Wins"] += win_counts[
                    np.where(wins == idx)
                ][0]
            if idx in t10:
                self.field_lineups[idx]["Lineup"]["Top10"] += t10_counts[
                    np.where(t10 == idx)
                ][0]

        end_time = time.time()
        diff = end_time - start_time
        print(
            str(self.num_iterations)
            + " tournament simulations finished in "
            + str(diff)
            + " seconds. Outputting."
        )

    def output(self):
        unique = {}
        for index, data in self.field_lineups.items():
            # if index == 0:
            #    print(data)
            lineup = data["Lineup"]["Lineup"]
            lineup_data = data["Lineup"]
            lu_type = lineup_data["Type"]

            salary = 0
            fpts_p = 0
            fieldFpts_p = 0
            ceil_p = 0
            own_p = []
            own_s = []
            lu_names = []
            lu_teams = []
            cpt_tm = ""
            def_opps = []
            players_vs_def = 0

            player_dict_values = {
                v["UniqueKey"]: v for k, v in self.player_dict.items()
            }

            for player_id in lineup:
                player_data = player_dict_values.get(player_id, {})
                if player_data:
                    if "DST" in player_data["Position"]:
                        def_opps.append(player_data["Opp"])
                    if "CPT" in player_data["rosterPosition"]:
                        cpt_tm = player_data["Team"]
                    if "MVP" in player_data["rosterPosition"]:
                        cpt_tm = player_data["Team"]

                    salary += player_data.get("Salary", 0)
                    fpts_p += player_data.get("Fpts", 0)
                    fieldFpts_p += player_data.get("fieldFpts", 0)
                    ceil_p += player_data.get("Ceiling", 0)
                    own_p.append(player_data.get("Ownership", 0) / 100)
                    own_s.append(player_data.get("Ownership", 0))
                    if self.site == "fd" and "CPT" in player_data["rosterPosition"]:
                        player_id = player_data.get("ID", "")
                        if player_id.endswith("69696969"):
                            player_id = player_id.replace("69696969", "")
                        lu_names.append(f"{player_data.get('Name', '')} ({player_id})")
                    else:
                        lu_names.append(
                            f"{player_data.get('Name', '').replace('#','-')} ({player_data.get('ID', '').replace('#','-')})"
                        )
                    lu_teams.append(player_data["Team"])

            counter = collections.Counter(lu_teams)
            stacks = counter.most_common()

            primary_stack = secondary_stack = ""
            for s in stacks:
                if s[0] == cpt_tm:
                    primary_stack = f"{cpt_tm} {s[1]}"
                    stacks.remove(s)
                    break

            if stacks:
                secondary_stack = f"{stacks[0][0]} {stacks[0][1]}"

            own_p = np.prod(own_p)
            own_s = np.sum(own_s)
            win_p = round(lineup_data["Wins"] / self.num_iterations * 100, 2)
            top10_p = round(lineup_data["Top10"] / self.num_iterations * 100, 2)
            cash_p = round(lineup_data["Cashes"] / self.num_iterations * 100, 2)
            num_dupes = data["count"]
            if self.use_contest_data:
                roi_p = round(
                    lineup_data["ROI"] / self.entry_fee / self.num_iterations * 100, 2
                )
                roi_round = round(lineup_data["ROI"] / self.num_iterations, 2)

            if self.use_contest_data:
                lineup_str = f"{lu_type},{','.join(lu_names)},{salary},{fpts_p},{fieldFpts_p},{ceil_p},{primary_stack},{secondary_stack},{win_p}%,{top10_p}%,{cash_p}%,{own_p},{own_s},{roi_p}%,${roi_round},{num_dupes}"
            else:
                lineup_str = f"{lu_type},{','.join(lu_names)},{salary},{fpts_p},{fieldFpts_p},{ceil_p},{primary_stack},{secondary_stack},{win_p}%,{top10_p}%,{cash_p}%,{own_p},{own_s},{num_dupes}"
            unique[
                lineup_str
            ] = fpts_p  # Changed data["Fpts"] to fpts_p, which contains the accumulated Fpts

        return unique

    def player_output(self):
        # out_path = os.path.join(self.output_dir, f"{self.slate_id}_{self.sport}_{self.site}_player_output.csv")
        # First output file
        out_path = os.path.join(
            os.path.dirname(__file__),
            "../output/{}_sd_sim_player_exposure_{}_{}.csv".format(
                self.site, self.field_size, self.num_iterations
            ),
        )
        with open(out_path, "w") as f:
            f.write(
                "Player,Roster Position,Position,Team,Win%,Top10%,Sim. Own%,Proj. Own%,Avg. Return\n"
            )
            unique_players = {}

            for val in self.field_lineups.values():
                lineup_data = val["Lineup"]
                counts = val["count"]
                for player_id in lineup_data["Lineup"]:
                    if player_id not in unique_players:
                        unique_players[player_id] = {
                            "Wins": lineup_data["Wins"],
                            "Top10": lineup_data["Top10"],
                            "In": val["count"],
                            "ROI": lineup_data["ROI"],
                        }
                    else:
                        unique_players[player_id]["Wins"] += lineup_data["Wins"]
                        unique_players[player_id]["Top10"] += lineup_data["Top10"]
                        unique_players[player_id]["In"] += val["count"]
                        unique_players[player_id]["ROI"] += lineup_data["ROI"]

            for player_id, data in unique_players.items():
                field_p = round(data["In"] / self.field_size * 100, 2)
                win_p = round(data["Wins"] / self.num_iterations * 100, 2)
                top10_p = round(data["Top10"] / self.num_iterations / 10 * 100, 2)
                roi_p = round(data["ROI"] / data["In"] / self.num_iterations, 2)
                for k, v in self.player_dict.items():
                    if v["UniqueKey"] == player_id:
                        player_info = v
                        break
                proj_own = player_info.get("Ownership", "N/A")
                p_name = player_info.get("Name", "N/A").replace("#", "-")
                sd_position = player_info.get("rosterPosition", ["N/A"])
                position = player_info.get("Position", ["N/A"])[0]
                team = player_info.get("Team", "N/A")

                f.write(
                    f"{p_name},{sd_position},{position},{team},{win_p}%,{top10_p}%,{field_p}%,{proj_own}%,${roi_p}\n"
                )

    def save_results(self):
        unique = self.output()

        # First output file
        # include timetsamp in filename, formatted as readable
        now = datetime.datetime.now().strftime("%a_%I_%M_%S%p").lower()
        out_path = os.path.join(
            os.path.dirname(__file__),
            "../output/{}_sd_sim_lineups_{}_{}_{}.csv".format(
                self.site, self.field_size, self.num_iterations, now
            ),
        )
        if self.site == "dk":
            if self.use_contest_data:
                with open(out_path, "w") as f:
                    header = "Type,CPT,UTIL,UTIL,UTIL,UTIL,UTIL,Salary,Fpts Proj,Field Fpts Proj,Ceiling,Primary Stack,Secondary Stack,Win %,Top 10%,Cash %,Proj. Own. Product,Proj. Own. Sum,ROI%,ROI$,Num Dupes\n"
                    f.write(header)
                    for lineup_str, fpts in unique.items():
                        f.write(f"{lineup_str}\n")
            else:
                with open(out_path, "w") as f:
                    header = "Type,CPT,UTIL,UTIL,UTIL,UTIL,UTIL,Salary,Fpts Proj,Field Fpts Proj,Ceiling,Primary Stack,Secondary Stack,Win %,Top 10%,Cash %,Proj. Own. Product,Proj. Own. Sum,Num Dupes\n"
                    f.write(header)
                    for lineup_str, fpts in unique.items():
                        f.write(f"{lineup_str}\n")
        else:
            if self.use_contest_data:
                with open(out_path, "w") as f:
                    header = "Type,MVP,STAR,PRO,UTIL,UTIL,Salary,Fpts Proj,Field Fpts Proj,Ceiling,Primary Stack,Secondary Stack,Win %,Top 10%,Cash %,Proj. Own. Product,Proj. Own. Sum,ROI,ROI/Entry Fee,Num Dupes\n"
                    f.write(header)
                    for lineup_str, fpts in unique.items():
                        f.write(f"{lineup_str}\n")
            else:
                with open(out_path, "w") as f:
                    header = "Type,MVP,STAR,PRO,UTIL,UTIL,Salary,Fpts Proj,Field Fpts Proj,Ceiling,Primary Stack,Secondary Stack,Win %,Top 10%,Cash %,Proj. Own. Product,Proj. Own. Sum,Num Dupes\n"
                    f.write(header)
                    for lineup_str, fpts in unique.items():
                        f.write(f"{lineup_str}\n")
        self.player_output()