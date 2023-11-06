import json
import csv
import os
import datetime
import re
import numpy as np
import pulp as plp
import random
import itertools
import re
import math
import multiprocessing
import time
from collections import Counter, defaultdict
from numba import jit, prange
from scipy.stats import norm, kendalltau, multivariate_normal, gamma
import requests

@jit(nopython=True)
def salary_boost(salary, max_salary):
    return (salary / max_salary) ** 2


class NBA_Swaptimizer_Sims:
    site = None
    config = None
    problem = None
    output_dir = None
    num_uniques = None
    contest_id = None
    contest_name = None
    field_size = None
    num_lineups = None
    use_contest_data = True
    payout_structure = {}
    entry_fee = None
    team_list = []
    lineups = []
    player_lineups = {}
    player_keys = []
    contest_lineups = {}
    output_lineups = []
    field_lineups = {}
    player_dict = {}
    at_least = {}
    at_most = {}
    team_limits = {}
    matchups = set()
    matchup_list = []
    matchup_limits = {}
    matchup_at_least = {}
    ids_to_gametime = {}
    locked_matchups = {}
    time_remaining_dict = {}
    global_team_limit = None
    projection_minimum = None
    user_lineups = False
    contest_entries = {}
    max_salary = None
    randomness_amount = 0
    num_minutes_per_player = 48
    optimal_score = 0
    min_salary = None
    teams_dict = defaultdict(list) 

    def __init__(self, num_iterations, site=None, num_uniques=1):
        self.site = site
        self.num_iterations = num_iterations
        self.num_uniques = int(num_uniques)
        if self.site == 'dk':
            self.roster_construction = ['PG','SG','SF','PF','C','G','F','UTIL']
            self.max_salary = 50000
        elif self.site == 'fd':
            self.roster_construction = ['PG','PG','SG','SG','SF','SF','PF','C']
            self.max_salary = 60000
        self.load_config()
        self.load_rules()
        self.get_live_scores()
        projection_path = os.path.join(
            os.path.dirname(__file__),
            "../{}_data/{}".format(self.site, self.config["projection_path"]),
        )
        self.load_projections(projection_path)

        player_path = os.path.join(
            os.path.dirname(__file__),
            "../{}_data/{}".format(self.site, self.config["player_path"]),
        )
        self.load_player_ids(player_path)
        
        self.get_optimal()
        contest_path = os.path.join(
        os.path.dirname(__file__),
            "../{}_data/{}".format(self.site, self.config["contest_structure_path"]),
        )
        self.load_contest_data(contest_path)
        print("Contest payout structure loaded.")
        
        live_contest_path = os.path.join(
            os.path.dirname(__file__),
            "../{}_data/{}".format(self.site, self.config["live_contest_path"]),
        )
        self.extract_player_points(live_contest_path)
        self.load_live_contest(live_contest_path)
        print('live contest loaded')
        if "late_swap_path" in self.config.keys():
            late_swap_path = os.path.join(
                os.path.dirname(__file__),
                "../{}_data/{}".format(self.site, self.config["late_swap_path"]),
            )
            if self.user_lineups == True:
                self.load_player_lineups(late_swap_path)

    # Load config from file
    def load_config(self):
        with open(
            os.path.join(os.path.dirname(__file__), "../config.json"),
            encoding="utf-8-sig",
        ) as json_file:
            self.config = json.load(json_file)

    # make column lookups on datafiles case insensitive
    def lower_first(self, iterator):
        return itertools.chain([next(iterator).lower()], iterator)
    
    def get_optimal(self):
        # Setup our linear programming equation - https://en.wikipedia.org/wiki/Linear_programming
        # We will use PuLP as our solver - https://coin-or.github.io/pulp/

        # We want to create a variable for each roster slot.
        # There will be an index for each player and the variable will be binary (0 or 1) representing whether the player is included or excluded from the roster.
        problem = plp.LpProblem("NBA", plp.LpMaximize)

        lp_variables = {}
        for player, attributes in self.player_dict.items():
            player_id = attributes["ID"]
            for pos in attributes["Position"]:
                lp_variables[(player, pos, player_id)] = plp.LpVariable(
                    name=f"{player}_{pos}_{player_id}", cat=plp.LpBinary
                )

        # set the objective - maximize fpts & set randomness amount from config
        problem += (
            plp.lpSum(
                self.player_dict[player]["fieldFpts"]
                * lp_variables[(player, pos, attributes["ID"])]
                for player, attributes in self.player_dict.items()
                for pos in attributes["Position"]
            ),
            "Objective",
        )

        # Set the salary constraints
        max_salary = 50000 if self.site == "dk" else 60000
        min_salary = 49000 if self.site == "dk" else 59000

        if self.projection_minimum is not None:
            min_salary = self.min_salary

        # Maximum Salary Constraint
        problem += (
            plp.lpSum(
                self.player_dict[player]["Salary"]
                * lp_variables[(player, pos, attributes["ID"])]
                for player, attributes in self.player_dict.items()
                for pos in attributes["Position"]
            )
            <= max_salary,
            "Max Salary",
        )

        # Minimum Salary Constraint
        problem += (
            plp.lpSum(
                self.player_dict[player]["Salary"]
                * lp_variables[(player, pos, attributes["ID"])]
                for player, attributes in self.player_dict.items()
                for pos in attributes["Position"]
            )
            >= min_salary,
            "Min Salary",
        )

        # Must not play all 8 or 9 players from the same team (8 if dk, 9 if fd)
        for matchup in self.matchup_list:
            problem += (
                plp.lpSum(
                    lp_variables[(player, pos, attributes["ID"])]
                    for player, attributes in self.player_dict.items()
                    for pos in attributes["Position"]
                    if attributes["Matchup"] == matchup
                )
                <= 8
                if self.site == "dk"
                else 9,
                f"Must not play all players from same matchup {matchup}",
            )

        if self.global_team_limit is not None:
            for teamIdent in self.team_list:
                problem += (
                    plp.lpSum(
                        lp_variables[(player, pos, attributes["ID"])]
                        for player, attributes in self.player_dict.items()
                        for pos in attributes["Position"]
                        if attributes["Team"] == teamIdent
                    )
                    <= int(self.global_team_limit),
                    f"Global team limit - at most {self.global_team_limit} players from {teamIdent}",
                )

        if self.site == "dk":
            # Constraints for specific positions
            for pos in ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]:
                problem += (
                    plp.lpSum(
                        lp_variables[(player, pos, attributes["ID"])]
                        for player, attributes in self.player_dict.items()
                        if pos in attributes["Position"]
                    )
                    == 1,
                    f"Must have at 1 {pos}",
                )

            # Constraint to ensure each player is only selected once
            for player in self.player_dict:
                player_id = self.player_dict[player]["ID"]
                problem += (
                    plp.lpSum(
                        lp_variables[(player, pos, player_id)]
                        for pos in self.player_dict[player]["Position"]
                    )
                    <= 1,
                    f"Can only select {player} once",
                )

        else:
            # Constraints for specific positions
            for pos in ["PG", "SG", "SF", "PF", "C"]:
                if pos == "C":
                    problem += (
                        plp.lpSum(
                            lp_variables[(player, pos, attributes["ID"])]
                            for player, attributes in self.player_dict.items()
                            if pos in attributes["Position"]
                        )
                        == 1,
                        f"Must have 1 {pos}",
                    )
                else:
                    problem += (
                        plp.lpSum(
                            lp_variables[(player, pos, attributes["ID"])]
                            for player, attributes in self.player_dict.items()
                            if pos in attributes["Position"]
                        )
                        == 2,
                        f"Must have 2 {pos}",
                    )

            # Constraint to ensure each player is only selected once
            for player in self.player_dict:
                player_id = self.player_dict[player]["ID"]
                problem += (
                    plp.lpSum(
                        lp_variables[(player, pos, player_id)]
                        for pos in self.player_dict[player]["Position"]
                    )
                    <= 1,
                    f"Can only select {player} once",
                )

        try:
            problem.solve(plp.PULP_CBC_CMD(msg=0))
        except plp.PulpSolverError:
            print(
                "Infeasibility reached - only generated {} lineups out of {}. Continuing with export.".format(
                    len(self.output_lineups), self.lineups
                )
            )

        ## Check for infeasibility
        if plp.LpStatus[problem.status] != "Optimal":
            print(
                "Infeasibility reached - only generated {} lineups out of {}. Continuing with export.".format(
                    len(self.lineups), self.num_lineups
                )
            )
        # Get the lineup and add it to our list
        player_unqiue_keys = [
            player for player in lp_variables if lp_variables[player].varValue != 0
        ]
        print(player_unqiue_keys[0][0])
        players = []
        for p in player_unqiue_keys:
            players.append(p[0])

        fpts_proj = sum(self.player_dict[player]["fieldFpts"] for player in players)
        # sal_used = sum(self.player_dict[player]["Salary"] for player in players)

        var_values = [
            value.varValue for value in problem.variables() if value.varValue != 0
        ]
        player_unqiue_keys = [
            player for player in lp_variables if lp_variables[player].varValue != 0
        ]
        #problem.writeLP('problem.lp')
        self.optimal_score = float(fpts_proj)
        print(self.optimal_score, player_unqiue_keys, var_values)
        
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

    @staticmethod
    def extract_players(lineup_string, positions_order):
        players = {}
        words = lineup_string.split()
        pos = None  # to keep track of current position
        player_name = []

        for word in words:
            if word in positions_order:
                if pos:
                    players[pos] = " ".join(player_name) if player_name else "LOCKED"
                pos = word
                player_name = []
            else:
                player_name.append(word)
        
        # for the last player in the string
        if pos:
            players[pos] = " ".join(player_name) if player_name else "LOCKED"
            
        # In case there are positions that weren't found in the lineup_string
        for position in positions_order:
            if position not in players:
                players[position] = "LOCKED"
        
        return players
    
    def update_bayesian_projection(self, player_key):
        player = self.player_dict[player_key]
        total_game_minutes = self.num_minutes_per_player  # total minutes in a game per player
        actual_fpts = player['ActualFpts']
        minutes_played = total_game_minutes - player['Minutes Remaining']

        if player['Minutes Remaining'] > 0:
            # Prior beliefs
            prior_mean = player['Fpts']
            prior_variance = player['StdDev'] ** 2

            # Points per minute based on prior mean
            ppm = prior_mean / total_game_minutes

            # Remaining variance considers the proportion of the game left
            remaining_variance = prior_variance * (player['Minutes Remaining'] / total_game_minutes)

            # Actual performance so far
            actual_ppm = actual_fpts / minutes_played

            # Variance for the actual performance so far
            actual_variance = remaining_variance * (minutes_played / total_game_minutes)

            # The scaling factor adjusts the weight given to prior vs actual based on the time played
            scaling_factor = minutes_played / total_game_minutes

            # Weighted actual performance more as more of the game is played
            weighted_actual_ppm = actual_ppm * scaling_factor + ppm * (1 - scaling_factor)

            # Calculate the updated projection for the remaining minutes
            updated_remaining_projection = weighted_actual_ppm * player['Minutes Remaining']

            # Combine actual points with updated remaining projection for total updated projection
            updated_projection = actual_fpts + updated_remaining_projection

            # The updated variance for the remaining projection should also be adjusted
            posterior_variance = 1 / ((1 / (prior_variance * (1 - scaling_factor))) + (1 / (actual_variance * scaling_factor)))

        else:
            # If the game is over, the final score is just the actual points
            updated_projection = actual_fpts
            posterior_variance = 0

        # Update the player's projections
        player['BayesianProjectedFpts'] = updated_projection
        player['BayesianProjectedVar'] = posterior_variance

        return player


    
    def get_live_scores(self):
        game_date = datetime.datetime.now().date()

        team_id_to_abbreviation = {
            1610612737: 'ATL',
            1610612738: 'BOS',
            1610612739: 'CLE',
            1610612740: 'NOP',
            1610612741: 'CHI',
            1610612742: 'DAL',
            1610612743: 'DEN',
            1610612744: 'GSW',
            1610612745: 'HOU',
            1610612746: 'LAC',
            1610612747: 'LAL',
            1610612748: 'MIA',
            1610612749: 'MIL',
            1610612750: 'MIN',
            1610612751: 'BKN',
            1610612752: 'NYK',
            1610612753: 'ORL',
            1610612754: 'IND',
            1610612755: 'PHI',
            1610612756: 'PHX',
            1610612757: 'POR',
            1610612758: 'SAC',
            1610612759: 'SAS',
            1610612760: 'OKC',
            1610612761: 'TOR',
            1610612762: 'UTA',
            1610612763: 'MEM',
            1610612764: 'WAS',
            1610612765: 'DET',
            1610612766: 'CHA'
        }

        # Format the date into the string format the NBA API expects ('YYYY-MM-DD')
        formatted_date = game_date.strftime('%Y-%m-%d')

        formatted_date = '2023-11-03'
        headers= {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
                'Referer': 'https://www.nba.com/'}

        # Insert the formatted date into the URL
        scoreboard_url = f'https://stats.nba.com/stats/scoreboardv2?DayOffset=0&GameDate={formatted_date}&LeagueID=00'


        scoreboard_json = requests.get(scoreboard_url, headers=headers).json()

        # Assuming `data` is the JSON response parsed into a Python dictionary
        games_info = scoreboard_json['resultSets'][0]['rowSet']

        # NBA regulation game length in minutes
        regulation_game_length = 48
        overtime_period_length = 5  # NBA overtime period length in minutes

        for game in games_info:
            game_id = game[2]
            home_team_id = game[6]
            visitor_team_id = game[7]
            live_period = game[9]
            live_pc_time = game[10].strip()

            # Calculate the total time remaining
            if live_period <= 4:  # Regulation time
                total_minutes_remaining = regulation_game_length - (live_period - 1) * 12
                if live_pc_time:
                    minutes, seconds = map(int, live_pc_time.split(":"))
                    total_minutes_remaining -= minutes
                    if seconds > 0:
                        total_minutes_remaining -= 1  # Subtract a minute if there are seconds remaining
            else:  # Overtime
                completed_overtimes = live_period - 4
                total_minutes_remaining = (completed_overtimes * overtime_period_length)
                if live_pc_time:
                    minutes, seconds = map(int, live_pc_time.split(":"))
                    total_minutes_remaining -= minutes
                    if seconds > 0:
                        total_minutes_remaining -= 1  # Subtract a minute if there are seconds remaining

            # For finished games, set the remaining time to 0
            if 'Final' in game[4]:
                total_minutes_remaining = 0

            # Mapping team IDs to abbreviations and adding to the dictionary
            home_team_abbreviation = team_id_to_abbreviation.get(home_team_id, 'Unknown')
            visitor_team_abbreviation = team_id_to_abbreviation.get(visitor_team_id, 'Unknown')
            self.matchups.add((home_team_abbreviation, visitor_team_abbreviation))
            self.time_remaining_dict[home_team_abbreviation] = total_minutes_remaining
            self.time_remaining_dict[visitor_team_abbreviation] = total_minutes_remaining
            self.time_remaining_dict =  {'IND': 0,
                                        'CLE': 0,
                                        'MIL': 6,
                                        'NYK': 6,
                                        'MIA': 7,
                                        'WAS': 7,
                                        'CHI': 6,
                                        'BKN': 6,
                                        'OKC': 9,
                                        'GSW': 9,
                                        'DEN': 48,
                                        'DAL': 48,
                                        'POR': 48,
                                        'MEM': 48}
            
    def extract_player_points(self, path):
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['Player'] == '':
                    break
                else:
                    name = row['Player']
                    pos = row['Roster Position']
                    for k,v in self.player_dict.items():
                        if v['Name'] == name and pos in v['Position']:
                            print('found player' + name)
                            self.player_dict[k]['ActualFpts'] = float(row['FPTS'])
                            self.update_bayesian_projection(k)
                            print(self.player_dict[k])
                            
        

    def get_username(self, text):
        # The regex will match any text up until it possibly encounters a space followed by (digit/digit)
        match = re.search(r"^(.+?)(?:\s*\(\d+/\d+\))?$", text)
        if match:
            return match.group(1).strip()  # Strip to remove any trailing whitespaces
        return None  # or an appropriate value indicating no match


    def load_live_contest(self, path):
        match = re.search(r'contest-standings-(\d+).csv', path)
        positions_order = ["C", "F", "G", "PF", "PG", "SF", "SG", "UTIL"]
        total_minutes_for_full_lineup = len(self.roster_construction)*self.num_minutes_per_player

        if match:
            self.contest_id = match.group(1)
        else:
            print('Unable to find contest id for loading live lineups for contest simulation')
        
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            for row in reader:
                name = self.get_username(row['EntryName'])
                lineup_updated_proj = 0
                lineup_updated_var = 0
                if name in self.contest_entries.keys():
                    self.contest_entries[name]['Entries'] += 1
                else:
                    self.contest_entries[name] = {'Entries':1, 'ROI': 0, 'Top1':0, 'Cashes':0, 'Wins':0}
                lineup_dict = {
                    "contest_id": self.contest_id,
                    "EntryId" : row['EntryId'],
                    'User' : name,
                    "Type" : 'opp'
                }
                lineup_proj_fpts = 0
                lineup_proj_stdv = 0
                lineup_salary = 0
                lineup_proj_fieldfpts = 0

                if not row["Lineup"].strip():
                    for pos in positions_order:
                        lineup_dict[f"{pos}_is_locked"] = True
                        lineup_dict[pos] = ''
                    lineup_dict["Points"] = 0
                    lineup_dict["TimeRemaining"] = 0
                    lineup_dict["ProjectedFpts"] = lineup_proj_fpts
                    lineup_dict["ProjectedStdDev"] = lineup_proj_stdv
                    lineup_dict['OriginalLineup'] =  {'C': '', 'F': '', 'G': '', 'PF': '', 'PG': '', 'SF': '', 'SG': '', 'UTIL': ''},
                    lineup_dict['LockedSalary'] = 50000 if self.site == 'dk' else 60000
                    lineup_dict['LockedPlayers'] = len(self.roster_construction)
                    lineup_dict['UnlockedPlayers'] = 0
                    total_minutes_for_full_lineup = len(self.roster_construction) * self.num_minutes_per_player
                    minutes_for_locked_players = total_minutes_for_full_lineup
                    minutes_used_by_actual_players = total_minutes_for_full_lineup
                    lineup_dict['Salary'] = 0
                    lineup_dict['SalaryRemaining'] = 0
                    lineup_dict["LockedPlayerMinutes"] = 0
                    lineup_dict['ProjectedFieldFpts'] = 0
                    lineup_dict['TotalMinutesForLineup'] = 0
                    lineup_dict['EmptyLu'] = True
                    lineup_dict['UserLu'] = False
                    self.contest_lineups[str(row["EntryId"])] = lineup_dict
                    
                else:
                    lineup_minutes_remaining = 0
                    extracted_players = self.extract_players(row["Lineup"], positions_order)
                    for pos, player_name in extracted_players.items():
                        locked_key = f"{pos}_is_locked"
                        if player_name == "LOCKED":
                            lineup_dict[locked_key] = False
                            lineup_dict[pos] = player_name
                            lineup_minutes_remaining += self.num_minutes_per_player
                            continue
                        else:
                            lineup_dict[locked_key] = True
                            
                            for k, v in self.player_dict.items():
                                player_found = False

                                transformed_player_name = player_name

                                # Print the comparison for debugging
                                #print(f"Comparing {v['Name']} with {transformed_player_name} and {pos} with {v['Position']}")

                                if v['Name'] == transformed_player_name and pos in v['Position']:
                                    lineup_proj_fpts += v['Fpts']
                                    lineup_salary += v['Salary']
                                    lineup_dict[pos] = v['ID']
                                    lineup_updated_proj += v['BayesianProjectedFpts']
                                    lineup_updated_var += v['BayesianProjectedVar']
                                    lineup_minutes_remaining += v['Minutes Remaining']
                                    player_found = True
                                    break

                            if not player_found:
                                print(f'player not found: {transformed_player_name} {pos}, {extracted_players}')
                                
                    lineup_dict["Points"] = float(row["Points"])
                    lineup_dict["TimeRemaining"] = lineup_minutes_remaining
                    lineup_dict["ProjectedFpts"] = lineup_proj_fpts
                    lineup_dict["ProjectedStdDev"] = lineup_proj_stdv
                    lineup_dict['OriginalLineup'] = extracted_players
                    if lineup_proj_stdv <= 0:
                        lineup_proj_stdv = 1
                    actual_minutes_used = total_minutes_for_full_lineup - lineup_minutes_remaining
                    efficiency_factor = 0.1  # A constant that scales the uncertainty based on minutes remaining
                    minutes_proportion_remaining = lineup_minutes_remaining / total_minutes_for_full_lineup
                    lineup_dict["BayesianProjectedFpts"] = lineup_updated_proj
                    lineup_dict["BayesianProjectedVar"] = lineup_updated_var
                    minutes_for_locked_players = row["Lineup"].count("LOCKED") * self.num_minutes_per_player
                    lineup_dict['Salary'] = lineup_salary
                    lineup_dict['TotalMinutesForLineup'] = total_minutes_for_full_lineup
                    lineup_dict['LockedSalary'] = lineup_salary
                    lineup_dict['SalaryRemaining'] = self.max_salary - lineup_salary
                    lineup_dict['ProjectedFieldFpts'] = lineup_proj_fieldfpts
                    lineup_dict["LockedPlayerMinutes"] = minutes_for_locked_players
                    lineup_dict['UsedPlayerMinutes'] = actual_minutes_used
                    lineup_dict['UnlockedPlayers'] = row['Lineup'].count("LOCKED")
                    lineup_dict['LockedPlayers'] = len(self.roster_construction) - lineup_dict['UnlockedPlayers']
                    lineup_dict['EmptyLu'] = False
                    lineup_dict['UserLu'] = False
                    self.contest_lineups[str(row["EntryId"])] = lineup_dict
        random_keys = random.sample(list(self.contest_lineups.keys()), 5)
        self.num_lineups = len(self.contest_lineups)
        
    # Load player IDs for exporting
    def load_player_ids(self, path):
        current_day = datetime.datetime.now().date()  # Get the current date
    
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            for row in reader:
                name_key = "Name" if self.site == "dk" else "Nickname"
                player_name = row[name_key]
                team = row["TeamAbbrev"] if self.site == "dk" else row["Team"]
                position = row["Position"]
                
                match = re.search(pattern="(\w{2,4}@\w{2,4})", string=row["Game Info"])
                team_opp = ''
                if match:
                    opp = match.groups()[0].split("@")
                    for m in opp:
                        if m != team:
                            team_opp = m
                    opp = tuple(opp)

                if (player_name, position, team) in self.player_dict:
                    if self.site == "dk":
                        self.player_dict[(player_name, position, team)]["ID"] = str(row["ID"])
                        self.player_dict[(player_name, position, team)]["Matchup"] = row["Game Info"].split(" ")[0]
                        self.player_dict[(player_name, position, team)]["UniqueKey"] = str(row["ID"])
                        self.player_dict[(player_name, position, team)]["Opp"] = team_opp
                        
                        # Check if game is In Progress
                        if "In Progress" in row["Game Info"]:
                            # If game is in progress, set the game time to 12:01 am of the current day
                            self.player_dict[(player_name, position, team)]["GameTime"] = datetime.datetime.combine(current_day, datetime.time(0, 1))
                            self.player_dict[(player_name, position, team)]["GameLocked"] = True
                        else:
                            self.player_dict[(player_name, position, team)]["GameTime"] = datetime.datetime.strptime(
                                " ".join(row["Game Info"].split()[1:])[:-3], 
                                "%m/%d/%Y %I:%M%p"
                            )
                            self.player_dict[(player_name, position, team)]["GameLocked"] = False
                        
                        self.ids_to_gametime[int(row["ID"])] = self.player_dict[(player_name, position, team)]["GameTime"]
                    else:
                        self.player_dict[(player_name, position, team)]["ID"] = str(row["ID"]).replace("-", "#")
                        self.player_dict[(player_name, position, team)]["UniqueKey"] = str(row["ID"]).replace("-", "#")
                        self.player_dict[(player_name, position, team)]["Matchup"] = row["Game"]
                        
                        if row["Game"] not in self.matchup_list:
                            self.matchup_list.append(row["Game"])

    def load_rules(self):
        self.at_most = self.config["at_most"]
        self.at_least = self.config["at_least"]
        self.team_limits = self.config["team_limits"]
        self.global_team_limit = int(self.config["global_team_limit"])
        self.projection_minimum = int(self.config["projection_minimum"])
        self.randomness_amount = float(self.config["randomness"])
        self.matchup_limits = self.config["matchup_limits"]
        self.matchup_at_least = self.config["matchup_at_least"]
        self.min_salary = int(self.config["min_lineup_salary"])
        self.default_var = float(self.config["default_var"])
        self.max_pct_off_optimal = float(self.config['max_pct_off_optimal'])

    def load_projections(self, path):
        # Read projections into a dictionary
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(self.lower_first(file))
            for row in reader:
                player_name = row["name"]
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
                if "fieldfpts" in row:
                    if row["fieldfpts"] == "":
                        fieldFpts = fpts
                    else:
                        fieldFpts = float(row["fieldfpts"])
                else:
                    fieldFpts = fpts
                position = [pos for pos in row["position"].split("/")]
                position.sort()
                if any(pos in ["PG", "SG"] for pos in position):
                    position.append("G")
                if any(pos in ["SF", "PF"] for pos in position):
                    position.append("F")
                position.append("UTIL")
                pos = position[0]
                if "stddev" in row:
                    if row["stddev"] == "" or float(row["stddev"]) == 0:
                        stddev = fpts * self.default_var
                    else:
                        stddev = float(row["stddev"])
                else:
                    stddev = fpts * self.default_var
                if stddev == 0:
                    stddev = 0.001
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
                    mins = row["minutes"]
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
                own = float(row["own%"].replace("%", ""))
                mins_rem = self.time_remaining_dict[team]
                if own == 0:
                    own = 0.1
                pos_str = str(row['position'])
                player_data = {
                    "Fpts": fpts,
                    "fieldFpts": fieldFpts,
                    "Position": position,
                    "Name": player_name,
                    "DK Name": row["name"],
                    "Team": team,
                    "Opp": "",
                    "ID": "",
                    "UniqueKey": "",
                    "Salary": int(row["salary"].replace(",", "")),
                    "StdDev": stddev,
                    "Ceiling": ceil,
                    "Ownership": own,
                    "Correlations": corr,
                    "Player Correlations": {},
                    "In Lineup": False,
                    "Minutes": mins,
                    "Minutes Remaining": mins_rem,
                    "BayesianProjectedFpts": fpts,
                    "BayesianProjectedVar": stddev**2,
                    "ActualFpts" : 0
                }

                # Check if player is in player_dict and get Opp, ID, Opp Pitcher ID and Opp Pitcher Name
                if (player_name, pos_str, team) in self.player_dict:
                    player_data["Opp"] = self.player_dict[
                        (player_name, pos_str, team)
                    ].get("Opp", "")
                    player_data["ID"] = self.player_dict[
                        (player_name, pos_str, team)
                    ].get("ID", "")

                self.player_dict[(player_name, pos_str, team)] = player_data
                self.teams_dict[team].append(
                    player_data
                )  # Add player data to their respective team

    # Load user lineups for late swap
    def load_player_lineups(self, path):
        # Read projections into a dictionary
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(self.lower_first(file))
            current_time = datetime.datetime.now()  # get the current time
            
            # current_time = datetime.datetime(2023, 10, 24, 20, 0) # testing time, such that LAL/DEN is locked
            print(f"Current time (UTC): {current_time}")
            for row in reader:
                if row["entry id"] != "" and self.site == "dk":
                    PG_id = re.search(r"\((\d+)\)", row["pg"]).group(1)
                    SG_id = re.search(r"\((\d+)\)", row["sg"]).group(1)
                    SF_id = re.search(r"\((\d+)\)", row["sf"]).group(1)
                    PF_id = re.search(r"\((\d+)\)", row["pf"]).group(1)
                    C_id = re.search(r"\((\d+)\)", row["c"]).group(1)
                    G_id = re.search(r"\((\d+)\)", row["g"]).group(1)
                    F_id = re.search(r"\((\d+)\)", row["f"]).group(1)
                    UTIL_id = re.search(r"\((\d+)\)", row["util"]).group(1)
                    if str(row['entry id']) in self.contest_lineups.keys():
                        lu = self.contest_lineups[str(row['entry id'])]
                        self.player_keys.append(str(row['entry id']))
                        lu['UserLu'] = True
                        lu['Type'] = 'user'
                        #print(row['entry id']+ 'found')
                        #print(lu)
                        for pos in ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]:
                            if lu[pos + "_is_locked"] == False:
                                if pos == "PG":
                                    lu[pos] = PG_id
                                elif pos == "SG":
                                    lu[pos] = SG_id
                                elif pos == "SF":
                                    lu[pos] = SF_id
                                elif pos == "PF":
                                    lu[pos] = PF_id
                                elif pos == "C":
                                    lu[pos] = C_id 
                                elif pos == "G":
                                    lu[pos] = G_id
                                elif pos == "F":
                                    lu[pos] = F_id
                                elif pos == "UTIL":
                                    #print(lu)
                                    #print(lu[pos], UTIL_id)
                                    lu[pos] = UTIL_id
                                    #print(lu[pos], self.contest_lineups[str(row['entry id'])][pos])
                    else:
                        print(f'Lineup {row["entry id"]} not found in contest file.')
        #print(f"Successfully loaded {len(self.player_keys)} lineups for late swap.")
        #print(self.player_keys)
        #print(self.contest_lineups['3983050569'])

    def swaptimize(self):
        # Setup our linear programming equation - https://en.wikipedia.org/wiki/Linear_programming
        # We will use PuLP as our solver - https://coin-or.github.io/pulp/

        # We want to create a variable for each roster slot.
        # There will be an index for each player and the variable will be binary (0 or 1) representing whether the player is included or excluded from the roster.
        for pk in self.player_keys:
            lineup_obj = self.contest_lineups[pk]
            print(
                f"Swaptimizing lineup {pk} in contest {lineup_obj['contest_id']}"
            )
            problem = plp.LpProblem("NBA", plp.LpMaximize)

            lp_variables = {}
            for player, attributes in self.player_dict.items():
                player_id = attributes["ID"]
                for pos in attributes["Position"]:
                    lp_variables[(player, pos, player_id)] = plp.LpVariable(
                        name=f"{player}_{pos}_{player_id}", cat=plp.LpBinary
                    )

            # set the objective - maximize fpts & set randomness amount from config
            if self.randomness_amount != 0:
                problem += (
                    plp.lpSum(
                        np.random.normal(
                            self.player_dict[player]["Fpts"],
                            (
                                self.player_dict[player]["StdDev"]
                                * self.randomness_amount
                                / 100
                            ),
                        )
                        * lp_variables[(player, pos, attributes["ID"])]
                        for player, attributes in self.player_dict.items()
                        for pos in attributes["Position"]
                    ),
                    "Objective",
                )
            else:
                problem += (
                    plp.lpSum(
                        self.player_dict[player]["Fpts"]
                        * lp_variables[(player, pos, attributes["ID"])]
                        for player, attributes in self.player_dict.items()
                        for pos in attributes["Position"]
                    ),
                    "Objective",
                )

            # Set the salary constraints
            max_salary = 50000 if self.site == "dk" else 60000
            min_salary = 49000 if self.site == "dk" else 59000

            if self.projection_minimum is not None:
                min_salary = self.min_salary

            # Maximum Salary Constraint
            problem += (
                plp.lpSum(
                    self.player_dict[player]["Salary"]
                    * lp_variables[(player, pos, attributes["ID"])]
                    for player, attributes in self.player_dict.items()
                    for pos in attributes["Position"]
                )
                <= max_salary,
                "Max Salary",
            )

            # Minimum Salary Constraint
            problem += (
                plp.lpSum(
                    self.player_dict[player]["Salary"]
                    * lp_variables[(player, pos, attributes["ID"])]
                    for player, attributes in self.player_dict.items()
                    for pos in attributes["Position"]
                )
                >= min_salary,
                "Min Salary",
            )

            # Must not play all 8 or 9 players from the same team (8 if dk, 9 if fd)
            for matchup in self.matchup_list:
                problem += (
                    plp.lpSum(
                        lp_variables[(player, pos, attributes["ID"])]
                        for player, attributes in self.player_dict.items()
                        for pos in attributes["Position"]
                        if attributes["Matchup"] == matchup
                    )
                    <= 8
                    if self.site == "dk"
                    else 9,
                    f"Must not play all players from same matchup {matchup}",
                )

            # Address limit rules if any
            for limit, groups in self.at_least.items():
                for group in groups:
                    problem += (
                        plp.lpSum(
                            lp_variables[(player, pos, attributes["ID"])]
                            for player, attributes in self.player_dict.items()
                            for pos in attributes["Position"]
                            if attributes["Name"] in group
                        )
                        >= int(limit),
                        f"At least {limit} players {group}",
                    )

            for limit, groups in self.at_most.items():
                for group in groups:
                    problem += (
                        plp.lpSum(
                            lp_variables[(player, pos, attributes["ID"])]
                            for player, attributes in self.player_dict.items()
                            for pos in attributes["Position"]
                            if attributes["Name"] in group
                        )
                        <= int(limit),
                        f"At most {limit} players {group}",
                    )

            for matchup, limit in self.matchup_limits.items():
                problem += (
                    plp.lpSum(
                        lp_variables[(player, pos, attributes["ID"])]
                        for player, attributes in self.player_dict.items()
                        for pos in attributes["Position"]
                        if attributes["Matchup"] == matchup
                    )
                    <= int(limit),
                    "At most {} players from {}".format(limit, matchup),
                )

            for matchup, limit in self.matchup_at_least.items():
                problem += (
                    plp.lpSum(
                        lp_variables[(player, pos, attributes["ID"])]
                        for player, attributes in self.player_dict.items()
                        for pos in attributes["Position"]
                        if attributes["Matchup"] == matchup
                    )
                    >= int(limit),
                    "At least {} players from {}".format(limit, matchup),
                )

            # Address team limits
            for teamIdent, limit in self.team_limits.items():
                problem += plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if team == teamIdent
                ) <= int(limit), "At most {} players from {}".format(limit, teamIdent)

            if self.global_team_limit is not None:
                for teamIdent in self.team_list:
                    problem += (
                        plp.lpSum(
                            lp_variables[(player, pos, attributes["ID"])]
                            for player, attributes in self.player_dict.items()
                            for pos in attributes["Position"]
                            if attributes["Team"] == teamIdent
                        )
                        <= int(self.global_team_limit),
                        f"Global team limit - at most {self.global_team_limit} players from {teamIdent}",
                    )

            # Force players to be used if they are locked
            POSITIONS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
            FORCE_PLAYERS = []
            for position in POSITIONS:
                if lineup_obj[position + "_is_locked"]:
                    player_id = lineup_obj[position]
                    for p_tuple, attributes in self.player_dict.items():
                        if str(attributes["ID"]) == str(player_id):
                            FORCE_PLAYERS.append((p_tuple, position, attributes["ID"]))

            for forced_player in FORCE_PLAYERS:
                problem += (
                    lp_variables[forced_player] == 1,
                    f"Force player {forced_player}",
                )

            if self.site == "dk":
                # Constraints for specific positions
                for pos in ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]:
                    problem += (
                        plp.lpSum(
                            lp_variables[(player, pos, attributes["ID"])]
                            for player, attributes in self.player_dict.items()
                            if pos in attributes["Position"]
                        )
                        == 1,
                        f"Must have at 1 {pos}",
                    )

                # Constraint to ensure each player is only selected once
                for player in self.player_dict:
                    player_id = self.player_dict[player]["ID"]
                    problem += (
                        plp.lpSum(
                            lp_variables[(player, pos, player_id)]
                            for pos in self.player_dict[player]["Position"]
                        )
                        <= 1,
                        f"Can only select {player} once",
                    )

            else:
                # Constraints for specific positions
                for pos in ["PG", "SG", "SF", "PF", "C"]:
                    if pos == "C":
                        problem += (
                            plp.lpSum(
                                lp_variables[(player, pos, attributes["ID"])]
                                for player, attributes in self.player_dict.items()
                                if pos in attributes["Position"]
                            )
                            == 1,
                            f"Must have 1 {pos}",
                        )
                    else:
                        problem += (
                            plp.lpSum(
                                lp_variables[(player, pos, attributes["ID"])]
                                for player, attributes in self.player_dict.items()
                                if pos in attributes["Position"]
                            )
                            == 2,
                            f"Must have 2 {pos}",
                        )

                # Constraint to ensure each player is only selected once
                for player in self.player_dict:
                    player_id = self.player_dict[player]["ID"]
                    problem += (
                        plp.lpSum(
                            lp_variables[(player, pos, player_id)]
                            for pos in self.player_dict[player]["Position"]
                        )
                        <= 1,
                        f"Can only select {player} once",
                    )

            # Don't dupe a lineup we already used
            i = 0
            for lineup, _ in self.output_lineups:
                player_ids = [tpl[2] for tpl in lineup]
                player_keys_to_exlude = []
                for key, attr in self.player_dict.items():
                    if attr["ID"] in player_ids:
                        for pos in attr["Position"]:
                            player_keys_to_exlude.append((key, pos, attr["ID"]))
                problem += (
                    plp.lpSum(lp_variables[x] for x in player_keys_to_exlude)
                    <= len(selected_vars) - self.num_uniques,
                    f"Lineup {i}",
                )
                i += 1

            try:
                problem.solve(plp.PULP_CBC_CMD(msg=0))
            except plp.PulpSolverError:
                print(
                    "Infeasibility reached - only generated {} lineups out of {}. Continuing with export.".format(
                        len(self.output_lineups), self.lineups
                    )
                )
                break

            ## Check for infeasibility
            if plp.LpStatus[problem.status] != "Optimal":
                print(
                    "Infeasibility reached - only generated {} lineups out of {}. Continuing with export.".format(
                        len(self.lineups), self.num_lineups
                    )
                )
                break

            # Get the lineup and add it to our list
            selected_vars = [
                player for player in lp_variables if lp_variables[player].varValue != 0
            ]
            for player in selected_vars:
                lineup_obj[player[1]] = player[2]
                
    def convert_player_dict_to_pid_keys(self):
        self.player_dict = {v['ID']: v for v in self.player_dict.values()}

    def compute_best_guesses_parallel(self):
        self.convert_player_dict_to_pid_keys()
        self.first_idx = list(self.contest_lineups.keys())[0]
        print('lineup after loading:')
        print(self.contest_lineups[self.first_idx])
        start = time.time()
        min_fpts = self.optimal_score - (self.max_pct_off_optimal * self.optimal_score)
        ids = []
        ownership = []
        salaries = []
        projections = []
        positions = []
        teams = []
        opponents = []
        matchups = []
        # put def first to make it easier to avoid overlap
        for k in self.player_dict.keys():
            if self.player_dict[k]['GameLocked'] == False:
                if "Team" not in self.player_dict[k].keys():
                    print(
                        self.player_dict[k]["Name"],
                        " name mismatch between projections and player ids!",
                    )
                ids.append(self.player_dict[k]["UniqueKey"])
                ownership.append(self.player_dict[k]["Ownership"])
                salaries.append(self.player_dict[k]["Salary"])
                if self.player_dict[k]["fieldFpts"] >= self.projection_minimum:
                    projections.append(self.player_dict[k]["fieldFpts"])
                else:
                    projections.append(0)
                teams.append(self.player_dict[k]["Team"])
                matchups.append(self.player_dict[k]["Matchup"])
                pos_list = []
                for pos in self.roster_construction:
                    if pos in self.player_dict[k]["Position"]:
                        pos_list.append(1)
                    else:
                        pos_list.append(0)
                positions.append(np.array(pos_list))
        in_lineup = np.zeros(shape=len(ids))
        ownership = np.array(ownership)
        salaries = np.array(salaries)
        projections = np.array(projections)
        pos_matrix = np.array(positions)
        ids = np.array(ids)
        optimal_score = self.optimal_score
        salary_floor = self.min_salary
        salary_ceiling = self.max_salary
        max_pct_off_optimal = self.max_pct_off_optimal
        teams = np.array(teams)
        opponents = np.array(opponents)
        problems = []
        num_players_in_roster = len(self.roster_construction)
        pos_index_dict = {pos: i for i, pos in enumerate(self.roster_construction)}
        # creating tuples of the above np arrays plus which lineup number we are going to create
        for key, lineup in self.contest_lineups.items():
            lu_tuple = (
                key,
                lineup,
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
                matchups,
                num_players_in_roster,
                self.site,
                self.roster_construction,
                pos_index_dict,
                np.min(salaries)
            )
            problems.append(lu_tuple)
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            results = pool.starmap(self.generate_lineups,problems)

        # New function to get names from IDs in a lineup
        def get_names_from_ids(lineup):
            #print(lineup)
            new_pos_dict = {}
            for pos in self.roster_construction:
                new_pos_dict[pos] = self.player_dict[lineup[pos]]['Name']
            return new_pos_dict

        bad_lu_count = 0
        bad_lus = []
        good_lu_count = 0
        minutes_count = {}  # Dictionary to keep track of count by UsedPlayerMinutes
        
        for r in results:
            for p in self.roster_construction:
                try:
                    if r[p] == 'LOCKED':
                        bad_lu_count += 1
                        if bad_lu_count < 5:
                            print(r)                
                        #if r['UsedPlayerMinutes'] == 288:
                        #    print(r)
                        # Increment the count for the specific UsedPlayerMinutes
                        if r['UsedPlayerMinutes'] in minutes_count:
                            minutes_count[r['UsedPlayerMinutes']] += 1
                        else:
                            minutes_count[r['UsedPlayerMinutes']] = 1
                except KeyError:
                    print(r)
            else:
                good_lu_count += 1

        print(f'bad lineups: {bad_lu_count}, good lineups: {good_lu_count}')
        # Extract and print names
        #for idx in [0, 5000, 10000, 25000, 103310]:
            #if idx < len(results):  # Ensure the index exists in the results
                #original_names = results[idx]['OriginalLineup']
                #try:
                #    new_names = get_names_from_ids(results[idx])
                #except:
                #    print(results)
                #    break
                #print(f"Original lineup for result {idx}: {original_names}")
                #print(f"New lineup for result {idx}: {new_names}, salary: {results[idx]['Salary']}, fpts: {results[idx]['Points']}")
                #print(results[idx])
                #print("-----")
        end = time.time()
# Assuming results is a list of tuples like the one you provided
        self.contest_lineups = {lineup['EntryId']: lineup for lineup in results}
        print('lineup after guessing:')
        print(self.contest_lineups[self.first_idx])
        self.count_lineups_and_extract_fields()
        print('guessing contest lines took {} seconds'.format(end-start))
    
    @staticmethod
    def generate_lineups(
        key,
        lineup,
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
        matchups,
        num_players_in_roster,
        site,
        roster_construction,
        pos_index_dict,
        player_salary_floor
    ):
        rng = np.random.default_rng()  # Use default_rng for a more modern RNG
        lineup_copy = lineup.copy()
        iteration_count = 0
        total_players = num_players_in_roster
        reasonable_projection = optimal_score - (max_pct_off_optimal * optimal_score)
        max_players_per_team = 4 if site == "fd" else None
        max_attempts = 10  # set a maximum number of attempts before reducing floors
        player_teams = []
        lineup_matchups = []
        backoff_factor = 0.95  # You can adjust this to change how much you back off each time
        min_salary_floor = 1  # Define the minimum salary floor
        min_reasonable_projection = 1  # Define the minimum reasonable projection

        def is_valid_lineup(salary, proj, player_teams):
            # Helper function to determine if the current lineup is valid
            if not (salary_floor <= salary <= salary_ceiling):
                return False
            if proj < reasonable_projection:
                return False
            if max_players_per_team is not None:
                if any(count > max_players_per_team for count in Counter(player_teams).values()):
                    return False
            return True

        def finalize_lineup(lineup, salary, proj, unlocked_proj):
            # Helper function to finalize the lineup before returning
            lineup['Salary'] = int(salary) if np.isscalar(salary) else int(salary[0])
            lineup['ProjectedFieldFpts'] = proj if np.isscalar(proj) else proj[0]
            lineup['UnlockedFieldFpts'] = unlocked_proj if np.isscalar(unlocked_proj) else unlocked_proj[0]
            #if lineup['EntryId'] == '3983870229':
            #    log_lineup_state("Finalized Lineup:", lineup)
            #print(lineup)
            return lineup
    
        def reset_lineup():
            # Helper function to reset lineup to the initial state
            nonlocal salary, proj, unlocked_proj, player_teams, lineup_matchups, in_lineup, players_remaining
            salary = lineup_copy['LockedSalary']
            proj = lineup_copy['ProjectedFieldFpts']
            unlocked_proj = 0
            player_teams = []
            lineup_matchups = []
            in_lineup.fill(0)
            players_remaining = lineup_copy['UnlockedPlayers']
            return lineup_copy.copy()

        def log_lineup_state(action, state):
            print(f"{action} - Lineup State for {key}: {state}")

        # Start of lineup generation logic
        if lineup['UserLu'] or lineup['EmptyLu'] or lineup['UnlockedPlayers'] == 0:
            return lineup.copy()
        while True:
            lineup = reset_lineup()
            for attempt in range(max_attempts):
                for position in roster_construction:
                    if lineup[f"{position}_is_locked"]:
                        continue  # Skip locked positions
                    k = pos_index_dict[position]
                    pos = pos_matrix[:, k]
                    remaining_salary = salary_ceiling - salary
                    max_allowable_salary = (remaining_salary - (player_salary_floor * (players_remaining - 1)))

                    valid_players = np.nonzero(
                        (pos > 0) &
                        (in_lineup == 0) &
                        (salaries <= max_allowable_salary) &  # Players must be affordable within the max allowable salary
                        (salaries >= player_salary_floor)  # Players must meet the minimum salary requirement
                    )[0]
                    if players_remaining == 1:
                        valid_players = np.nonzero(
                            (pos > 0)
                            & (in_lineup == 0)
                            & (salaries <= remaining_salary)
                            & (salary + salaries >= salary_floor)
                            & (salaries <= max_allowable_salary) 
                        )[0]
                    else:
                        valid_players = np.nonzero(
                            (pos > 0)
                            & (in_lineup == 0)
                            & (salaries <= remaining_salary)
                            & (salaries <= max_allowable_salary) 
                        )[0]
                    if not valid_players.size:
                        lineup = reset_lineup()  # No valid players, reset lineup
                        break  # Break out of the position loop to restart
                    #print(f'{key} found valid players for position {p}: {valid_players}')
                    # grab names of players eligible
                    plyr_list = ids[valid_players]
                    # create np array of probability of being seelcted based on ownership and who is eligible at the position
                    prob_list = ownership[valid_players]
                    prob_list = prob_list / prob_list.sum()
                    if players_remaining == 1:
                        boosted_salaries = np.array(
                            [
                                salary_boost(s, salary_ceiling)
                                for s in salaries[valid_players]
                            ]
                        )
                        boosted_probabilities = prob_list * boosted_salaries
                        boosted_probabilities /= (
                            boosted_probabilities.sum()
                        )  # normalize to ensure it sums to 1
                    try:
                        if players_remaining == 1:
                            choice = rng.choice(plyr_list, p=boosted_probabilities)
                        else:
                            choice = rng.choice(plyr_list, p=prob_list)
                    except:
                        lineup = reset_lineup()
                        #if lineup['EntryId'] == '3983870229':
                        #    log_lineup_state("can't find choice", lineup)
                        #    if players_remaining == 1:
                        #        print(f'{key} unable to find choice for position {position}, plyr_list {plyr_list}, boosted probs {boosted_probabilities}')
                        #    else:
                        #        print(f'{key} unable to find choice for position {position}, plyr_list {plyr_list}, prob list {prob_list}')                                
                        #print(f'{key} restarting lineup generation {lineup}, {players_remaining}')
                        continue  # Skip to the next iteration of the while loop
                    players_remaining -= 1
                    choice_idx = np.nonzero(ids == choice)[0]
                    #if lineup['EntryId'] == '3983870229':
                    #    print(choice, salary, proj, reasonable_projection, salary_floor)
                    #    log_lineup_state("Before adding player", lineup)
                    lineup[position] = str(choice)  # Adding player
                    in_lineup[choice_idx] = 1
                    salary += salaries[choice_idx]
                    proj += projections[choice_idx]
                    unlocked_proj += projections[choice_idx]
                    #if lineup['EntryId'] == '3983870229':
                    #    print(choice, salary, proj, reasonable_projection, salary_floor)
                    #    log_lineup_state("After adding player", lineup)
                    player_teams.append(teams[choice_idx][0])
                    lineup_matchups.append(matchups[choice_idx[0]])
                if players_remaining == 0:
                    # Check if the lineup is valid after all positions are filled
                    if is_valid_lineup(salary, proj, player_teams):
                        return finalize_lineup(lineup, salary, proj, unlocked_proj)  # If lineup is valid, finalize and return
                #print(f'unable to find valid lineup for {key}, {lineup}')
                lineup = reset_lineup()
                
        # Backoff strategy if max_attempts is reached without a valid lineup
            print(f"Backoff triggered for lineup with key: {key}. New salary floor: {salary_floor}, new projection: {reasonable_projection}")

            if salary_floor > min_salary_floor:
                salary_floor *= backoff_factor
            else:
                salary_floor = min_salary_floor  # Ensure not to go below min

            if reasonable_projection > min_reasonable_projection:
                reasonable_projection *= backoff_factor
            else:
                reasonable_projection = min_reasonable_projection  # Ensure not to go below min
            # Conditions to stop trying if it gets too low
            if salary_floor == min_salary_floor and reasonable_projection == min_reasonable_projection:
                #print(f"Minimum thresholds reached without a valid lineup for key {key}.")
                return lineup  # Return the best attempt or a lineup indicating failure


    def calc_gamma(self, mean, sd):
        alpha = (mean / sd) ** 2
        beta = sd**2 / mean
        return alpha, beta

    def count_lineups_and_extract_fields(self):

        for entry_id, lineup_info in self.contest_lineups.items():
            #print(entry_id, lineup_info)
            # Create a list of player IDs to represent the actual lineup
            try:
                actual_lineup_list = [lineup_info[pos] for pos in self.roster_construction]
            except:
                print(lineup_info)

            # Create a frozenset of player IDs to represent the lineup uniquely for counting duplicates
            lineup_set = frozenset(actual_lineup_list)

            # If this is the first time we see this lineup, initialize its info in the dictionary
            if lineup_set not in self.field_lineups:
                self.field_lineups[lineup_set] = {
                    'Count': 1,
                    'BayesianProjectedFpts': lineup_info['BayesianProjectedFpts'],
                    'BayesianProjectedVar': lineup_info['BayesianProjectedVar'],
                    'Lineup': actual_lineup_list,  # The list of player IDs,
                    'EntryIds': [],
                    'ROI': 0,
                    'Wins': 0,
                    'Top1Percent': 0,
                    'Cashes': 0
                }
                self.field_lineups[lineup_set]['EntryIds'].append(entry_id)
            else:
                # Increment the count for this lineup
                self.field_lineups[lineup_set]['Count'] += 1
                self.field_lineups[lineup_set]['EntryIds'].append(entry_id)

        

    @staticmethod
    def run_simulation_for_game(
        team1_id,
        team1,
        team2_id,
        team2,
        num_iterations,
        roster_construction,
        time_remaining_dict
    ):
        # Define correlations between positions
        if time_remaining_dict[team1_id] == 0:
            game = team1 + team2
            temp_fpts_dict = {}
            for i, player in enumerate(game):
                temp_fpts_dict[player["ID"]] = np.full(num_iterations, player["BayesianProjectedFpts"])
                # If time remaining is zero, we set the player's fantasy points to their BayesianProjectedFpts
        else: 
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
                    "C": -0.1231,
                }

                if player1["Team"] == player2["Team"] and player1["Position"][0] in [
                    "PG",
                    "SG",
                    "SF",
                    "PF",
                    "C",
                ]:
                    primary_position = player1["Position"][0]
                    return position_correlations[primary_position]

                if player1["Team"] != player2["Team"]:
                    player_2_pos = "Opp " + str(player2["Position"][0])
                else:
                    player_2_pos = player2["Position"][0]

                return player1["Correlations"].get(
                    player_2_pos, 0
                )  # Default to 0 if no correlation is found

            def build_covariance_matrix(players):
                N = len(players)
                matrix = [[0 for _ in range(N)] for _ in range(N)]
                corr_matrix = [[0 for _ in range(N)] for _ in range(N)]

                for i in range(N):
                    for j in range(N):
                        if i == j:
                            matrix[i][j] = (
                                players[i]["BayesianProjectedVar"]
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
                return matrix

            game = team1 + team2
            covariance_matrix, corr_matrix = build_covariance_matrix(game)
            # print(team1_id, team2_id)
            # print(corr_matrix)
            corr_matrix = np.array(corr_matrix)

            # Given eigenvalues and eigenvectors from previous code
            eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

            # Set negative eigenvalues to zero
            eigenvalues[eigenvalues < 0] = 0

            # Reconstruct the matrix
            covariance_matrix = eigenvectors.dot(np.diag(eigenvalues)).dot(eigenvectors.T)

            try:
                samples = multivariate_normal.rvs(
                    mean=[player["BayesianProjectedFpts"] for player in game],
                    cov=covariance_matrix,
                    size=num_iterations,
                )
            except:
                print(team1_id, team2_id, "bad matrix", covariance_matrix)

            player_samples = []
            for i, player in enumerate(game):
                if "QB" in player["Position"]:
                    sample = samples[:, i]
                else:
                    sample = samples[:, i]
                # if player['Team'] in ['LAR','SEA']:
                #     print(player['Name'], player['Fpts'], player['StdDev'], sample, np.mean(sample), np.std(sample))
                player_samples.append(sample)

            temp_fpts_dict = {}
            # print(team1_id, team2_id, len(game), uniform_samples.T.shape, len(player_samples), covariance_matrix.shape )

            for i, player in enumerate(game):
                temp_fpts_dict[player["ID"]] = player_samples[i]
                
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

        start_time = time.time()
        temp_fpts_dict = {}
        size = self.num_iterations
        game_simulation_params = []
        #random_keys = random.sample(list(self.contest_lineups.keys()), 5)
        #random_5_entries = {key: self.contest_lineups[key] for key in random_keys}
        #print(random_5_entries)
        if len(self.matchups)>0:
            for m in self.matchups:
                game_simulation_params.append(
                    (
                        m[0],
                        self.teams_dict[m[0]],
                        m[1],
                        self.teams_dict[m[1]],
                        self.num_iterations,
                        self.roster_construction,
                        self.time_remaining_dict
                    )
                )
            with multiprocessing.Pool() as pool:
                results = pool.starmap(self.run_simulation_for_game, game_simulation_params)

            for res in results:
                temp_fpts_dict.update(res)  

        field_lineups_count = np.array(
            [self.field_lineups[idx]["Count"] for idx in self.field_lineups.keys()]
        )

        fpts_array = np.zeros((len(self.field_lineups), self.num_iterations))
        
        for index, (keys, values) in enumerate(self.field_lineups.items()):
            try:
                fpts_sim = sum([temp_fpts_dict[player] for player in values["Lineup"]])
            except KeyError:
                for player in values["Lineup"]:
                    if player not in temp_fpts_dict.keys():
                        print(player)
                        # for k,v in self.player_dict.items():
                        # if v['ID'] == player:
                        #        print(k,v)
                # print('cant find player in sim dict', values["Lineup"], temp_fpts_dict.keys())
            # store lineup fpts sum in 2d np array where index (row) corresponds to index of field_lineups and columns are the fpts from each sim
            fpts_array[index] = fpts_sim
            
        fpts_array = fpts_array.astype(np.float16)
        print(fpts_array)
        # ranks = np.argsort(fpts_array, axis=0)[::-1].astype(np.uint16)
        ranks = np.argsort(-fpts_array, axis=0).astype(np.uint32)

        # count wins, top 10s vectorized
        wins, win_counts = np.unique(ranks[0, :], return_counts=True)

        top1pct, top1pct_counts = np.unique(
            ranks[0 : math.ceil(0.01 * len(self.field_lineups)), :], return_counts=True
        )

        payout_array = np.array(list(self.payout_structure.values()))
        # subtract entry fee
        payout_array = payout_array - self.entry_fee
        l_array = np.full(
            shape=self.field_size - len(payout_array), fill_value=-self.entry_fee
        )
        payout_array = np.concatenate((payout_array, l_array))
        
        cashes, cash_counts = np.unique(ranks[0:len(list(self.payout_structure.values()))], return_counts=True)

        # Adjusted ROI calculation
        # print(field_lineups_count.shape, payout_array.shape, ranks.shape, fpts_array.shape)

        # Split the simulation indices into chunks
        self.lineup_to_int = {lineup: index for index, lineup in enumerate(self.field_lineups.keys())}
        field_lineups_keys_array = np.array([self.lineup_to_int[lineup] for lineup in self.field_lineups.keys()])

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
        with multiprocessing.Pool() as pool:
            results = pool.map(self.calculate_payouts, simulation_chunks)

        combined_result_array = np.sum(results, axis=0)
        
        total_sum = 0
        index_to_key = list(self.field_lineups.keys())
        for idx, roi in enumerate(combined_result_array):
            lineup_int_key = self.lineup_to_int[index_to_key[idx]]  # Convert lineup string to integer key
            lineup_count = self.field_lineups[index_to_key[idx]]["Count"]  # Access using the original lineup string
            total_sum += roi * lineup_count
            self.field_lineups[index_to_key[idx]]["ROI"] += roi  # Access using the original lineup string

        # Second loop for wins and top1pct
        # Assuming wins and top1pct are arrays or lists of integers that correspond to the mapped lineup integers

        for lineup_key in self.field_lineups.keys():  # loop through lineup strings
            lineup_int_key = self.lineup_to_int[lineup_key]  # Convert lineup string to integer key

            if lineup_int_key in wins:
                # Find the index where lineup_int_key is found in wins array
                win_index = np.where(wins == lineup_int_key)[0][0]
                self.field_lineups[lineup_key]["Wins"] += win_counts[win_index]

            if lineup_int_key in top1pct:
                # Find the index where lineup_int_key is found in top1pct array
                top1pct_index = np.where(top1pct == lineup_int_key)[0][0]
                self.field_lineups[lineup_key]["Top1Percent"] += top1pct_counts[top1pct_index]

            if lineup_int_key in cashes:
                # Find the index where lineup_int_key is found in wins array
                cash_index = np.where(cashes == lineup_int_key)[0][0]
                self.field_lineups[lineup_key]["Cashes"] += cash_counts[cash_index]

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
        for index, y in self.field_lineups.items():
            win_p = round(y["Wins"] / self.num_iterations * 100, 2)
            top10_p = round((y["Top1Percent"] / self.num_iterations) * 100, 2)
            cash_p = round(y["Cashes"] / self.num_iterations * 100, 2)
            simDupes = y["Count"]
            lu_idx = self.lineup_to_int[index]
            for entry in y['EntryIds']:
                x = self.contest_lineups[entry]
                # if index == 0:
                #    print(x)
                lu_type = x["Type"]
                userName = x['User']
                salary = x['Salary']
                fpts_p = x['BayesianProjectedFpts']
                std_p = np.sqrt(x['BayesianProjectedVar'])
                own_p = []
                lu_names = []
                lu_teams = []
                players_vs_def = 0
                def_opps = []
                for p in self.roster_construction:
                    p_str = f'{p}_is_locked'
                    id = x[p]
                    if id == '':
                        lu_names.append('null')
                        lu_teams.append('null')
                        own_p.append(0)
                        continue
                    for k, v in self.player_dict.items():
                        if v["ID"] == id:
                            lu_names.append(v["DK Name"])
                            lu_teams.append(v["Team"])
                            own_p.append(v["Ownership"] / 100)
                            if x[p_str] == False:
                                fpts_p += v['Fpts']
                                std_p += v['StdDev']
                ceil_p = fpts_p + std_p
                counter = Counter(lu_teams)
                stacks = counter.most_common()
                

                # Find the QB team in stacks and set it as primary stack, remove it from stacks and subtract 1 to make sure qb isn't counted
                primaryStack = str(stacks[0][0]) + " " + str(stacks[0][1])
                # After removing QB team, the first team in stacks will be the team with most players not in QB stack
                try:
                    secondaryStack = str(stacks[1][0]) + " " + str(stacks[1][1])
                except:
                    secondaryStack = ''
                own_s = np.sum(own_p)
                own_p = np.prod(own_p)
                if self.site == "dk":
                    if self.use_contest_data:
                        roi_p = round(
                            y["ROI"] / self.entry_fee / self.num_iterations * 100, 2
                        )
                        roi_round = round(y["ROI"] / self.num_iterations, 2)                        
                        self.contest_entries[userName]['ROI'] += y['ROI']
                        self.contest_entries[userName]['Wins'] += y['Wins']
                        self.contest_entries[userName]['Top1'] += y['Top1Percent']
                        lineup_str = (
                            f"{lu_names[0].replace('#', '-')}"
                            f" ({x['PG']}),"
                            f"{lu_names[1].replace('#', '-')}"
                            f" ({x['SG']}),"
                            f"{lu_names[2].replace('#', '-')}"
                            f" ({x['SF']}),"
                            f"{lu_names[3].replace('#', '-')}"
                            f" ({x['PF']}),"
                            f"{lu_names[4].replace('#', '-')}"
                            f" ({x['C']}),"
                            f"{lu_names[5].replace('#', '-')}"
                            f" ({x['G']}),"
                            f"{lu_names[6].replace('#', '-')}"
                            f" ({x['F']}),"
                            f"{lu_names[7].replace('#', '-')}"
                            f" ({x['UTIL']}),"
                            f"{fpts_p},{ceil_p},{salary},{win_p}%,{top10_p}%,{roi_p}%,{own_p},{own_s},${roi_round},{primaryStack},{secondaryStack},{lu_type},{simDupes},{userName},{lu_idx},{entry}"
                        )
                    else:
                        lineup_str = (
                            f"{lu_names[0].replace('#', '-')}"
                            f" ({x['Lineup'][0]}),"
                            f"{lu_names[1].replace('#', '-')}"
                            f" ({x['Lineup'][1]}),"
                            f"{lu_names[2].replace('#', '-')}"
                            f" ({x['Lineup'][2]}),"
                            f"{lu_names[3].replace('#', '-')}"
                            f" ({x['Lineup'][3]}),"
                            f"{lu_names[4].replace('#', '-')}"
                            f" ({x['Lineup'][4]}),"
                            f"{lu_names[5].replace('#', '-')}"
                            f" ({x['Lineup'][5]}),"
                            f"{lu_names[6].replace('#', '-')}"
                            f" ({x['Lineup'][6]}),"
                            f"{lu_names[7].replace('#', '-')}"
                            f" ({x['Lineup'][7]}),"
                            f"{fpts_p},{ceil_p},{salary},{win_p}%,{top10_p}%,{own_p},{own_s},{primaryStack},{secondaryStack},{lu_type},{simDupes},{userName},{lu_idx},{entry}"
                        )
                elif self.site == "fd":
                    if self.use_contest_data:
                        roi_p = round(
                            x["ROI"] / self.entry_fee / self.num_iterations * 100, 2
                        )
                        roi_round = round(x["ROI"] / self.num_iterations, 2)
                        lineup_str = (
                            f"{lu_names[0].replace('#', '-')}"
                            f" ({x['Lineup'][0]}),"
                            f"{lu_names[1].replace('#', '-')}"
                            f" ({x['Lineup'][1]}),"
                            f"{lu_names[2].replace('#', '-')}"
                            f" ({x['Lineup'][2]}),"
                            f"{lu_names[3].replace('#', '-')}"
                            f" ({x['Lineup'][3]}),"
                            f"{lu_names[4].replace('#', '-')}"
                            f" ({x['Lineup'][4]}),"
                            f"{lu_names[5].replace('#', '-')}"
                            f" ({x['Lineup'][5]}),"
                            f"{lu_names[6].replace('#', '-')}"
                            f" ({x['Lineup'][6]}),"
                            f"{lu_names[7].replace('#', '-')}"
                            f" ({x['Lineup'][7]}),"
                            f"{lu_names[8].replace('#', '-')}"
                            f" ({x['Lineup'][8]}),"
                            f"{fpts_p},{ceil_p},{salary},{win_p}%,{top10_p}%,{roi_p}%,{own_p},{own_s},${roi_round},{primaryStack},{secondaryStack},{lu_type},{simDupes},{userName},{lu_idx},{entry}"
                        )
                    else:
                        lineup_str = (
                            f"{lu_names[0].replace('#', '-')}"
                            f" ({x['Lineup'][0]}),"
                            f"{lu_names[1].replace('#', '-')}"
                            f" ({x['Lineup'][1]}),"
                            f"{lu_names[2].replace('#', '-')}"
                            f" ({x['Lineup'][2]}),"
                            f"{lu_names[3].replace('#', '-')}"
                            f" ({x['Lineup'][3]}),"
                            f"{lu_names[4].replace('#', '-')}"
                            f" ({x['Lineup'][4]}),"
                            f"{lu_names[5].replace('#', '-')}"
                            f" ({x['Lineup'][5]}),"
                            f"{lu_names[6].replace('#', '-')}"
                            f" ({x['Lineup'][6]}),"
                            f"{lu_names[7].replace('#', '-')}"
                            f" ({x['Lineup'][7]}),"
                            f"{lu_names[8].replace('#', '-')}"
                            f" ({x['Lineup'][8]}),"
                            f"{fpts_p},{ceil_p},{salary},{win_p}%,{top10_p}%,{own_p},{own_s},{primaryStack},{secondaryStack},{lu_type},{simDupes},{userName},{lu_idx},{entry}"
                        )
                unique[index] = lineup_str

        out_path = os.path.join(
            os.path.dirname(__file__),
            "../output/{}_gpp_sim_lineups_{}_{}.csv".format(
                self.site, self.field_size, self.num_iterations
            ),
        )
        with open(out_path, "w") as f:
            if self.site == "dk":
                if self.use_contest_data:
                    f.write(
                        "PG,SG,SF,PF,C,G,F,UTIL,Fpts Proj,Ceiling,Salary,Win %,Top 1%,ROI%,Proj. Own. Product,Own. Sum,Avg. Return,Stack1 Type,Stack2 Type,Lineup Type,Sim Dupes,User Name,Lineup Index,Entry ID\n"
                    )
                else:
                    f.write(
                        "PG,SG,SF,PF,C,G,F,UTIL,Fpts Proj,Ceiling,Salary,Win %,Top 1%, Proj. Own. Product,Own. Sum,Stack1 Type,Stack2 Type,Lineup Type,Sim Dupes,User Name,Lineup Index,Entry ID\n"
                    )
            elif self.site == "fd":
                if self.use_contest_data:
                    f.write(
                        "PG,PG,SG,SG,SF,SF,PF,PF,C,DST,Fpts Proj,Ceiling,Salary,Win %,Top 1%,ROI%,Proj. Own. Product,Own. Sum,Avg. Return,Stack1 Type,Stack2 Type,Lineup Type,Sim Dupes,User Name,Lineup Index,Entry ID\n"
                    )
                else:
                    f.write(
                        "PG,PG,SG,SG,SF,SF,PF,PF,C,Fpts Proj,Ceiling,Salary,Win %,Top 1%,Proj. Own. Product,Own. Sum,Stack1 Type,Stack2 Type,Lineup Type,Sim Dupes,User Name,Lineup Index,Entry ID\n"
                    )

            for fpts, lineup_str in unique.items():
                f.write("%s\n" % lineup_str)
        out_path = os.path.join(
            os.path.dirname(__file__),
            "../output/{}_lateswap_sim_user_equity{}_{}.csv".format(
                self.site, self.field_size, self.num_iterations
            ),
        )
        with open(out_path, 'w', newline='') as csvfile:
            # Determine the fieldnames from the keys of the first item in contest_entries
            fieldnames = ['Name'] + list(next(iter(self.contest_entries.values())).keys())
            # Create a CSV DictWriter instance
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write the header
            writer.writeheader()

            # Write the rows
            for name, data in self.contest_entries.items():
                # Insert the username into the row dict
                row = {'Name': name}
                # Update the row dict with the data from contest_entries
                row.update(data)
                writer.writerow(row)
        out_path = os.path.join(
            os.path.dirname(__file__),
            "../output/{}_lateswap_sim_player_exposure_{}_{}.csv".format(
                self.site, self.field_size, self.num_iterations
            ),
        )
        with open(out_path, "w") as f:
            f.write(
                "Player,Position,Team,Win%,Top1%,Sim. Own%,Proj. Own%,Avg. Return\n"
            )
            unique_players = {}
            for val in self.field_lineups.values():
                for player in val["Lineup"]:
                    if player not in unique_players:
                        unique_players[player] = {
                            "Wins": val["Wins"],
                            "Top1Percent": val["Top1Percent"],
                            "In": val["Count"],
                            "ROI": val["ROI"],
                        }
                    else:
                        unique_players[player]["Wins"] = (
                            unique_players[player]["Wins"] + val["Wins"]
                        )
                        unique_players[player]["Top1Percent"] = (
                            unique_players[player]["Top1Percent"] + val["Top1Percent"]
                        )
                        unique_players[player]["In"] += val["Count"]
                        unique_players[player]["ROI"] = (
                            unique_players[player]["ROI"] + val["ROI"]
                        )

            for player, data in unique_players.items():
                field_p = round(data["In"] / self.field_size * 100, 2)
                win_p = round(data["Wins"] / self.num_iterations * 100, 2)
                top10_p = round(data["Top1Percent"] / self.num_iterations, 2)
                roi_p = round(data["ROI"] / data["In"] / self.num_iterations, 2)
                for k, v in self.player_dict.items():
                    if player == v["ID"]:
                        proj_own = v["Ownership"]
                        p_name = v["Name"]
                        position = "/".join(v.get("Position"))
                        team = v.get("Team")
                        break
                f.write(
                    "{},{},{},{}%,{}%,{}%,{}%,${}\n".format(
                        p_name.replace("#", "-"),
                        position,
                        team,
                        win_p,
                        top10_p,
                        field_p,
                        proj_own,
                        roi_p,
                    )
                )
        sorted_lineups = []
        for lineup, old_lineup in self.output_lineups:
            sorted_lineup = self.sort_lineup(lineup)
            sorted_lineup = self.adjust_roster_for_late_swap(sorted_lineup, old_lineup)
            new_sorted_lineup = []
            sorted_lineups.append((sorted_lineup, old_lineup))

        late_swap_lineups_contest_entry_dict = {
            (old_lineup["contest_id"], old_lineup["entry_id"]): new_lineup
            for new_lineup, old_lineup in sorted_lineups
        }
        if 'late_swap_path' in self.config.keys():
            late_swap_path = os.path.join(
                os.path.dirname(__file__),
                "../{}_data/{}".format(self.site, self.config["late_swap_path"]),
            )

            # Read the existing data first
            fieldnames = []
            with open(late_swap_path, "r", encoding="utf-8-sig") as file:
                reader = csv.DictReader(file)
                fieldnames = reader.fieldnames
                rows = [row for row in reader]

            PLACEHOLDER = "PLACEHOLDER_FOR_NONE"
            # If any row has a None key, ensure the placeholder is in the fieldnames
            for row in rows:
                if None in row and PLACEHOLDER not in fieldnames:
                    fieldnames.append(PLACEHOLDER)

            # Now, modify the rows
            updated_rows = []
            for row in rows:
                if row["Entry ID"] != "":
                    contest_id = row["Contest ID"]
                    entry_id = row["Entry ID"]
                    matching_lineup = late_swap_lineups_contest_entry_dict.get(
                        (contest_id, entry_id)
                    )
                    if matching_lineup:
                        row[
                            "PG"
                        ] = f"{self.player_dict[matching_lineup[0]]['Name']} ({self.player_dict[matching_lineup[0]]['ID']})"
                        row[
                            "SG"
                        ] = f"{self.player_dict[matching_lineup[1]]['Name']} ({self.player_dict[matching_lineup[1]]['ID']})"
                        row[
                            "SF"
                        ] = f"{self.player_dict[matching_lineup[2]]['Name']} ({self.player_dict[matching_lineup[2]]['ID']})"
                        row[
                            "PF"
                        ] = f"{self.player_dict[matching_lineup[3]]['Name']} ({self.player_dict[matching_lineup[3]]['ID']})"
                        row[
                            "C"
                        ] = f"{self.player_dict[matching_lineup[4]]['Name']} ({self.player_dict[matching_lineup[4]]['ID']})"
                        row[
                            "G"
                        ] = f"{self.player_dict[matching_lineup[5]]['Name']} ({self.player_dict[matching_lineup[5]]['ID']})"
                        row[
                            "F"
                        ] = f"{self.player_dict[matching_lineup[6]]['Name']} ({self.player_dict[matching_lineup[6]]['ID']})"
                        row[
                            "UTIL"
                        ] = f"{self.player_dict[matching_lineup[7]]['Name']} ({self.player_dict[matching_lineup[7]]['ID']})"

                updated_rows.append(row)

            new_late_swap_path = os.path.join(
                os.path.dirname(__file__),
                "../output/late_swap_{}.csv".format(
                    datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                ),
            )

            with open(new_late_swap_path, "w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(
                    file, fieldnames=[PLACEHOLDER if f is None else f for f in fieldnames]
                )
                writer.writeheader()
                for row in updated_rows:
                    if None in row:
                        row[PLACEHOLDER] = row.pop(None)
                    writer.writerow(row)

            with open(new_late_swap_path, "r", encoding="utf-8-sig") as file:
                content = file.read().replace(PLACEHOLDER, "")

            with open(new_late_swap_path, "w", encoding="utf-8-sig") as file:
                file.write(content)

        print("Output done.")

    def sort_lineup(self, lineup):
        if self.site == "dk":
            order = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
            sorted_lineup = [None] * 8
        else:
            order = ["PG", "PG", "SG", "SG", "SF", "SF", "PF", "PF", "C"]
            sorted_lineup = [None] * 9

        for player in lineup:
            player_key, pos, _ = player
            order_idx = order.index(pos)
            if sorted_lineup[order_idx] is None:
                sorted_lineup[order_idx] = player_key
            else:
                sorted_lineup[order_idx + 1] = player_key
        return sorted_lineup
    
    def is_valid_for_position(self, player, position_idx):
        return any(
            pos in self.position_map[position_idx]
            for pos in self.player_dict[player]['Position']
        )

    def adjust_roster_for_late_swap(self, lineup, old_lineup):
        if self.site == "fd":
            return lineup

        POSITIONS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]

        def is_locked(position_index):
            if 0 <= position_index < 8:
                position_name = POSITIONS[position_index]
                return old_lineup[f"{position_name}_is_locked"]
            else:
                raise ValueError(f"Invalid position index: {position_index}")

        for i, position in enumerate(POSITIONS):
            # Only check G, F, and UTIL positions
            if position in ["G", "F", "UTIL"]:
                current_player = lineup[i]
                current_player_start_time = self.player_dict[current_player]["GameTime"]

                # Look for a swap candidate among primary positions
                for primary_i, primary_pos in enumerate(
                    POSITIONS[:5]
                ):  # Only the primary positions (0 to 4)
                    primary_player = lineup[primary_i]
                    primary_player_start_time = self.player_dict[primary_player][
                        "GameTime"
                    ]

                    if is_locked(primary_i) or is_locked(i):
                        print(
                            f"Skipping swap between {primary_player} and {current_player} because one of them is locked"
                        )
                        continue

                    # Check the conditions for the swap
                    primary_player_positions = self.player_dict[primary_player][
                        "Position"
                    ]
                    current_player_positions = self.player_dict[current_player][
                        "Position"
                    ]
                    # Check the conditions for the swap
                    if (
                        primary_player_start_time > current_player_start_time
                        and self.is_valid_for_position(primary_player, i)
                        and self.is_valid_for_position(current_player, primary_i)
                    ):
                        print(
                            f"Swapping {primary_player} and {current_player}, for positions {primary_pos} and {position}"
                        )
                        lineup[i], lineup[primary_i] = lineup[primary_i], lineup[i]
                        break  # Break out of the inner loop once a swap is made
        return lineup


    def sort_lineup(self, lineup):
        if self.site == "dk":
            order = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
            sorted_lineup = [None] * 8
        else:
            order = ["PG", "PG", "SG", "SG", "SF", "SF", "PF", "PF", "C"]
            sorted_lineup = [None] * 9

        for player in lineup:
            player_key, pos, _ = player
            order_idx = order.index(pos)
            if sorted_lineup[order_idx] is None:
                sorted_lineup[order_idx] = player_key
            else:
                sorted_lineup[order_idx + 1] = player_key
        return sorted_lineup

    def adjust_roster_for_late_swap(self, lineup, old_lineup):
        if self.site == "fd":
            return lineup

        POSITIONS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]

        def is_locked(position_index):
            if 0 <= position_index < 8:
                position_name = POSITIONS[position_index]
                return old_lineup[f"{position_name}_is_locked"]
            else:
                raise ValueError(f"Invalid position index: {position_index}")

        for i, position in enumerate(POSITIONS):
            # Only check G, F, and UTIL positions
            if position in ["G", "F", "UTIL"]:
                current_player = lineup[i]
                current_player_start_time = self.player_dict[current_player]["GameTime"]

                # Look for a swap candidate among primary positions
                for primary_i, primary_pos in enumerate(
                    POSITIONS[:5]
                ):  # Only the primary positions (0 to 4)
                    primary_player = lineup[primary_i]
                    primary_player_start_time = self.player_dict[primary_player][
                        "GameTime"
                    ]

                    if is_locked(primary_i) or is_locked(i):
                        print(
                            f"Skipping swap between {primary_player} and {current_player} because one of them is locked"
                        )
                        continue

                    # Check the conditions for the swap
                    primary_player_positions = self.player_dict[primary_player][
                        "Position"
                    ]
                    current_player_positions = self.player_dict[current_player][
                        "Position"
                    ]
                    if (
                        primary_player_start_time > current_player_start_time
                        and any(
                            pos in primary_player_positions
                            for pos in current_player_positions
                        )
                        and any(
                            pos in current_player_positions
                            for pos in primary_player_positions
                        )
                    ):
                        # Perform the swap
                        print(
                            f"Swapping {primary_player} and {current_player}, for positions {primary_pos} and {position}"
                        )
                        lineup[i], lineup[primary_i] = lineup[primary_i], lineup[i]
                        break  # Break out of the inner loop once a swap is made
        return lineup
