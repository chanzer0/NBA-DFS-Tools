

import sys
from nfl_optimizer import *
from nba_optimizer import *
from nba_gpp_simulator import *
from nba_evolutionary_lineup_selector import *
from windows_inhibitor import *

def main(arguments):
    if len(arguments) < 3 or len(arguments) > 6:
        print('For opto: `python .\main.py <site> opto <num_lineups> <num_uniques> <use_rand>`')
        print('For sim: `python .\main.py <site> sim <field_size> <num_iterations>`')
        exit()

    site = arguments[1]
    process = arguments[2]
    if process == 'opto':
        num_lineups = arguments[3]
        num_uniques = arguments[4]
        use_randomness = arguments[5]
        opto = NBA_Optimizer(site, num_lineups, use_randomness, num_uniques)
        opto.optimize()
        opto.output()

    elif process == 'sim':
        pass


if __name__ == "__main__":
    main(sys.argv)