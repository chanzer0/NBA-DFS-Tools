

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
        site = arguments[1]
        field_size = -1
        num_iterations = -1
        contest_id = -1
        # 100691372
        if arguments[3] == 'cid':
            contest_id = arguments[4]
            num_iterations = arguments[5]
        else:
            field_size = arguments[3]
            num_iterations = arguments[4]

        sim = NBA_GPP_Simulator(site, field_size, num_iterations, contest_id)
        sim.generate_field_lineups()
        sim.run_tournament_simulation()
        sim.output()

    elif process == 'swaptimize':
        opto = NBA_Optimizer(site)
        opto.swaptimize()


if __name__ == "__main__":
    main(sys.argv)