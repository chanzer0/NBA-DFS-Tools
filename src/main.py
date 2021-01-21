

import sys
from nfl_optimizer import *
from nba_optimizer import *
from nba_gpp_simulator import *
from nba_evolutionary_lineup_selector import *
from nba_showdown_optimizer import *
from windows_inhibitor import *

def main(arguments):
    if len(arguments) < 3 or len(arguments) > 6:
        print('Incorrect usage. Please see `README.md` for proper usage.')
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

    elif process == 'sd':
        num_lineups = arguments[3]
        num_uniques = arguments[4]
        use_randomness = arguments[5]
        opto = NBA_Showdown_Optimizer(site, num_lineups, use_randomness, num_uniques)
        opto.optimize()
        opto.output()

    elif process == 'sim':
        site = arguments[1]
        field_size = -1
        num_iterations = -1
        use_contest_data = False
        use_file_upload = False
        if arguments[3] == 'cid':
            use_contest_data = True
        else:
            field_size = arguments[3]

        if arguments[4] == 'file':
            use_file_upload = True
            num_iterations = arguments[5]
        else:
            num_iterations = arguments[4]

        sim = NBA_GPP_Simulator(site, field_size, num_iterations, use_contest_data, use_file_upload)
        sim.generate_field_lineups()
        sim.run_tournament_simulation()
        sim.output()

    elif process == 'swaptimize':
        opto = NBA_Optimizer(site)
        opto.swaptimize()


if __name__ == "__main__":
    main(sys.argv)