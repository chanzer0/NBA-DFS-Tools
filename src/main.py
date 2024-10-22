from nba_showdown_optimizer import *
from windows_inhibitor import *
from nba_late_swaptimizer import *
import sys
from nba_optimizer import *
from windows_inhibitor import *
from nba_late_swaptimizer import *
from nba_pick5_optimizer import *


def main(arguments):
    if len(arguments) < 3 or len(arguments) > 7:
        print("Incorrect usage. Please see `README.md` for proper usage.")
        exit()

    site = arguments[1]
    process = arguments[2]

    if process == "opto":
        num_lineups = arguments[3]
        num_uniques = arguments[4]
        opto = NBA_Optimizer(site, num_lineups, num_uniques)
        opto.optimize()
        opto.output()

    if process == "pick5":
        num_lineups = arguments[3]
        num_uniques = arguments[4]
        pick5 = NBA_Pick5_Optimizer(site, num_lineups, num_uniques)
        pick5.optimize()
        pick5.output()

    if process == "swap":
        num_uniques = arguments[3]
        swapto = NBA_Late_Swaptimizer(site, num_uniques)
        swapto.swaptimize()
        swapto.output()

    elif process == "swap_sim":
        import nba_swap_sims

        num_uniques = arguments[3]
        num_iterations = int(arguments[4])
        simto = nba_swap_sims.NBA_Swaptimizer_Sims(num_iterations, site, num_uniques)
        simto.swaptimize()
        simto.compute_best_guesses_parallel()
        simto.run_tournament_simulation()
        simto.output()

    elif process == "sd_opto":
        num_lineups = arguments[3]
        num_uniques = arguments[4]
        opto = NBA_Showdown_Optimizer(site, num_lineups, num_uniques)
        opto.optimize()
        opto.output()

    elif process == "sd_sim":
        import nba_showdown_simulator

        field_size = -1
        num_iterations = -1
        use_contest_data = False
        use_file_upload = False
        match_lineup_input_to_field_size = True
        if arguments[3] == "cid":
            use_contest_data = True
        else:
            field_size = arguments[3]

        if arguments[4] == "file":
            use_file_upload = True
            num_iterations = arguments[5]
        else:
            num_iterations = arguments[4]
        # if 'match' in arguments:
        #    match_lineup_input_to_field_size = True
        sim = nba_showdown_simulator.nba_showdown_simulator(
            site, field_size, num_iterations, use_contest_data, use_file_upload
        )
        sim.generate_field_lineups()
        sim.run_tournament_simulation()
        sim.save_results()

    elif process == "sim":
        import nba_gpp_simulator

        site = arguments[1]
        field_size = -1
        num_iterations = -1
        use_contest_data = False
        use_file_upload = False
        match_lineup_input_to_field_size = True
        if arguments[3] == "cid":
            use_contest_data = True
        else:
            field_size = arguments[3]

        if arguments[4] == "file":
            use_file_upload = True
            num_iterations = arguments[5]
        else:
            num_iterations = arguments[4]
        # if 'match' in arguments:
        #    match_lineup_input_to_field_size = True
        sim = nba_gpp_simulator.NBA_GPP_Simulator(
            site, field_size, num_iterations, use_contest_data, use_file_upload
        )
        sim.generate_field_lineups()
        sim.run_tournament_simulation()
        sim.output()


if __name__ == "__main__":
    main(sys.argv)
