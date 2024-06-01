import argparse
from get_trajectories import get_trajectory_choose_date, get_trajectory_book_flight
from get_trajectories import get_trajectory_search_engine


FUNCTION = {'choose-date': get_trajectory_choose_date,
            'book-flight': get_trajectory_book_flight,
            'search-engine': get_trajectory_search_engine}

def get_trajectories_check_repeats(env_type):
    samples = []
    while len(samples) < 100:
        try:
            samples = FUNCTION[env_type](samples)
        except Exception as error:
            print(error)
            continue
            
def get_trajectories_general(env_type):
    for i in range(100):
        FUNCTION[env_type]()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="choose-date"
                        , help="[book-flight, choose-date, search-engine]")
    args = parser.parse_args()
    
    if args.env in ['choose-date', 'search-engine']:
        get_trajectories_check_repeats(args.env)
    
    else:
        get_trajectories_general(args.env)
        