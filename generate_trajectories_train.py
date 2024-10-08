import argparse
import time
from get_trajectories import get_trajectory_choose_date, get_trajectory_book_flight
from get_trajectories import get_trajectory_search_engine, get_trajectory_email_inbox
from get_trajectories import get_trajectory_click_checkboxes_soft, get_trajectory_click_tab2_hard
from get_trajectories import get_trajectory_social_media, get_trajectory_social_media_some
from get_trajectories import get_trajectory_use_autocomplete, get_trajectory_tic_tac_toe

FUNCTION = {'choose-date': get_trajectory_choose_date,
            'book-flight': get_trajectory_book_flight,
            'search-engine': get_trajectory_search_engine,
           'email-inbox': get_trajectory_email_inbox,
           'click-checkboxes-soft': get_trajectory_click_checkboxes_soft,
           'click-tab-2-hard':get_trajectory_click_tab2_hard,
           'social-media': get_trajectory_social_media,
           'social-media-some':get_trajectory_social_media_some,
           'use-autocomplete':get_trajectory_use_autocomplete,
           'tic-tac-toe':get_trajectory_tic_tac_toe}

def get_trajectories_env(env_type):
    samples = []
    while len(samples) < 100:
        try:
            samples = FUNCTION[env_type](samples)
        except Exception as error:
            print(error)
            continue
        time.sleep(10)

            
def get_trajectories_general(env_type):
    count =0
    while count < 100:
        try:
            reward = FUNCTION[env_type]()
            if reward > 0:
                count += 1
            print(reward)
        except Exception as error:
            print(error)
            time.sleep(10)
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="social-media")
    args = parser.parse_args()
    
    if args.env in ['click-checkboxes-soft', 'social-media-some','tic-tac-toe']:
        get_trajectories_general(args.env)
    else:
        get_trajectories_env(args.env)
    
        