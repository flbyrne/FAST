from datetime import datetime, timedelta
import argparse
import numpy as np
import re
import random
import os
import time
import dill
import yaml
from math import ceil
from collections import Counter
from environment import MiniWoBEnvironment
from utils import find_ref, save_trajectory, name_file, parse_response, prompt_together, ref2text, get_page_link, search_page,search_item,ref2classes,find_by_id,count_search_results

### NEEDED FOR GEMINI PROMPTS#########
from google.cloud import aiplatform
import vertexai.preview
from vertexai.preview.generative_models import GenerativeModel, Part, ChatSession

from PIL import Image
from google.cloud import storage

#BUCKET = 'miniwobimages'
project_id = "cs224n-420704"
project_id = "neon-bank-425302-s5"
aiplatform.init(project=project_id)
vertexai.preview.init()
model = GenerativeModel(model_name="gemini-1.5-pro-preview-0514")
#client = storage.Client(project=project_id)
#bucket = client.get_bucket(BUCKET)
######################################


### NEEDED FOR LLAMA PROMPTS##########
with open('config.yaml') as f: config = yaml.safe_load(f)
API_KEY = config['api_key']
MODEL="meta-llama/Llama-3-70b-chat-hf"
######################################

SAVE_TO_FOLDER = 'trajectories_train'
MONTHS = ['December', 'November', 'October'
          , 'September', 'August', 'July', 'June', 'May', 'April', 'March', 'February', 'January']



def get_trajectory_search_engine(queries):   
    env_type = 'search-engine'
    action_types = ['CLICK_ELEMENT', 'TYPE_TEXT']
    actions, doms, images, times = [], [], [], []
    
    def execute_action(action_type, ref, text, action_txt):
        print(action_txt)
        action = env.create_action(action_type, ref=ref, text=text)
        observation, reward, terminated, truncated, info = env.step(action)
        dom = observation['dom_elements']
        image = observation['screenshot']
        actions.append(action_txt)
        doms.append(dom)
        images.append(image)
        times.append(int(time.time()-start_time))
        return reward
    
    env = MiniWoBEnvironment(env_type, wait_ms=500,render_mode=None)
    start_time = time.time()
    observation, info = env.reset()
    
    fields = dict(observation['fields'])

    goal = observation['utterance']
    print(goal)
    query = fields['query']
    rank = int(fields['rank'])
    rank1 = rank*1
    for q,r in queries:
        if q == query and r == rank:
            raise Exception('Query and Rank already in list')
    queries.append((query,rank))
    
    
    pages = ceil(rank/3)*1
    item_not_found = True
    
    action_txt  = ''
    dom = observation['dom_elements']
    image = observation['screenshot']

    actions.append(action_txt)
    doms.append(dom)
    images.append(image)
    times.append(int(time.time()-start_time))
    
    action_type, ref, text = action_types[0], 5, ''
    action_txt = 'Click on the textbox - {} {}'.format(action_type, ref)
    execute_action(action_type, ref, text, action_txt)
    
    action_type, ref, text = action_types[1], 5, query
    action_txt = 'Now that you have clicked on the textbox, type {} in the textbox - {} {}'.format(text, action_type, ref)
    execute_action(action_type, ref, text, action_txt)
    
    action_type, ref, text = action_types[0], 6, ''
    action_txt = 'Click on the Search button - {} {}'.format(action_type, ref)
    execute_action(action_type, ref, text, action_txt)

    page = 1
    count = 0
    while item_not_found:
        ref = search_page(query, doms[-1])
        if ref > 0:
            item_not_found = False
            if rank%3 !=0:
                rank1 = rank%3
            else:
                rank1 = 3
        else:
            page +=1
            count += count_search_results(doms[-1])
            rank1 = rank1 - 3
            ref = get_page_link(page, doms[-1])
            action_type, text = action_types[0], ''
            action_txt = 'Counted {} results so far. Go to page {} because {} was not found on page {} - {} {}'.format(
                count, page, query, page-1, action_type, ref)
            execute_action(action_type, ref, text, action_txt)
        
        
    action_type, text = action_types[0], ''
    action_txt = 'Counted {} results, click Search Title {} in position {} - {} {}'.format(rank, query, rank1, action_type, ref)
    reward = execute_action(action_type, ref, text, action_txt)
    
    print(reward)
    env.close()
    save_trajectory(SAVE_TO_FOLDER,actions, doms, images, times, env_type, reward, goal)
    return queries

def get_trajectory_email_inbox(samples):
    env_type = 'email-inbox'
    action_types = ['CLICK_ELEMENT', 'TYPE_TEXT']
    actions, doms, images, times = [], [], [], []
    
    def execute_action(action_type, ref, text, action_txt):
        action = env.create_action(action_type, ref=ref, text=text)
        observation, reward, terminated, truncated, info = env.step(action)
        dom = observation['dom_elements']
        image = observation['screenshot']
        actions.append(action_txt)
        doms.append(dom)
        images.append(image)
        times.append(int(time.time()-start_time))
        return reward
    
    env = MiniWoBEnvironment(env_type, wait_ms=500,render_mode=None)
    start_time = time.time()
    observation, info = env.reset()
    
    fields = dict(observation['fields'])
    
    goal = observation['utterance']
    task = fields['task']
    by = fields['by']
    
    if task in ['delete', 'star']:
        other = ''
    else:
        other = list(fields.items())[2][1]
    for t, b, o in samples:
        if t == task and b == by and o == other:
            raise Exception('Sample already in list')
    samples.append((task, by, other))
    
    action_txt  = ''
    dom = observation['dom_elements']
    image = observation['screenshot']

    actions.append(action_txt)
    doms.append(dom)
    images.append(image)
    times.append(int(time.time()-start_time))
    
    chat = model.start_chat(response_validation = False)
    if task in ['delete', 'star']:
        instruction = '{} the email message by {}'.format(task.capitalize(), by)
        print(instruction)
        prompt = create_prompt(instruction, doms[-1])
        response = chat.send_message(prompt).text
        print(response)
        ref = int(re.search('Dom element ref number:[\* ]*([0-9]*)',response).group(1))
        action_type, text = action_types[0], ''
        action_txt = 'Click on {} icon - {} {}'.format(task, action_type, ref)
        reward = execute_action(action_type, ref, text, action_txt)
    else:
        ref = search_item(by, 'email-sender', doms[-1])
        action_type, text = action_types[0], ''
        action_txt = 'Click on the thread by {} - {} {}'.format(by,action_type, ref)
        execute_action(action_type, ref, text, action_txt)
        
        ref = find_ref(doms[-1], task.capitalize())
        action_type, text = action_types[0], ''
        action_txt = 'Click on the {} icon - {} {}'.format(task.capitalize(), action_type, ref)
        execute_action(action_type, ref, text, action_txt)
        
        if task == 'forward':
            field = 'to'
            click_on = 'send-forward'
        else:
            field = 'message'
            click_on = 'send-reply'
        text = fields[field]

        instruction = 'Click on the input box next to the word {}'.format(field.capitalize())
        print(instruction)
        prompt = create_prompt(instruction, doms[-1])
        response = chat.send_message(prompt).text
        print(response)
        ref = int(re.search('Dom element ref number:[\* ]*([0-9]*)',response).group(1))
        action_type = action_types[0]
        action_txt = 'Click on the input box next to the word {} - {} {}'.format(field.capitalize(), action_type, ref)
        execute_action(action_type, ref, text, action_txt)
        
        instruction = 'Enter text {} in the input box next to the word {}'.format(text,field.capitalize())
        print(instruction)
        prompt = create_prompt(instruction, doms[-1])
        response = chat.send_message(prompt).text
        print(response)
        ref = int(re.search('Dom element ref number:[\* ]*([0-9]*)',response).group(1))
        action_type = action_types[1]
        action_txt = 'Type {} on the input box next to the word {} - {} {}'.format(text, field.capitalize(), action_type, ref)
        execute_action(action_type, ref, text, action_txt)

        ref = find_by_id(click_on, doms[-1])
        action_type = action_types[0]
        action_txt = 'Click on the {} icon - {} {}'.format(click_on, action_type, ref)
        reward = execute_action(action_type, ref, text, action_txt)

        
    print(reward)
    env.close()
    save_trajectory(SAVE_TO_FOLDER,actions, doms, images, times, env_type, reward, goal)
    return samples

def get_trajectory_click_checkboxes_soft():
    env_type = 'click-checkboxes-soft'
    action_types = ['CLICK_ELEMENT', 'TYPE_TEXT']
    actions, doms, images, times = [], [], [], []
    
    def execute_action(action_type, ref, text, action_txt):
        print(action_txt)
        action = env.create_action(action_type, ref=ref, text=text)
        observation, reward, terminated, truncated, info = env.step(action)
        dom = observation['dom_elements']
        image = observation['screenshot']
        actions.append(action_txt)
        doms.append(dom)
        images.append(image)
        times.append(int(time.time()-start_time))
        return reward
    
    env = MiniWoBEnvironment(env_type, wait_ms=500,render_mode=None)
    start_time = time.time()
    observation, info = env.reset()
    
    goal = observation['utterance']
    print(goal)
    n_clicks =len([x for x in observation['fields'] if 'target' in x[0]])
    
    action_txt  = ''
    dom = observation['dom_elements']
    image = observation['screenshot']

    actions.append(action_txt)
    doms.append(dom)
    images.append(image)
    times.append(int(time.time()-start_time))
    
    chat = model.start_chat(response_validation = False)
    
    prompt = create_prompt1(goal, doms[-1])
    response = chat.send_message(prompt).text
    print(response)
    
    ref = int(re.search('Dom element ref number:[\* ]*([0-9]*)',response).group(1))
    text = re.search('Word:[ ]*(.+)',response).group(1)
    action_type = action_types[0]
    action_txt = 'Click on checkbox next to the word {} - {} {}'.format(text, action_type, ref)
    reward = execute_action(action_type, ref, text, action_txt)
    
    while True:
        prompt = 'Keeping your goal in mind, what would be your next step? If no more steps, write "DONE"'
        response = chat.send_message(prompt).text
        print(response)
        
        if 'DONE' in response:
            break
        ref = int(re.search('Dom element ref number:[\* ]*([0-9]*)',response).group(1))
        text = re.search('Word:[ ]*(.+)',response).group(1)
        action_type = action_types[0]
        action_txt = 'Click on checkbox next to the word {} - {} {}'.format(text, action_type, ref)
        if 'Submit' in response:
            action_txt = 'Click on the Submit button - {} {}'.format(action_type, ref)
        reward = execute_action(action_type, ref, text, action_txt)
    
    print(reward)
    env.close()
    if reward > 0:
        save_trajectory(SAVE_TO_FOLDER,actions, doms, images, times, env_type, reward, goal)
    time.sleep(10)
    return reward

def get_trajectory_click_tab2_hard(samples):
    env_type = 'click-tab-2-hard'
    action_types = ['CLICK_ELEMENT', 'TYPE_TEXT']
    actions, doms, images, times = [], [], [], []
    
    def execute_action(action_type, ref, text, action_txt):
        action = env.create_action(action_type, ref=ref, text=text)
        observation, reward, terminated, truncated, info = env.step(action)
        dom = observation['dom_elements']
        image = observation['screenshot']
        actions.append(action_txt)
        doms.append(dom)
        images.append(image)
        times.append(int(time.time()-start_time))
        return reward
    
    env = MiniWoBEnvironment(env_type, wait_ms=500,render_mode='human')
    start_time = time.time()
    observation, info = env.reset()
    
    goal = observation['utterance']
    fields = dict(observation['fields'])
    target = fields['target']
    
    action_txt  = ''
    dom = observation['dom_elements']
    image = observation['screenshot']

    actions.append(action_txt)
    doms.append(dom)
    images.append(image)
    times.append(int(time.time()-start_time))
    
    ref1 = 0
    tab_no = 1
    while ref1 == 0:
        ref1 = search_item(target, 'alink', doms[-1])
        if ref1 > 0:
            break
        tab_no += 1
        action_type, text = action_types[0], str(tab_no)
        ref = search_item(text, 'ui-tabs-anchor', doms[-1])
        action_txt = 'Click on tab number {} because the link was not found in tab {} - {} {}'.format(
            tab_no, tab_no - 1, action_type, ref)
        execute_action(action_type, ref, text, action_txt)
    
    for t,tn in samples:
        if t == target and tn == tab_no:
            raise Exception('Repeated sample')
    samples.append((target, tab_no))
    
    action_type, text = action_types[0], target
    action_txt = 'Click on link {}  - {} {}'.format(target, action_type, ref1)
    reward = execute_action(action_type, ref1, text, action_txt)
    print(reward)
    env.close()
    save_trajectory(SAVE_TO_FOLDER,actions, doms, images, times, env_type, reward, goal)
    return samples

def get_trajectory_social_media(samples):
    env_type = 'social-media'
    action_types = ['CLICK_ELEMENT', 'TYPE_TEXT']
    actions, doms, images, times = [], [], [], []
    
    def execute_action(action_type, ref, text, action_txt):
        action = env.create_action(action_type, ref=ref, text=text)
        observation, reward, terminated, truncated, info = env.step(action)
        dom = observation['dom_elements']
        image = observation['screenshot']
        actions.append(action_txt)
        doms.append(dom)
        images.append(image)
        times.append(int(time.time()-start_time))
        return reward
    
    env = MiniWoBEnvironment(env_type, wait_ms=500,render_mode=None)
    start_time = time.time()
    observation, info = env.reset()
    
    goal = observation['utterance']
    fields = dict(observation['fields'])
    user = fields['user']
    button = fields['button']
    
    for u,b in samples:
        if u == user and b == button:
            raise Exception('Repeated sample')
    samples.append((user, button))
    
    action_txt  = ''
    dom = observation['dom_elements']
    image = observation['screenshot']

    actions.append(action_txt)
    doms.append(dom)
    images.append(image)
    times.append(int(time.time()-start_time))
    
    chat = model.start_chat(response_validation = False)
    
    if button in ['Like', 'Reply', 'Retweet']:
        prompt = create_prompt(goal, doms[-1])
        response = chat.send_message(prompt).text
        print(response)
        ref = int(re.search('Dom element ref number:[\* ]*([0-9]*)',response).group(1))
        action_type, text = action_types[0], ''
        action_txt = 'Look for the tweet by {} and click on the {} button - {} {}'.format(
            user, button, action_type, ref)
        reward = execute_action(action_type, ref, text, action_txt)
    else:
        instruction = 'For user {}, click on the MORE button (3 dots)'.format(user)
        prompt = create_prompt(instruction, doms[-1])
        response = chat.send_message(prompt).text
        print(response)
        ref = int(re.search('Dom element ref number:[\* ]*([0-9]*)',response).group(1))
        action_type, text = action_types[0], ''
        action_txt = 'Look for the tweet by {} and click on the MORE button (3 dots) - {} {}'.format(
            user, action_type, ref)
        reward = execute_action(action_type, ref, text, action_txt)

        prompt = create_prompt(goal, doms[-1])
        response = chat.send_message(prompt).text
        print(response)
        ref = int(re.search('Dom element ref number:[\* ]*([0-9]*)',response).group(1))
        action_type, text = action_types[0], ''
        action_txt = 'Select the {} option from the MORE dropdown menu - {} {}'.format(
            button, action_type, ref)
        reward = execute_action(action_type, ref, text, action_txt)

    print(reward)
    env.close()
    save_trajectory(SAVE_TO_FOLDER,actions, doms, images, times, env_type, reward, goal)
    return samples

def get_trajectory_social_media_some():
    env_type = 'social-media-some'
    action_types = ['CLICK_ELEMENT', 'TYPE_TEXT']
    actions, doms, images, times = [], [], [], []
    
    def execute_action(action_type, ref, text, action_txt):
        print(action_txt)
        action = env.create_action(action_type, ref=ref, text=text)
        observation, reward, terminated, truncated, info = env.step(action)
        dom = observation['dom_elements']
        image = observation['screenshot']
        actions.append(action_txt)
        doms.append(dom)
        images.append(image)
        times.append(int(time.time()-start_time))
        return reward
    
    env = MiniWoBEnvironment(env_type, wait_ms=500,render_mode=None)
    start_time = time.time()
    observation, info = env.reset()
    
    goal = observation['utterance']
    fields = dict(observation['fields'])
    order = ['first','second','third','fourth','fifth','sixth']
    user = fields['user']
    button = fields['button']
    amount = int(fields['amount'])

    action_txt  = ''
    dom = observation['dom_elements']
    image = observation['screenshot']

    actions.append(action_txt)
    doms.append(dom)
    images.append(image)
    times.append(int(time.time()-start_time))
    
    chat = model.start_chat(response_validation = False)
    
    instruction = 'Look for the first tweet by {} and click on the {} button'.format(user, button)
    prompt = create_prompt(instruction, doms[-1])
    response = chat.send_message(prompt).text
    print(response)
    ref = int(re.search('Dom element ref number:[\* ]*([0-9]*)',response).group(1))
    action_type, text = action_types[0], ''
    action_txt = '{} - {} {}'.format(instruction, action_type, ref)
    execute_action(action_type, ref, text, action_txt)
    
    for i in range(1,amount):
        prompt = 'Now that you have clicked on the {} tweet by {}, look for the {} tweet by {} and click on the {} button'.format(order[i-1], user, order[i],user,button)
        response = chat.send_message(prompt).text
        print(response)
        ref = int(re.search('Dom element ref number:[\* ]*([0-9]*)',response).group(1))
        action_type, text = action_types[0], ''
        action_txt = '{} - {} {}'.format(prompt, action_type, ref)
        execute_action(action_type, ref, text, action_txt)

    text = 'Submit'
    action_type, ref = action_types[0], find_ref(doms[-1], text)
    action_txt = 'Click on the {} button - {} {}'.format(text, action_type, ref)
    reward = execute_action(action_type, ref, text, action_txt)
    
    print(reward)
    env.close()
    save_trajectory(SAVE_TO_FOLDER,actions, doms, images, times, env_type, reward, goal)
    return reward

def get_trajectory_use_autocomplete(samples):   
    env_type = 'use-autocomplete'
    action_types = ['CLICK_ELEMENT', 'TYPE_TEXT']
    actions, doms, images, times = [], [], [], []
    
    def execute_action(action_type, ref, text, action_txt):
        action = env.create_action(action_type, ref=ref, text=text)
        observation, reward, terminated, truncated, info = env.step(action)
        dom = observation['dom_elements']
        image = observation['screenshot']
        actions.append(action_txt)
        doms.append(dom)
        images.append(image)
        times.append(int(time.time()-start_time))
        return reward
    
    env = MiniWoBEnvironment(env_type, wait_ms=500,render_mode=None)
    start_time = time.time()
    observation, info = env.reset()

    fields = dict(observation['fields'])
    goal = observation['utterance']

    start = fields['start']
    end = fields.get('end','')
    if end:
        end = 'and ends with {}'.format(end)
        
    for s,e in samples:
        if s == start and e == end:
            raise Exception('Repeated sample')
    samples.append((start, end))
    
    
    chat = model.start_chat(response_validation = False)
    
    action_txt  = ''
    dom = observation['dom_elements']
    image = observation['screenshot']
    actions.append(action_txt)
    doms.append(dom)
    images.append(image)
    times.append(int(time.time()-start_time))

    action_type, ref, text = action_types[0], 5, ''
    action_txt = 'Click on input field - {} {}'.format(action_type, ref)
    execute_action(action_type, ref, text, action_txt)
    
    action_type, ref, text = action_types[1], 5, start
    action_txt = 'Now that you have clicked on the input field, type {} in input field - {} {}'.format(text, action_type, ref)
    execute_action(action_type, ref, text, action_txt)
    
    instruction = 'Click on the autocomplete menu option that starts with {} {}'.format(start,end)
    prompt = create_prompt(instruction, doms[-1])
    response = chat.send_message(prompt).text
    print(response)
    ref = int(re.search('Dom element ref number:[\* ]*([0-9]*)',response).group(1))
    action_type, text = action_types[0], ''
    action_txt = '{} - {} {}'.format(instruction,action_type, ref)
    execute_action(action_type, ref, text, action_txt)
    
    text = 'Submit'
    action_type, ref = action_types[0], find_ref(doms[-1], text)
    action_txt = 'Click on the {} button - {} {}'.format(text, action_type, ref)
    reward = execute_action(action_type, ref, text, action_txt)
    
    print(reward)
    env.close()
    save_trajectory(SAVE_TO_FOLDER,actions, doms, images, times, env_type, reward, goal)
    return samples

def update_board(board, dom, refs):
    for i in range(3):
        for j in range(3):
            c = ref2classes(refs[i,j],dom)
            if c:
                board[i,j] = c[-1]
                refs[i,j] = 0
    return board

def next_move(board, refs):
    for i,row in enumerate(board):
        if Counter(row).get('x',0) == 2  and 'o' not in row:
            return refs[i, np.where(row == '')[0][0]]
    for i,row in enumerate(board.T):
        if Counter(row).get('x',0) == 2  and 'o' not in row:
            return refs[np.where(row == '')[0][0], i]
    d1 = np.array([board[0,0],board[1,1],board[2,2]])
    if Counter(d1).get('x',0) == 2  and 'o' not in d1:
        empty = np.where(d1 == '')[0][0]
        return refs[empty,empty]
    d2 = np.array([board[2,0],board[1,1],board[0,2]])
    if Counter(d2).get('x',0) == 2  and 'o' not in d2:
        empty = np.where(d1 == '')[0][0]
        return refs[2-empty,empty]
    
    for i,row in enumerate(board):
        if 'x' in row and '' in row and 'o' not in row:
            return refs[i, np.where(row == '')[0][0]]
    for i,row in enumerate(board.T):
        if 'x' in row and '' in row and 'o' not in row:
            return refs[np.where(row == '')[0][0], i]
    if 'x' in d1 and '' in d1 and 'o' not in d1:
        empty = np.where(d1 == '')[0][0]
        return refs[empty,empty]
    if 'x' in d2 and '' in d2 and 'o' not in d2:
        empty = np.where(d1 == '')[0][0]
        return refs[2-empty,empty]
    if '' in board:
        return refs[np.where(board == '')[0][0],np.where(board == '')[1][0]]
    return 0

def get_trajectory_tic_tac_toe():   
    env_type = 'tic-tac-toe'
    action_type = 'CLICK_ELEMENT'
    actions, doms, images, times = [], [], [], []
    
    def execute_action(action_type, ref, text, action_txt):
        action = env.create_action(action_type, ref=ref, text=text)
        observation, reward, terminated, truncated, info = env.step(action)
        dom = observation['dom_elements']
        image = observation['screenshot']
        actions.append(action_txt)
        doms.append(dom)
        images.append(image)
        times.append(int(time.time()-start_time))
        return reward, info['done']
    
    env = MiniWoBEnvironment(env_type, wait_ms=500,render_mode=None)
    start_time = time.time()
    observation, info = env.reset()

    goal = observation['utterance']
    board = np.array([['','',''],['','',''],['','','']])
    action_txt  = ''
    dom = observation['dom_elements']
    image = observation['screenshot']
    actions.append(action_txt)
    doms.append(dom)
    images.append(image)
    times.append(int(time.time()-start_time))
    
    refs = np.array([[5,6,7],[9,10,11],[13,14,15]])
    board = update_board(board, doms[-1], refs)
    ref = next_move(board, refs)
    action_txt = 'Click on any empty square - {} {}'.format(action_type, ref)

    done = False
    while ref != 0 and not done:
        reward, done = execute_action(action_type, ref, '', action_txt)
        board = update_board(board, doms[-1], refs)
        ref = next_move(board, refs)
        action_txt = '''Click on a square in a row, column or diagonal where there is an X and an empty space and no O
        or two Xs and no O - {} {}'''.format(action_type, ref)
    
    print(reward)
    env.close()
    save_trajectory(SAVE_TO_FOLDER,actions, doms, images, times, env_type, reward, goal)
    return reward

def get_trajectory_choose_date(dates):
    env_type = 'choose-date'
    action_type = 'CLICK_ELEMENT'
    actions, doms, images, times = [], [], [], []

    def append(action,dom,image):
        actions.append(action)
        doms.append(dom)
        images.append(image)
        times.append(int(time.time()-start_time))
    
    env = MiniWoBEnvironment(env_type, wait_ms=500,render_mode=None)
    start_time = time.time()
    observation, info = env.reset()

    goal = observation['utterance']
    print(goal)
    date = re.search('Select (.*) as the date', goal).group(1)
    date = datetime.strptime(date, '%m/%d/%Y')
    
    if date in dates:
        raise Exception('Date already in list')
    dates.append(date)
    prev_times = 12 - date.month
    
    action_txt  = ''
    dom = observation['dom_elements']
    image = observation['screenshot']
    append(action_txt, dom, image)
    
    action = env.create_action(action_type, ref=5)
    observation, reward, terminated, truncated, info = env.step(action)
    observation, reward, terminated, truncated, info = env.step(action)
    
    action_txt = 'Click on datepicker - {} 5'.format(action_type)
    dom = observation['dom_elements']
    image = observation['screenshot']
    append(action_txt, dom, image)
    
    for i in range(prev_times):
        if i==0:
            ref=10
        else:
            ref = find_ref(dom,'Prev')
        action = env.create_action(action_type, ref=ref)
        observation, reward, terminated, truncated, info = env.step(action)
        
        action_txt = 'You are currently in {}. Click on Prev to change month from {} to {}, because {}>{} - {} {}'.format(
            MONTHS[i],MONTHS[i], MONTHS[i+1],12-i,date.month, action_type, ref)
        dom = observation['dom_elements']
        image = observation['screenshot']
        append(action_txt, dom, image)
    
    day = str(date.day)
    ref = find_ref(dom, day)
    
    action = env.create_action(action_type, ref=ref)
    observation, reward, terminated, truncated, info = env.step(action)

    action_txt = 'Click on day {} of {} because you have already navigated to the right month ({}) - {} {}'.format(
        day, MONTHS[-date.month], MONTHS[-date.month],  action_type, ref)
    print(action_txt)
    dom = observation['dom_elements']
    image = observation['screenshot']
    append(action_txt, dom, image)
    
    ref = find_ref(dom, 'Submit')
    
    action = env.create_action(action_type, ref=ref)
    observation, reward, terminated, truncated, info = env.step(action)

    action_txt = 'Click on Submit - {} {}'.format(action_type, ref)
    dom = observation['dom_elements']
    image = observation['screenshot']
    append(action_txt, dom, image)
    env.close()
    print(reward)
    save_trajectory(SAVE_TO_FOLDER, actions, doms, images, times, env_type, reward, goal)
    return dates

def get_trajectory_book_flight(samples):   
    env_type = 'book-flight'
    action_types = ['CLICK_ELEMENT', 'TYPE_TEXT']
    actions, doms, images, times = [], [], [], []
    
    def execute_action(action_type, ref, text, action_txt):
        print(action_txt)
        action = env.create_action(action_type, ref=ref, text=text)
        observation, reward, terminated, truncated, info = env.step(action)
        dom = observation['dom_elements']
        image = observation['screenshot']
        actions.append(action_txt)
        doms.append(dom)
        images.append(image)
        times.append(int(time.time()-start_time))
        return reward
    
    env = MiniWoBEnvironment(env_type, wait_ms=500,render_mode=None)
    start_time = time.time()
    observation, info = env.reset()

    fields = dict(observation['fields'])
    
    goal = observation['utterance']
    print(goal)
    from_ = fields['from']
    to_ = fields['to']
    date = datetime.strptime(fields['date'], '%m/%d/%Y')
    criterion = fields['criterion']
    
    prev_times = 12 - date.month
    
    action_txt  = ''
    dom = observation['dom_elements']
    image = observation['screenshot']
    actions.append(action_txt)
    doms.append(dom)
    images.append(image)
    times.append(int(time.time()-start_time))
    
    action_type, ref, text = action_types[0], 7, ''
    action_txt = 'Click on the From input field before entering text - {} {}'.format(action_type, ref)
    execute_action(action_type, ref, text, action_txt)
    
    action_type, ref, text = action_types[1], 7, from_
    action_txt = 'Now that you have clicked on the From input field, type {} in the From input field - {} {}'.format(text, action_type, ref)
    execute_action(action_type, ref, text, action_txt)
    
    action_type, ref, text = action_types[0], 20, ''
    action_txt = 'Click on autocomplete option {} - {} {}'.format(ref2text(ref, doms[-1]),action_type, ref)
    execute_action(action_type, ref, text, action_txt)
    
    action_type, ref, text = action_types[0], 9, ''
    action_txt = 'Click on the To input field before entering text- {} {}'.format(action_type, ref)
    execute_action(action_type, ref, text, action_txt)
    
    action_type, ref, text = action_types[1], 9, to_
    action_txt = 'Now that you have clicked on the From input field, type {} in the To input field - {} {}'.format(text, action_type, ref)
    execute_action(action_type, ref, text, action_txt)
    
    action_type, ref, text = action_types[0], 24, ''
    action_txt = 'Click on autocomplete option {} - {} {}'.format(ref2text(ref, doms[-1]),action_type, ref)
    execute_action(action_type, ref, text, action_txt)
    
    action_type, ref, text = action_types[0], 13, ''
    action_txt = 'Click on the datepicker - {} {}'.format(action_type, ref)
    execute_action(action_type, ref, text, action_txt)
    
    for i in range(prev_times):
        action_type, text = action_types[0], 'Prev'
        ref = find_ref(doms[-1], text)
        action_txt = 'You are currently in {}. Click on Prev to change month from {} to {}, because {}>{} - {} {}'.format(
            MONTHS[i],MONTHS[i], MONTHS[i+1],12-i,date.month, action_type, ref)
        execute_action(action_type, ref, text, action_txt)
    
    action_type, text = action_types[0], str(date.day)
    ref = find_ref(doms[-1], text)
    action_txt = 'Click on day {} of {} because you have already navigated to the right month ({}) - {} {}'.format(
        text, MONTHS[-date.month], MONTHS[-date.month],  action_type, ref)
    execute_action(action_type, ref, text, action_txt)
    
    action_type, text = action_types[0], 'Search'
    ref = find_ref(doms[-1], text)
    action_txt = 'Click on {} - {} {}'.format(text, action_type, ref)
    execute_action(action_type, ref, text, action_txt)
    
    instruction = 'Find the {} flight'.format(criterion)
    print(instruction)
    prompt = create_prompt(instruction, doms[-1])
    
    chat = model.start_chat(response_validation = False)
    response = chat.send_message(prompt).text
    print(response)
    
    for f,t,d,c in samples:
        if f == from_ and t == to_ and d == date and c == criterion:
            raise Exception('Query and Rank already in list')
    samples.append((from_, to_, date, criterion))
    
    ref = int(re.search('Dom element ref number:[\* ]*([0-9]*)',response).group(1))
    action_type = action_types[0]
    text = ref2text(ref, doms[-1])
    action_txt = 'Click on button {} - {} {}'.format(text, action_type, ref)
    reward = execute_action(action_type, ref, text, action_txt)
    print(reward)
    env.close()
    save_trajectory(SAVE_TO_FOLDER, actions, doms, images, times, env_type, reward, goal)
    return samples


def create_prompt(goal,dom):

    part1 = '''You are a web-agent capable of reasoning to make decisions based on the information
    you are given. You are able to click on website buttons to select the best option based on the
    criterion provided. For example, you are able to compare durations, to select the shortest flight.

You are given the following goal: 
{}
You observe the following DOM elements from the web-page HTML: {}'''.format(goal,dom)
    part2 = '''
Start by comparing the different choices available and then find
the best answer. Provide your answer for the ONE next action in the following format:

Action description:
'''
    part3 = ''' Once you identified the next action to take, identify the ref number of the element
    on which you need to perform the action. Write your answer in the following format:
    
    Dom element ref number:
    
    \n'''
    
    return part1+part2+part3

def create_prompt1(goal,dom):

    part1 = '''You are a web-agent capable of reasoning to make decisions based on the information
    you are given. You are able to click on website buttons to select the best option based on the
    criterion provided. For example, you are able to compare durations, to select the shortest flight.

You are given the following goal: 
{}
You observe the following DOM elements from the web-page HTML: {}'''.format(goal,dom)
    part2 = '''
Start by comparing the different choices available and then find
the best answer. Provide your answer for the ONE next action in the following format:

Action description:
'''
    part3 = ''' Once you identified the next action to take, identify the ref number of the element
    on which you need to perform the action. Write your answer in the following format:
    
    Dom element ref number:
    Word:
    \n'''
    
    return part1+part2+part3