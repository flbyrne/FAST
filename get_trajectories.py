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
from environment import MiniWoBEnvironment
from utils import find_ref, save_trajectory, name_file, parse_response, prompt_together, ref2text, get_page_link, search_page,search_item

### NEEDED FOR GEMINI PROMPTS#########
from google.cloud import aiplatform
import vertexai.preview
from vertexai.preview.generative_models import GenerativeModel, Part, ChatSession

from PIL import Image
from google.cloud import storage

BUCKET = 'miniwobimages'
project_id = "cs224n-420704"
aiplatform.init(project=project_id)
vertexai.preview.init()
model = GenerativeModel(model_name="gemini-1.5-pro-preview-0514")
client = storage.Client(project=project_id)
bucket = client.get_bucket(BUCKET)
######################################


### NEEDED FOR LLAMA PROMPTS##########
with open('config.yaml') as f: config = yaml.safe_load(f)
API_KEY = config['api_key']
MODEL="meta-llama/Llama-3-70b-chat-hf"
######################################

SAVE_TO_FOLDER = 'trajectories_train'
MONTHS = ['December', 'November', 'October'
          , 'September', 'August', 'July', 'June', 'May', 'April', 'March', 'February', 'January']

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
        
        action_txt = 'Click on Prev to get to the month of {}- {} {}'.format(MONTHS[i+1], action_type, ref)
        dom = observation['dom_elements']
        image = observation['screenshot']
        append(action_txt, dom, image)
    
    day = str(date.day)
    ref = find_ref(dom, day)
    
    action = env.create_action(action_type, ref=ref)
    observation, reward, terminated, truncated, info = env.step(action)

    action_txt = 'Click on day {} - {} {}'.format(day, action_type, ref)
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
    save_trajectory(SAVE_TO_FOLDER, actions, doms, images, times, env_type, reward, goal)
    return dates

def get_trajectory_book_flight(samples):   
    env_type = 'book-flight'
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
    action_txt = 'Click on From input field - {} {}'.format(action_type, ref)
    execute_action(action_type, ref, text, action_txt)
    
    action_type, ref, text = action_types[1], 7, from_
    action_txt = 'Type {} in From input field - {} {}'.format(text, action_type, ref)
    execute_action(action_type, ref, text, action_txt)
    
    action_type, ref, text = action_types[0], 20, ''
    action_txt = 'Click on autocomplete option {} - {} {}'.format(ref2text(ref, doms[-1]),action_type, ref)
    execute_action(action_type, ref, text, action_txt)
    
    action_type, ref, text = action_types[0], 9, ''
    action_txt = 'Click on To input field - {} {}'.format(action_type, ref)
    execute_action(action_type, ref, text, action_txt)
    
    action_type, ref, text = action_types[1], 9, to_
    action_txt = 'Type {} in To input field - {} {}'.format(text, action_type, ref)
    execute_action(action_type, ref, text, action_txt)
    
    action_type, ref, text = action_types[0], 24, ''
    action_txt = 'Click on autocomplete option {} - {} {}'.format(ref2text(ref, doms[-1]),action_type, ref)
    execute_action(action_type, ref, text, action_txt)
    
    action_type, ref, text = action_types[0], 13, ''
    action_txt = 'Click on datepicker - {} {}'.format(action_type, ref)
    execute_action(action_type, ref, text, action_txt)
    
    for i in range(prev_times):
        action_type, text = action_types[0], 'Prev'
        ref = find_ref(doms[-1], text)
        action_txt = 'Click on Prev to get to the month of {}- {} {}'.format(MONTHS[i+1], action_type, ref)
        execute_action(action_type, ref, text, action_txt)
    
    action_type, text = action_types[0], str(date.day)
    ref = find_ref(doms[-1], text)
    action_txt = 'Click on day {} - {} {}'.format(text, action_type, ref)
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

def get_trajectory_search_engine(queries):   
    env_type = 'search-engine'
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
    query = fields['query']
    rank = int(fields['rank'])
    for q,r in queries:
        if q == query and r == rank:
            raise Exception('Query and Rank already in list')
    queries.append((query,rank))
    
    
    #page = ceil(rank/3) # will not be used to generalize to other environments
    item_not_found = True
    
    action_txt  = ''
    dom = observation['dom_elements']
    image = observation['screenshot']

    actions.append(action_txt)
    doms.append(dom)
    images.append(image)
    times.append(int(time.time()-start_time))
    
    action_type, ref, text = action_types[0], 5, ''
    action_txt = 'Click on textbox - {} {}'.format(action_type, ref)
    execute_action(action_type, ref, text, action_txt)
    
    action_type, ref, text = action_types[1], 5, query
    action_txt = 'Type {} in textbox - {} {}'.format(text, action_type, ref)
    execute_action(action_type, ref, text, action_txt)
    
    action_type, ref, text = action_types[0], 6, ''
    action_txt = 'Click on the Search button - {} {}'.format(action_type, ref)
    execute_action(action_type, ref, text, action_txt)

    page = 1
    while item_not_found:
        ref = search_page(query, doms[-1])
        if ref > 0:
            item_not_found = False
        else:
            page +=1
            ref = get_page_link(page, doms[-1])
            action_type, text = action_types[0], ''
            action_txt = 'Go to page {} because {} was not found on page {} - {} {}'.format(
                page, query, page-1, action_type, ref)
            execute_action(action_type, ref, text, action_txt)
        
    action_type, text = action_types[0], ''
    action_txt = 'Click on Search Title {} - {} {}'.format(query,action_type, ref)
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
        else:
            field = 'message'
        text = fields[field]
            
        instruction = 'Enter text {} in the {} field'.format(text,field.capitalize())
        print(instruction)
        prompt = create_prompt(instruction, doms[-1])
        response = chat.send_message(prompt).text
        print(response)
        ref = int(re.search('Dom element ref number:[\* ]*([0-9]*)',response).group(1))
        action_type = action_types[1]
        action_txt = 'Type {} on {} field - {} {}'.format(text, field.capitalize(), action_type, ref)
        reward = execute_action(action_type, ref, text, action_txt)
        action_type, text = action_types[0], ''
        
    print(reward)
    env.close()
    save_trajectory(SAVE_TO_FOLDER,actions, doms, images, times, env_type, reward, goal)
    return samples

def get_trajectory_click_checkboxes_soft():
    env_type = 'click-checkboxes-soft'
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
    
    prompt = create_prompt(goal, doms[-1])
    response = chat.send_message(prompt).text
    print(response)
    
    ref = int(re.search('Dom element ref number:[\* ]*([0-9]*)',response).group(1))
    text = ref2text(ref, doms[-1])
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
        text = ref2text(ref, doms[-1])
        action_type = action_types[0]
        action_txt = 'Click on {} box - {} {}'.format(text, action_type, ref)
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
    amount = int(fields['amount'])

    action_txt  = ''
    dom = observation['dom_elements']
    image = observation['screenshot']

    actions.append(action_txt)
    doms.append(dom)
    images.append(image)
    times.append(int(time.time()-start_time))
    
    chat = model.start_chat(response_validation = False)
    
    instruction = 'Look for a tweet by {} and click on the {} button'.format(user, button)
    prompt = create_prompt(instruction, doms[-1])
    response = chat.send_message(prompt).text
    print(response)
    ref = int(re.search('Dom element ref number:[\* ]*([0-9]*)',response).group(1))
    action_type, text = action_types[0], ''
    action_txt = '{} - {} {}'.format(instruction, action_type, ref)
    execute_action(action_type, ref, text, action_txt)
    
    for _ in range(amount-1):
        prompt = 'Look for another tweet by {} and click on the {} button'.format(user,button)
        response = chat.send_message(prompt).text
        print(response)
        ref = int(re.search('Dom element ref number:[\* ]*([0-9]*)',response).group(1))
        action_type, text = action_types[0], ''
        action_txt = '{} - {} {}'.format(instruction, action_type, ref)
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
    action_txt = 'Type {} in From input field - {} {}'.format(text, action_type, ref)
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