from environment import MiniWoBEnvironment
import numpy as np
import time
import re
import dill
import os

from google.cloud import aiplatform
import vertexai.preview
from vertexai.preview.generative_models import GenerativeModel, Part, ChatSession

from PIL import Image
from google.cloud import storage

envs_bagel = {'click-checkboxes-soft':10,
 'click-tab-2-hard':20,
 'social-media':15,
 'email-inbox':30,
 'social-media-some':30,
 'tic-tac-toe':10,
 'use-autocomplete':10,
    'book-flight':30,
 'choose-date':20,
 'search-engine':20}


SAVE_TO_FOLDER = 'trajectories_gemini'

MAX_STEPS = 14

project_id = "cs224n-420704"
aiplatform.init(project=project_id)
vertexai.preview.init()
model = GenerativeModel(model_name="gemini-1.5-pro-preview-0514")
client = storage.Client(project=project_id)
bucket = client.get_bucket('bagel-ft')

def create_prompt1(goal,dom):

    inventory_str = '''
    1 - Click on *description* : This action will click on element that matches *description*.
    Examples: 
        - Click on the red button
        - Click on the first result in the autocomplete pop up menu
        - Click on a date in the calendar
        - Click on the scroll bar to scroll up and down
        - Click on a left arrow to go to a previous item, click on a right arrow to go to the next item
        - Click on the scroll bar to scroll up or down
        - Click on an input field before entering text
    The action code for the clicking action is CLICK_ELEMENT
    . 
    2 - Type *text* on *description*: This action will type *text* into the web element matching *description*.
    The action code for typing is TYPE_TEXT
'''
    element_rules = '''RULES FOR INTERACTING WITH DOM ELEMENTS:
    - Datepicker: If the month displayed is the desired month, select the desired day by clicking on 
    a number. If the month desired is before or after the month displayed, use the right and left arrows to
    navigate to the correct month.
    - Text input box: click on it before and after entering text, click on first autocomplete option after typing.
    - Autocomplete: select an option by clicking on it
    '''
    part1 = '''You are a web-agent capable of executing the following kinds of tasks on a webpage:
{}
You are given the following goal: 
{}
You observe the following image from the web-page HTML:'''
    part2 = '''
Start by thinking about what action you should take next. Write down all the different choices and then, choose
the best answer taking into account the following rules:
{}
Provide your answer for the ONE next action in the following format:

Action description:
'''
    part3 = ''' Once you identified next action to take, look at the DOM elements from the web-page HTML:
    {}
    and identify the ref number of the element on which you need to perform the action. Write your answer in the following format:
    
    Action code:(choose either TYPE_TEXT or CLICK_ELEMENT)
    Dom element ref number:
    Text: (only for TYPE_TEXT action code)
    \n'''.format(dom)
    part1 = part1.format(inventory_str,goal)
    part2 = part2.format(element_rules)
    
    return part1, part2, part3

def create_prompt2(actions,dom):
    prev_actions = '\n'.join(['{} - {}'.format(
        x[0]+1,x[1]) for x in enumerate(actions)]).replace('"','')

    part1 = '''After performing these actions:
    {},
    this is how image of the web-page looks like:'''.format(prev_actions)
    part2 = '''Start by thinking about what action you should take next. Write down all the different choices and then, choose
the best answer taking into account the previously given rules.
Provide your answer for the ONE next action in the following format:

Action description:
'''
    part3 = ''' Once you identified next action to take, look at the DOM elements from the web-page HTML:
    {}
    and identify the ref number of the element on which you need to perform the action. Write your answer in the following format:
    
    Action code:(choose either TYPE_TEXT or CLICK_ELEMENT)
    Dom element ref number:
    Text: (only for TYPE_TEXT action code)
    Done: (True if no more actions needed, otherwise leave blank)
    \n'''.format(dom)
    return part1, part2, part3

def parse_response(response):
    '''Converts llm response to dictionary for use in creating miniwob actions
    Assumes the response has a section with the following format:
    
    Action description:\n
    Action code:\n
    Dom element ref number:\n
    Text: (not required for CLICK_ELEMENT)\n
    '''
    response = response.replace('*','').replace('"','')
    action_codes = ['CLICK_ELEMENT','TYPE_TEXT']
    possible_click_codes =['CLICK','Click','click']
    possible_type_codes =['TYPE','Type','type','write','Write','WRITE','type text']
    action = re.search('Action code:(.*)\n',response).group(1)
    for code in action_codes:
        if code in action:
            action = code
            break
    if action not in action_codes:
        if action in possible_click_codes:
            action = 'CLICK_ELEMENT'
        else:
            action = 'TYPE_TEXT'
    ref = int(re.search('Dom element ref number:[\* ]*([0-9]*)',response).group(1))
    action_text = re.search('Action description:(.*)\n',response).group(1).strip()
    action_text = '{} - {} {}'.format(action_text,action,ref)
    if 'Text:' in response:
        text = re.search('Text:(.*)\n',response).group(1).strip()
        
    else:
        text = ''
    return {'action':action, 'ref':ref, 'text':text, 'action_text':action_text}

def assemble_prompt(goal, actions, dom, image):
    if actions:
        part1,part2,part3 = create_prompt2(actions, dom)
    else:
        part1,part2,part3 = create_prompt1(goal, dom)
    convert_to_image(image)
    return [Part.from_text(part1)
            , Part.from_uri("gs://bagel/image.png",mime_type="image/png")
            , Part.from_text(part2)
            , Part.from_text(part3)]

def convert_to_image(nparray):
    filename = 'image.png'
    image = Image.fromarray(nparray)
    image.save(filename)
    source_file_name = filename
    blob = bucket.blob(filename)
    blob.upload_from_filename(filename)
    
def validate_response(text):
    items_to_find = ['Action code','Dom element ref number:']
    for item in items_to_find:
        if item not in text:
            return False
    return True

def save_trajectory(actions, doms, images, times, env_type, reward, utterance=''):
    traj = {}
    traj['utterance'] = utterance
    traj['reward'] = reward
    traj['states'] = []
    for i in range(len(actions)):
        state = {}
        state['time'] = times[i]
        state['action'] = actions[i]
        state['dom_elements'] = doms[i]
        state['screenshot'] = images[i]
        traj['states'].append(state)
        filename = name_file(env_type)
    dill.dump(traj, open(filename,'wb'))
    print('saved trajectory in file',filename)
 
    return traj

def name_file(env_name):
    files = os.listdir(SAVE_TO_FOLDER+'/'+env_name)
    same_name = sorted([x for x in files if env_name in x])
    if same_name:
        number = int(same_name[-1][-6:-4])
        name = '{}/{}/{}{}.pkd'.format(SAVE_TO_FOLDER,env_name,env_name,str(number+1).zfill(2))
    else:
        name = '{}/{}/{}01.pkd'.format(SAVE_TO_FOLDER,env_name,env_name)
    return name

def perform_tasks(env_type,env_time):
    env = MiniWoBEnvironment(env_type,render_mode=None)
    observation, info = env.reset()
    goal = observation['utterance']
    actions, doms, images, times = [], [], [], []
    add_click = False
    count = 0
    start_time = time.time()
    previous_action = ''
    previous_ref = 0
    reward = None
    response = ''
    d_response = {'action':'', 'ref':0, 'text':'', 'action_text':''}
    
    actions.append('')
    images.append(observation['screenshot'])
    times.append(0)
    doms.append(observation['dom_elements'])

    def perform_action(action_type, ref, text=''):
        action = env.create_action(action_type, ref=ref, text=text)
        observation, reward, terminated, truncated, info = env.step(action)
        images.append(observation['screenshot'])
        doms.append(observation['dom_elements'])
        return reward
            
    def init_step(count,times):
        count += 1
        times.append(int(time.time()-start_time))
        print('-----------------------------------------------------------')
        print('Step {}'.format(count))
        print(goal)
        print('TRAJECTORY')
        print('\n'.join(['{} - {}'.format(x[0]+1,x[1]) for x in enumerate(actions)]))
        return count,times
    
    def check_response(response,reminder):
        valid = validate_response(response)
        
        while not valid:
            print(reminder)
            response = chat.send_message(reminder).text
            valid = validate_response(response)
        return response
        
    chat = model.start_chat()
    while not info['done']:
        
        if count > MAX_STEPS or time.time()-start_time >= env_time or 'Done: True' in response:
            break
        count,times = init_step(count,times)
        
        prompt = assemble_prompt(goal, actions,observation['dom_elements'],observation['screenshot'])
        response = chat.send_message(prompt).text

        print('RESPONSE:')
        print(response)
        
        
        response = check_response(response, '''Please make sure your answer is in the following format:
            
            Action description:
            Action code:(choose either TYPE_TEXT or CLICK_ELEMENT)
            Dom element ref number:
            Text: (only for TYPE_TEXT action code)
            \n
            ''')
        
        previous_action = d_response['action']
        previous_ref = d_response['ref']

        if previous_action == 'TYPE_TEXT':
            if d_response['action'] != 'CLICK_ELEMENT':#and previous_ref == d_response['ref']):
                reminder = '''Don't forget to click 
            on the text field after you enter text, to get autocomplete options.'''
                print(reminder)
                response = chat.send_message(reminder).text
                response = check_response(response, reminder)
                d_response = parse_response(response)
        

        d_response = parse_response(response)
        print(d_response)
        
        if d_response['action'] == 'TYPE_TEXT':
            if previous_action != 'CLICK_ELEMENT':#and previous_ref == d_response['ref']):
                reminder = '''Don't forget to click on the text field before and after you enter text.'''
                print(reminder)
                response = chat.send_message(reminder).text
                print(response)
                response = check_response(response, reminder)
                d_response = parse_response(response)
        
        reward = perform_action(d_response['action'], d_response['ref'], d_response['text'].replace('"',''))   
        actions.append(d_response['action_text']) 
        print('reward',reward)
        if len(actions) % 4 ==0:
            time.sleep(10)
    env.close()

    return save_trajectory(actions, doms, images, times, env_type, reward, goal)
    
for task,duration in envs_bagel.items():
    print(task)
    count = 0
    while count < 60:
        print('Trajectory',count)
        try:
            perform_tasks(task,duration)
            count +=1
        except Exception as error:
            print(error)
            print('Retrying')
        time.sleep(10)
                  
                  
        
