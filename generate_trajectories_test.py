import argparse

from vertexai.preview.generative_models import GenerativeModel
from vertexai.preview import tuning
from vertexai.preview.tuning import sft
from vertexai.preview.generative_models import SafetySetting,HarmCategory,HarmBlockThreshold
from vertexai.preview import generative_models

import dill
from collections import Counter
from environment import MiniWoBEnvironment
from utils import remove_nonessentials, remove_nonessentials_sms, ref2text
import json
import re
import time
import os

ENVS = ['click-checkboxes-soft',
 'click-tab-2-hard',
 'social-media',
 'email-inbox',
 'social-media-some',
 'tic-tac-toe',
 'use-autocomplete',
 'book-flight',
 'choose-date',
 'search-engine']

TEXT_FIELDS = {'click-checkboxes-soft':[],
 'click-tab-2-hard':[],
 'social-media':[],
 'email-inbox':['message','to'],
 'social-media-some':[],
 'tic-tac-toe':[],
 'use-autocomplete':['start'],
 'book-flight':['from','to'],
 'choose-date':[],
 'search-engine':['query']}

MODELS = {'click-tab-2-hard': 7945776118500425728
          ,'book-flight': 6449991703980933120
          ,'book-flight1': 5385290614340321280
          ,'choose-date': 3875634761124806656
          ,'choose-date1': 561302094728921088
          ,'choose-date3': 2480319321104973824
          ,'search-engine': 2449189947899379712
          ,'click-checkboxes-soft': 9151830423007920128
          ,'social-media-some1': 2354368065119977472
          ,'social-media-some2': 6279105606791987200
         , 'partial': 3725331521607827456
         ,'full': 501717360596484096
         ,'full1': 2650224653921943552}
 

SAFETY_SETTINGS = {
generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH
}

TRAINING_STATS = dill.load(open('training_stats.pkd','rb'))
SAVE_TO_FOLDER = 'trajectories_test'
MAX_SAMPLES = 51

MODEL_PATH = 'projects/829542692869/locations/us-central1/tuningJobs/{}'

def save_trajectory(folder, actions, doms, images, times, env_type, reward, utterance, model_dir_name):
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
    filename,number = name_file(env_type, folder,model_dir_name)
    if number > MAX_SAMPLES:
        print('Trajectory not saved. Already {} trajectories saved'.format(MAX_SAMPLES))
        return False
    dill.dump(traj, open(filename,'wb'))
    print('saved trajectory in file',filename)
    return True
    
def name_file(env_name, folder,model_dir_name):
    files = os.listdir(folder+'/'+model_dir_name)
    same_name = sorted([x for x in files if env_name in x])
    number = 0
    if same_name:
        number = int(same_name[-1][-7:-4])
        
    name = '{}/{}/{}{}.pkd'.format(folder,model_dir_name,env_name,str(number+1).zfill(3))
    return name,number

    
def create_prompt(goal,dom):
    dom = remove_nonessentials(dom)
    return "Goal: {}, DOM elements: {}".format(goal,dom).replace(
        ', dtype=float32)','').replace(
        ', dtype=int8)','').replace('array(','').replace("'","")

def test_env(env_type, training_prompts, tuned_model,model_dir_name):
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
        print('Done:',info['done'])
        return reward, info['done']
        
    env = MiniWoBEnvironment(env_type, wait_ms=500,render_mode=None)
    start_time = time.time()
    observation, info = env.reset()
    
    goal = observation['utterance']
    fields = dict(observation['fields'])
    print('Prompt:',goal)

    texts = []
    for key in TEXT_FIELDS[env_type]:
        if fields.get(key,False):
            texts.append(fields[key])
    text_index = 0

    actions.append('')
    doms.append(observation['dom_elements'])
    images.append(observation['screenshot'])
    times.append(int(time.time()-start_time))
    
    prompt = create_prompt(goal,doms[-1])
    done = False
    while not done:
        response = tuned_model.generate_content(prompt).text
        print('Response:',response)
        if 'CLICK_ELEMENT' in response:
            action_type = 'CLICK_ELEMENT'
            text = ''
        elif 'TYPE_TEXT' in response:
            action_type = 'TYPE_TEXT'
            text = texts[text_index]
            if text not in response:
                text = ''
        ref = int(re.search('{} ([0-9]+)'.format(action_type),response).group(1))
        reward, done = execute_action(action_type, ref, text, response)
        if 'TYPE_TEXT' in response:
            if ref2text(ref, doms[-1]) == text:
                text_index +=1
        prompt = create_prompt(goal,doms[-1])
    print('Reward:',reward)
    env.close()
    saved = save_trajectory(SAVE_TO_FOLDER, actions, doms, images, times, env_type, reward, goal,model_dir_name)
    return saved

def get_trajectories(model):
    if model == 'click-tab-2-hard':
        test_envs = ['click-tab-2-hard']
        model_dir_name = 'model_click-tab-2-hard'
    elif model == 'book-flight':
        test_envs = ['book-flight']
        model_dir_name = 'model_book-flight'
    elif model == 'book-flight1':
        test_envs = ['book-flight','choose-date']
        model_dir_name = 'model_book-flight1'
    elif model == 'choose-date':
        test_envs = ['choose-date']
        model_dir_name = 'model_choose-date'
    elif model == 'choose-date1':
        test_envs = ['choose-date']
        model_dir_name = 'model_choose-date1'
    elif model == 'choose-date3':
        test_envs = ['choose-date']
        model_dir_name = 'model_choose-date3'
    elif model == 'search-engine':
        test_envs = ['search-engine']
        model_dir_name = 'model_search-engine'
    elif model == 'click-checkboxes-soft':
        test_envs = ['click-checkboxes-soft']
        model_dir_name = 'model_click-checkboxes-soft'
    elif model == 'social-media-some1':
        test_envs = ['social-media-some']
        model_dir_name = 'model_social-media-some1'
    elif model == 'social-media-some2':
        test_envs = ['social-media-some']
        model_dir_name = 'model_social-media-some2'
    elif model == 'partial':
        test_envs = ENVS
        model_dir_name = 'model_partial'
    elif model == 'full':
        test_envs = ENVS
        model_dir_name = 'model_full'
    elif model == 'full1':
        test_envs = ENVS
        model_dir_name = 'model_full1'
    
    sft_tuning_job = sft.SupervisedTuningJob(MODEL_PATH.format(MODELS[model]))
    tuned_model = GenerativeModel(sft_tuning_job.tuned_model_endpoint_name, safety_settings = SAFETY_SETTINGS )

    for env_type in test_envs:
        print('------------------')
        print(env_type)
        print('------------------')
        if env_type == 'tic-tac-toe':#because tic-tac-toe has always the same prompt
            training_prompts = []
        else:
            training_prompts = TRAINING_STATS[TRAINING_STATS.env_name == env_type].goal.values
        saved = True ## it will save only a number of files specified by MAX_SAMPLES
        while saved:
            try:
                saved = test_env(env_type,training_prompts, tuned_model,model_dir_name)
            except Exception as error:
                print(error)
                continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="full1")
    args = parser.parse_args()

    
    get_trajectories(args.model)