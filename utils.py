import numpy as np
import os
import re
import dill
    
def prompt_together(model, api_key, prompt):
    '''input: prompt (str)
       output: str '''
    from together import Together
    client = Together(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

def find_ref(dom, s):
    for element in dom:
        if element['text'] == s:
            return element['ref']
        
def ref2text(ref, dom):
    for element in dom:
        if element['ref'] == ref:
            text = element['text']
            return text
        
def get_page_link(page_no, dom):
    text = str(page_no)
    for element in dom:
        if element['classes'] =='page-link' and element['text'] == text:
            return element['ref']
        
def search_page(text, dom):
    for element in dom:
        if element['classes'] =='search-title' and element['text'] == text:
            return element['ref']
    return 0

        
def save_trajectory(folder, actions, doms, images, times, env_type, reward, utterance=''):
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
        filename = name_file(env_type, folder)
    dill.dump(traj, open(filename,'wb'))
    print('saved trajectory in file',filename)
    return traj

def name_file(env_name, folder):
    files = os.listdir(folder+'/'+env_name)
    same_name = sorted([x for x in files if env_name in x])
    if same_name:
        number = int(same_name[-1][-7:-4])
        name = '{}/{}/{}{}.pkd'.format(folder,env_name,env_name,str(number+1).zfill(3))
    else:
        name = '{}/{}/{}001.pkd'.format(folder,env_name,env_name)
    return name

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

