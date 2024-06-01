import numpy as np
import math
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def plot_trajectory(images, actions, cols=4, filename = 'trajectory.png'):
    '''Plots screenshots of every observation in a trajectory
    input: images (list of np.arrays)
           actions (list of strings from parse_response 'action_text') 
           cols (int - number of columns for constructing the grid)
           filename (str with filename where to save the trajectory picture)'''
    
    rows = math.ceil(len(images) / cols)

    fig = plt.figure(figsize=(12*rows,8*cols))
    grid = ImageGrid(fig, 111, nrows_ncols = (rows,cols), axes_pad = 1)
    titles = ['{}.{}'.format(x[0]+1,x[1]) for x in enumerate(actions)]
    for i,image in enumerate(images):
        grid[i].axis('off')
        grid[i].set_title(titles[i])
        grid[i].imshow(image)
    plt.savefig(filename,bbox_inches='tight')

def compare_dom(ob1,ob2):
    '''finds differences in miniwob++ observation dom elements between observations
    input: ob1,ob2 = observation['dom_elements'] consecutive observations in ascending order
    output: differences'''
    if len(ob1) != len(ob2):
        ob1_refs = [x['ref'] for x in ob1]
        ob2_refs = [x['ref'] for x in ob2]
        print('refs found in 1:{} total={}',ob1_refs,len(ob1))
        print('refs found in 2:{} total={}',ob2_refs,len(ob2))
        if len(ob1) < len(ob2):
            changed = [x['ref'] for x in ob2 if x['ref'] not in ob1_refs]
            print('Added references',changed)
            longer = ob2
        else:
            changed = [x['ref'] for x in ob1 if x['ref'] not in ob2_refs]
            print('Removed references',changed)
            longer = ob1
        for item in longer:
            if item['ref'] in changed:
                pprint.pprint(item)
        return 'Mismatched length'
    for i in range(len(ob1)):
        key_diffs12 = [x for x in ob1[i].keys() if x not in ob2[i].keys()]
        key_diffs21 = [x for x in ob2[i].keys() if x not in ob1[i].keys()]
        if  key_diffs12 or  key_diffs21:
            return 'Added keys in ref {}'.format(i+1)
        for key,value in ob1[i].items():
            if type(value) in [int,str,float]:
                if value != ob2[i][key]:
                    print(i,value,ob2[i][key])
            elif type(value) == np.ndarray:
                if len(np.where(value != ob2[i][key])[0]) >0:
                    print('ref: {} key: {} value1: {} value2:{}'.format(i,key,value,ob2[i][key]))
            else:
                print(type(value))
                
def compare_screenshots(ob1,ob2):
    '''finds differences in miniwob++ observation screenshots between observations
    input: ob1,ob2 = observation['dom_elements']
    output: differences'''
    return np.where(ob1['screenshot'] != ob2['screenshot'])