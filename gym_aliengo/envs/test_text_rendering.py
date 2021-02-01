''' 
Script created to rendering speed when adding user debug text.
'''
import pybullet as p
import pybullet_data
import os
import time
import numpy as np

path = os.path.abspath(os.path.dirname(pybullet_data.__file__))
client = p.connect(p.GUI)
robot = p.loadSDF(os.path.join(path, 'kuka_iiwa/kuka_with_gripper.sdf'), physicsClientId=client)[0]

txt = p.addUserDebugText('{:.1f}'.format(2.246843), 
                textPosition=[0,0,0], 
                textColorRGB=[0]*3,
                physicsClientId=client)

# line = p.addUserDebugLine([0,0,0], [0,0,1], physicsClientId=client)

iters = 100000
non_text_render_avg = 0.0
count = 0.0
x = np.random.randn(3)
for _ in range(iters): # increase number of iterations to show more memory leaking
    begin = time.time()
    p.stepSimulation(physicsClientId=client)
    end = time.time()
    elapsed = end - begin
    count += 1.0
    non_text_render_avg += (elapsed - non_text_render_avg)/count


    # test print to screen
    info = p.getDebugVisualizerCamera()
    view_matrix = np.array(info[2]).reshape(4, 4).T
    _inv_camera_tsf = np.linalg.inv(view_matrix)
    pos_in = np.r_[[0,0,0], 1]
    world_frame_pos = _inv_camera_tsf @ pos_in
    txt = p.addUserDebugText('asdf', world_frame_pos[:3], physicsClientId=client, replaceItemUniqueId=txt,
                            textColorRGB=[255]*3, textSize=10)
    # txt = p.addUserDebugText('asdf', [,0,0], physicsClientId=client, replaceItemUniqueId=txt, textColorRGB=[255]*3)
    time.sleep(1/240.)




text_render_avg = 0.0
count = 0.0
for _ in range(iters): # increase number of iterations to show more memory leaking
    begin = time.time()
    p.stepSimulation(physicsClientId=client)
    p.addUserDebugText('{}'.format(2.246843), 
                        textPosition=[0,0,0], 
                        replaceItemUniqueId=txt,
                        textColorRGB=[0]*3,
                        physicsClientId=client)
    # p.addUserDebugLine([0,0,0], [0,0,1], physicsClientId=client, replaceItemUniqueId=line)
    end = time.time()
    elapsed = end - begin
    count += 1.0
    text_render_avg += (elapsed - text_render_avg)/count




print('\n' + '#' * 100)
print('Average render time per sim step WITHOUT text: {} s'
            '\nAverage render time per sim step WITH text: {} s'.format(non_text_render_avg, text_render_avg))
diff = text_render_avg - non_text_render_avg
print('Increase in render time with text: {} s'
            '\n{:.0f}x Increase'.format(diff, diff/non_text_render_avg))
print( '#' * 100 + '\n')
