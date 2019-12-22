# -*- coding: utf-8 -*-
#/usr/bin/python3

from __future__ import print_function
import matplotlib.pyplot as plt

def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    '''Noam scheme learning rate decay
    init_lr: initial learning rate. scalar.
    global_step: scalar.
    warmup_steps: scalar. During warmup_steps, learning rate increases
        until it reaches init_lr.
    '''
    step = global_step + 1
    return init_lr * warmup_steps ** -0.5 * min(step * warmup_steps ** -1.5, step ** -0.5)

x = []
y = []
for i in range(323291):
    lr = noam_scheme(init_lr = 0.0003, global_step = i, warmup_steps=16000.)
    x.append(i)    
    y.append(lr)
    
plt.plot(x,y)
plt.show()
    