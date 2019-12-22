# -*- coding: utf-8 -*-
#/usr/bin/python3

from __future__ import print_function

#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')
import commands
import sys
import matplotlib.pyplot as plt

def plot(name, l1, l2, l):
    x = range(1, l+1)
    l1 = l1[:l]
    l2 = l2[:l]
    print(len(x),len(l1))
    plt.plot(x, l1, 's-', color = 'r', label="BL")
    plt.plot(x, l2, 'o-',color = 'b',label="IL")
    plt.xlabel("position")
    plt.ylabel(name)
    plt.legend(loc = "best")
    plt.show()
    plt.close()

def load_data(fpath1, fpath2,l):
    fn = lambda f: [line.split() for line in open(f, 'r').read().split('\n')]
    content1 = fn(fpath1)
    content2 = fn(fpath2)
    assert len(content1) == len(content2)
    corr = [0] * l
    total = [0] * l
    t1 = 0
    c1 = 0
    t2 = 0
    c2 = 0
    t = 0
    c = 0
    
    for i in range(len(content1)):
        if len(content1[i]) > l or len(content2[i]) > l:
            continue
        for j in range(len(content1[i])):
            if j < len(content2[i]):
                total[j] += 1
                t += 1
                if j < 11:
                    t1 += 1
                elif j > 40:
                    t2 += 1
                if content1[i][j] == content2[i][j]:
                    corr[j] += 1
                    c += 1
                    if j < 11:
                        c1 += 1
                    elif j > 40:
                        c2 += 1
    for i in range(l):
        if total[i] == 0:
            corr[i] = 0
        else:
            corr[i] = corr[i]/(total[i]*1.0)
    return corr, c/(t*1.0),c1/(t1*1.0),c2/(t2*1.0)

def select(f1,f2,f):
    fn = lambda f: [line.split() for line in open(f, 'r').read().split('\n')]
    fn1 = lambda f: [line for line in open(f, 'r').read().split('\n')]
    s0 = fn1("source.txt")
    s1 = fn1(f)
    print
    s2 = fn1(f1)
    s3 = fn1(f2)
    content1 = fn(f1)
    content2 = fn(f2)
    s = fn(f)
    for i in range(len(s)):
        a1 = 2
        a2 = 5
        if len(content1[i])> 2 and  len(content2[i])< 5:
            print("____________________________________________________________")
            print(len(s[i]))
            print(s0[i])
            print(s1[i])
            print(s2[i])
            print(s3[i])           


                
    

def test():
    fpath1 = "c.txt"
    f1 = "a.txt"
    f2 = "b.txt"
    #select(f1,f2,fpath1)
    l = 100000

    a,at,aF10,aL10 = load_data(fpath1,"baseline.txt",l)
    b,bt,bF10,bL10 = load_data(fpath1,"re.txt",l)
    print('%.4f' % at, '%.4f' % bt)
    print('%.3f' % aF10, '%.3f' % bF10)
    print('%.3f' % aL10, '%.3f' % bL10)
    print("__________")
    for k in range(7):
        print('%.3f' % a[k], '%.3f' % b[k])
    
    plot('accuracy ', a, b,10)


if __name__ == '__main__':
    test()
