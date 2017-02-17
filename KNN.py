import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm

fig = plt.figure()
plt.gray()


def nearest_k(uv, k, points,lbl):
    v = points - uv
    v = v* v
    v = v.dot(np.array([1,1]))
    v = v.argsort()[:k]
    lbls = lbl[v]
    #print ("uv={} lbl={}".format(uv,lbls))
    return np.mean(lbls)


def KNN(w,h, k,points,lbl):
    ret = np.ndarray((w,h))
    for x in range(w):
        for y in range(h):
            uv = (y/float(h)-0.5,x/float(w)-0.5)
            m = nearest_k(uv,k,points,lbl)
            ret[x,y] = 0.5
    return ret








training_set = np.array(((1,0),(-.5,0)))
                   #(np.random.rand(20,2) - 0.5)*2
training_label = np.array((0,1)) #np.random.randint(0,2,len(training_set))
colors = list(map(lambda l: cm.rainbow(l / 2.0), training_label))
'''
print (training_set)
print (training_label)
print (colors)
'''

sct = plt.scatter(training_set[:,0], training_set[:,1],color = colors)

im = plt.imshow(KNN(256,256,1,training_set,training_label), animated=True , extent=[-1,1,-1,1], cmap="gray")



def updatefig(*args):
    global x, y
    #im.set_array(KNN(10,10,3))
    return im,sct

#ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
plt.show()





