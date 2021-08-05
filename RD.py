#RD system

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.linalg import eig
import random
from scipy.stats import norm, uniform, multivariate_normal



par={
    'K_ahl_green':100.0,
    'n_ahl_green':2.0,
    'alpha_green':0.00,
    'beta_green':100,
    'K_GREEN':1,
    'n_GREEN':2.0,
    'delta_green':1,

    'K_ahl_red':100.0,
    'n_ahl_red':2.0,
    'alpha_red':0,
    'beta_red':100,
    'K_RED':1,
    'n_RED':2.0,
    'delta_red':1,

    'K_ahl':10,#100,
    'n_ahl':2.0,
    'beta_ahl':100.,
    'delta_ahl':1,#1.,
    
    'D_ahl':1.
    }



parlist = [ 
    #GREEN
    {'name' : 'K_ahl_green', 'lower_limit':0.0,'upper_limit':100.0},
    {'name' : 'n_ahl_green', 'lower_limit':0.5,'upper_limit':2.0},
    {'name' : 'alpha_green', 'lower_limit':0.0,'upper_limit':50.0},
    {'name' : 'beta_green', 'lower_limit':0.0,'upper_limit':100.0},
    {'name' : 'delta_green', 'lower_limit':1.0,'upper_limit':1.0},
    {'name' : 'K_GREEN', 'lower_limit':0.0,'upper_limit':100.0},
    {'name' : 'n_GREEN', 'lower_limit':0.5,'upper_limit':2.0},

    #RED
    {'name' : 'K_ahl_red', 'lower_limit':0.0,'upper_limit':100.0},
    {'name' : 'n_ahl_red', 'lower_limit':0.5,'upper_limit':2.0},
    {'name' : 'alpha_red', 'lower_limit':0.0,'upper_limit':50.0},
    {'name' : 'beta_red', 'lower_limit':0.0,'upper_limit':100.0},
    {'name' : 'delta_red', 'lower_limit':1.0,'upper_limit':1.0},
    {'name' : 'K_RED', 'lower_limit':0.0,'upper_limit':100.0},
    {'name' : 'n_RED', 'lower_limit':0.5,'upper_limit':2.0},

    #AHL
    {'name' : 'K_ahl', 'lower_limit':0.0,'upper_limit':100.0},
    {'name' : 'n_ahl', 'lower_limit':0.5,'upper_limit':2.0},
    {'name' : 'beta_ahl', 'lower_limit':0.0,'upper_limit':100.0},
    {'name' : 'delta_ahl', 'lower_limit':1.0,'upper_limit':1.0},
    {'name' : 'D_ahl', 'lower_limit':0.01,'upper_limit':1.0},
]





maxD= 1.1
d = 0.1 #interval size
d2 = d*d
dt = d2 * d2 / (2 * maxD * (d2 + d2)) #interval time according to diffusion and interval dist


def pars_to_dict(pars,parlist):
### This function is not necessary, but it makes the code a bit easier to read,
### it transforms an array of pars e.g. p[0],p[1],p[2] into a
### named dictionary e.g. p['k0'],p['B'],p['n'],p['x0']
### so it is easier to follow the parameters in the code
    dict_pars = {}
    for ipar,par in enumerate(parlist):
        dict_pars[par['name']] = pars[ipar] 
    return dict_pars


def diffusion(u0,d,D,oneD=False):
    d2 = d*d
    dt = d2 * d2 / (2 * 1.0 * (d2 + d2))
    u = u0.copy()
    uxx = (u0[2:, 1:-1] - 2*u0[1:-1, 1:-1] + u0[:-2, 1:-1])/d2
    uyy = (u0[1:-1, 2:] - 2*u0[1:-1, 1:-1] + u0[1:-1, :-2])/d2
    if oneD:
        uxx= uxx*0
    u[1:-1, 1:-1] =  D*(uxx + uyy)

    return u


def model(GREENi,REDi,AHLi,d,density,par, isdiffusion=True,oneD =False):
    if isdiffusion:
        GREEN = par['alpha_green']+((par['beta_green']-par['alpha_green'])*np.power(AHLi*par['K_ahl_green'],par['n_ahl_green']))/(1+np.power(AHLi*par['K_ahl_green'],par['n_ahl_green']))
        GREEN = GREEN / (1 + np.power(REDi*par['K_RED'],par['n_RED']))
        GREEN = GREEN - par['delta_green']*GREENi
                                                                 
        RED = par['alpha_red']+((par['beta_red']-par['alpha_red'])*np.power(AHLi*par['K_ahl_red'],par['n_ahl_red']))/(1+np.power(AHLi*par['K_ahl_red'],par['n_ahl_red']))
        RED = RED / (1 + np.power(GREENi*par['K_GREEN'],par['n_GREEN']))
        RED = RED - par['delta_red']*REDi

        AHL = (par['beta_ahl']*np.power(GREENi*par['K_ahl'],par['n_ahl']))/(1+np.power(GREENi*par['K_ahl'],par['n_ahl']))
        AHLdif = diffusion(AHLi,d,par['D_ahl'],oneD)
        AHL = AHL*density + AHLdif - par['delta_ahl']*(AHLi)
        
    if isdiffusion == False:
        GREEN = par['alpha_green']+((par['beta_green']-par['alpha_green'])*np.power(AHLi*par['K_ahl_green'],par['n_ahl_green']))/(1+np.power(AHLi*par['K_ahl_green'],par['n_ahl_green']))
        GREEN = GREEN / (1 + np.power(REDi*par['K_RED'],par['n_RED']))
        GREEN = GREEN - par['delta_green']*GREENi
                                                                 
        RED = par['alpha_red']+((par['beta_red']-par['alpha_red'])*np.power(AHLi*par['K_ahl_red'],par['n_ahl_red']))/(1+np.power(AHLi*par['K_ahl_red'],par['n_ahl_red']))
        RED = RED / (1 + np.power(GREENi*par['K_GREEN'],par['n_GREEN']))
        RED = RED - par['delta_red']*REDi

        AHL = (par['beta_ahl']*np.power(GREENi*par['K_ahl'],par['n_ahl']))/(1+np.power(GREENi*par['K_ahl'],par['n_ahl']))
        AHL = AHL*density  - par['delta_ahl']*(AHLi)

    return GREEN,RED, AHL


def Integration(GREEN0,RED0,AHL0,density,par,totaltime,dt,d=0.1,oneD =False, dimensionless=False,isdiffusion=True):
    

    
    t=0
    i=0
    
    if dimensionless:
        GREEN = []
        RED=[]
        AHL=[]
        GREENi=GREEN0
        REDi=RED0
        AHLi=AHL0
        
        while t < totaltime:
            g,r,a = model(GREENi,REDi,AHLi,d,density,par,isdiffusion,oneD)    
            GREENi = GREENi  + g*dt
            REDi = REDi  + r*dt
            AHLi = AHLi  + a*dt

            AHL.append(AHLi)
            GREEN.append(GREENi)
            RED.append(REDi)       
            t=t+dt
    else:
        GREEN = np.zeros((round(totaltime/dt)+1,round(w/d),round(h/d)))
        RED = np.zeros((round(totaltime/dt)+1,round(w/d),round(h/d)))
        AHL = np.zeros((round(totaltime/dt)+1,round(w/d),round(h/d)))     
        GREENi=GREEN0.copy()
        REDi=RED0.copy()
        AHLi=AHL0.copy()
        
        while t < totaltime:
            g,r,a = model(GREENi,REDi,AHLi,d,density,par,isdiffusion,oneD)
           
            GREENi[1:-1, 1:-1] = GREENi[1:-1, 1:-1] + g[1:-1, 1:-1]*dt
            REDi[1:-1, 1:-1] = REDi[1:-1, 1:-1] + r[1:-1, 1:-1]*dt 
            AHLi[1:-1, 1:-1] = AHLi[1:-1, 1:-1]  + a[1:-1, 1:-1]*dt
            
            GREEN[i]=GREENi.copy()
            RED[i]=REDi.copy()
            AHL[i]=AHLi.copy()
            
            t=t+dt
            i=i+1
    return GREEN,RED,AHL




def jacobianMatrix(G,R,A,par):
    # [ DG/dg   DG/dr   DG/da ]
    # [ DR/dg   DR/dr   DR/da ]
    # [ DA/dg   DA/dr   DA/da ]

    #need to double check , some mistake are here
    dGdg = - par['delta_green']    
    dGdr = -((par['alpha_green'] - par['beta_green'])*np.power((par['K_ahl_green']*A),par['n_ahl_green'])*par['n_RED']*np.power((par['K_RED']*R),par['n_RED']))
    dGdr =  dGdr/((np.power((par['K_ahl_green']*A),par['n_ahl_green'])+1)*R*np.power((np.power((par['K_RED']*R),par['n_RED'])+1),2))   
    dGda =((par['alpha_green'] - par['beta_green']) *par['n_ahl_green']*np.power((par['K_ahl_green']*A),par['n_ahl_green'])) 
    dGda =  dGda/((np.power((par['K_RED']*R),par['n_RED'])+1)*A*np.power((np.power((par['K_ahl_green']*A),par['n_ahl_green'])+1),2))
    
    dRdg = - ((par['alpha_red'] - par['beta_red'])*np.power((par['K_ahl_red']*A),par['n_ahl_red'])*par['n_GREEN']*np.power((par['K_GREEN']*G),par['n_GREEN']))
    dRdg =  dRdg/((np.power((par['K_ahl_red']*A),par['n_ahl_red'])+1)*G*np.power((np.power((par['K_GREEN']*G),par['n_GREEN'])+1),2))    
    dRdr = - par['delta_red']    
    dRda = ((par['alpha_red'] - par['beta_red']) *par['n_ahl_red']*np.power((par['K_ahl_red']*A),par['n_ahl_red']))
    dRda=dRda/((np.power((par['K_GREEN']*G),par['n_GREEN'])+1)*A*np.power((np.power((par['K_ahl_red']*A),par['n_ahl_red'])+1),2))
    
    dAdg = (par['beta_ahl']*par['n_ahl']*(np.power((par['K_ahl']*G),par['n_ahl'])))
    dAdg = dAdg/ (G*np.power((np.power((G*par['K_ahl']),par['n_ahl'])+1),2))
    dAdr = 0
    dAda = -par['delta_ahl']

    # add diffusion as in scholes et al.
   # dAda = dAda - (q**2)*par['D_ahl']
    A = np.array([[dGdg,dGdr,dGda],[dRdg,dRdr,dRda],[dAdg,dAdr,dAda]])
    
    return A



def findsteadystate(par,nstep=10.0,tt=1000,dt=0.1):
    #select some point to cover the phase portrait
    den=1
    G= np.arange(0,par['beta_green'],par['beta_green']/nstep)
    R= np.arange(0,par['beta_red'],par['beta_red']/nstep)
    A= np.arange(0,par['beta_ahl'],par['beta_ahl']/nstep)

    ss=np.zeros((len(G)*len(R)*len(A),3))

    m=0
    for i,g in enumerate(G):
        for j,r in enumerate(R):
            for k,a in enumerate(A):

                gi,ri,ai= Integration(g,r,a,density=den,par=par,totaltime=tt,dt=dt,d=d,oneD=True, dimensionless=True,isdiffusion=False)
                ss[m][0]=round(gi[-2],5)
                ss[m][1]=round(ri[-2],5)
                ss[m][2]=round(ai[-2],5)

                m=m+1
    
    ss=np.unique(ss, axis=0)
    print(ss)
    final_ss=ss.copy()
    index=[]
    for i,s in enumerate(ss):   
        if np.all(s>0.001):
            index.append(i) #print message
    final_ss=ss[index]
    if len(final_ss)>1:
        final_ss=[] #don't want to mess with multi stable state and oscillation now...
        print("multiple steady point")
    return final_ss



def turinginstability(par,nstep=10):
    #step one find steady stateS
    turing_type=0
    q=np.arange(0,10,0.2) 
    ss =findsteadystate(par,nstep)
    eigens=np.array([])
    print(ss)
    #jacobian matrix
    for i,s in enumerate(ss): #there is only one SS possible for the moment
        A=jacobianMatrix(s[0],s[1],s[2],par)
        eigvals, eigvecs =eig(A)
        sse=eigvals.real
 
    # add diffusion as in scholes et al.    
        eigens=np.array([])
        for qi in q:
            A=jacobianMatrix(s[0],s[1],s[2],par)
            A[2][2] = A[2][2] - (qi**2)*par['D_ahl']
            eigvals, eigvecs =eig(A)
            eigens=np.append(eigens,eigvals.real)
        if np.any(eigens>0) and np.all(sse<0):
            turing_type=2
            if eigens[-1]<0:
                turing_type=1


    return turing_type, eigens


def choosepar(parlist):
    #choose random par in the defined range
    samplepar=[]
    for ipar,par in enumerate(parlist):
        samplepar.append(random.uniform(par['lower_limit'], par['upper_limit']))
   # p=pars_to_dict(samplepar,parlist)
    return np.array(samplepar)


################################################3
#GRAPH function
################################################

def movieplot1d(g,r,a):    
    fig, ax = plt.subplots()
    line1, = ax.plot(r[-1][1],'r')
    line2, = ax.plot(g[-1][1],'g')
    line3, = ax.plot(a[-1][1],'b')

    #ax.set_ylim([0,2.1])
    def animate(i):
        line1.set_ydata(r[i][1])  # update the data.
        line2.set_ydata(g[i][1])  # update the data.
        line3.set_ydata(a[i][1])
        return line1,line2,

    ani = animation.FuncAnimation(
        fig, animate, interval=0, blit=True, frames=round(tt/dt), save_count=50)

    # To save the animation, use e.g.
    #
    # ani.save("movie.mp4")
    #
    # or
    #
    # writer = animation.FFMpegWriter(
    #     fps=15, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save("movie.mp4", writer=writer)

    plt.show()

def plot1d(g,r,a,t):
    fig, ax = plt.subplots()
    line1, = ax.plot(r[t][1],'r')
    line2, = ax.plot(g[t][1],'g')
    line2, = ax.plot(a[t][1],'b')

   # ax.set_ylim([0,2.1])
    plt.show()
    
def plottime(g,r,a,pos):    
    fig, ax = plt.subplots()
    if pos == 0:
        line1, = ax.plot(r[:],'r')
        line2, = ax.plot(g[:],'g')
        line3, = ax.plot(a[:],'b')
    else:
        line1, = ax.plot(r[:,pos[0],pos[1]],'r') #might bug didn't test
        line2, = ax.plot(g[:,pos[0],pos[1]],'g')
        line3, = ax.plot(a[:,pos[0],pos[1]],'b')

   # ax.set_ylim([0,2.1])
    plt.show()

    
##############################################################################

#ss =findsteadystate(par,6)
#s=ss[0]
'''
A=jacobianMatrix(100,100,100,par)
tr=np.trace(A)
det=np.linalg.det(A)
print(A,tr,det)
print(".------------.")
tt=10
tu,e = turinginstability(par,4)
print(tu)
gi,ri,ai= Integration(100,100,100,density=1,par=par,totaltime=tt,dt=dt,d=d,oneD=True, dimensionless=True,isdiffusion=False)
'''
print(".------.")
#to be parallelized

par=choosepar(parlist)
for i in np.arange(0,1000):
    newpar=choosepar(parlist)
    par=np.vstack((par,newpar))
    p=pars_to_dict(newpar,parlist)
    tu,e = turinginstability(p,4)
    print(tu,i)


for k in np.arange(0,len(parlist)):
    plt.subplot(4,5,k+1)
    plt.hist(par[:,k],bins=50)
    plt.xlabel(parlist[k]['name'])
plt.show()


'''
plt.plot(gi,'g')
plt.plot(ri,'r')
plt.plot(ai,'--b')
plt.ylim(0,105)
plt.show()

g,r,a = model(100,100,100,d,density=1,par=par,isdiffusion=False,oneD=True)    
'''

#0D
'''
# plate size & tie
tt = 20 #totaltime
w = h = 10 #10
w= 0.3
nx, ny = round(w/d), round(h/d)

ss=findsteadystate()
print(ss)
'''

'''
q,eigens=turinginstability(par)
plt.plot(q,eigens)
plt.ylim(-10,10)
plt.show()
'''


#1D
'''
tt = 100 #totaltime
h = 10 #10
w= 0.3
nx, ny = round(w/d), round(h/d)

#ss=findsteadystate()
#s=ss[1]
s=[100,100,100]



density=np.ones((nx, ny))
green = np.ones((nx, ny))*0
red = np.ones((nx, ny))*0
ahl = np.ones((nx, ny))*0



green[1,round(5/d)]=s[0]
red[1,round(5/d)]=s[1]
ahl[1,round(5/d)]=s[2]

#for j in np.arange(1,ny-1):
#            AHL[1][j]=AHL[1][j]+random.randint(0,1)/2

'''


'''
g,r,a= Integration(green*0,red,ahl,density,par,totaltime=tt,dt=dt,d=d,oneD=True, dimensionless=False,isdiffusion=False)
g2,r2,a2= Integration(green,red*0,ahl,density,par,totaltime=tt,dt=dt,d=d,oneD=True, dimensionless=False,isdiffusion=False)

plt.plot(g[-2][1][1:-2],'-g')
plt.plot(r[-2][1][1:-2],'-r')
plt.plot(a[-2][1][1:-2],'-b')
plt.plot(g2[-2][1][1:-2],'--g')
plt.plot(r2[-2][1][1:-2],'--r')
plt.plot(a2[-2][1][1:-2],'--b')

plt.show()

g,r,a= Integration(green*0,red,ahl,density,par,totaltime=tt,dt=dt,d=d,oneD=True, dimensionless=False,isdiffusion=True)
g2,r2,a2= Integration(green,red*0,ahl,density,par,totaltime=tt,dt=dt,d=d,oneD=True, dimensionless=False,isdiffusion=True)

plt.plot(g[-2][1][1:-2],'-g')
plt.plot(r[-2][1][1:-2],'-r')
plt.plot(a[-2][1][1:-2],'-b')
plt.plot(g2[-2][1][1:-2],'--g')
plt.plot(r2[-2][1][1:-2],'--r')
plt.plot(a2[-2][1][1:-2],'--b')

plt.show()
'''




#2d plot
'''

tt = 100 #totaltime
h= 8 #10
w= 8
nx, ny = round(w/d), round(h/d)


density=np.ones((nx, ny))
AHL = np.ones((nx, ny))*0
GREEN = np.ones((nx, ny))*0
RED = np.ones((nx, ny))*0
for j in np.arange(1,ny-1):
    for i in np.arange(1,nx-1):
            GREEN[i][j]=GREEN[i][j]+random.randint(0,1)*100

dg,dr,da= Integration(GREEN,RED,AHL,density,par,totaltime=tt,dt=dt,d=d,oneD=False, dimensionless=False,isdiffusion=True)

plt.subplot(1,3,1)
plt.imshow(dg[-2][:,:],cmap='Greens')#,vmin=0,vmax=0.5)
plt.subplot(1,3,2)
plt.imshow(dr[-2][:,:],cmap='Reds')#,vmin=0,vmax=0.5)
plt.subplot(1,3,3)
plt.imshow(da[-2][:,:],cmap='Blues')#,vmin=0,vmax=0.5)
plt.show()
'''
