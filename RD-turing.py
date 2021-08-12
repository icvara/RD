#RD system

import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.animation as animation
from numpy.linalg import eig
import multiprocessing
import time
from functools import partial
from scipy import optimize
from scipy.optimize import brentq
import pandas as pd

par={
    
    'K_RED':94,#3.5,
    'n_RED':2.0,
    'K_ahl':55,#35.0,#0.15,
    'n_ahl':2.0,   
    'beta_ahl':30.,
    'delta_ahl':1,#0.5,
    'D_ahl':0.01,
    'K_ahl2':27,#35.0,#0.15,
    'n_ahl2':2.0,  
    'beta_ahl2':2.80,  
    'delta_ahl2':1,#0.5,
    'D_ahl2':0.01
    }


parlist = [ 

    {'name' : 'K_RED', 'lower_limit':0.0,'upper_limit':100.0},    
    {'name' : 'K_ahl', 'lower_limit':0.0,'upper_limit':100.0},
    {'name' : 'beta_ahl', 'lower_limit':0.0,'upper_limit':100.0},  
    {'name' : 'K_ahl2', 'lower_limit':0.0,'upper_limit':100.0},
    {'name' : 'beta_ahl2', 'lower_limit':0.0,'upper_limit':100.0},

]





maxD= 1.1
d = 0.1 #interval size
d2 = d*d
dt = d2 * d2 / (2 * maxD * (d2 + d2)) #interval time according to diffusion and interval dist
tt = 100 #totaltime
h = 10 #10
w= 0.3
nx, ny = round(w/d), round(h/d)


###########################################3

def choosepar(parlist):
    #choose random par in the defined range
    samplepar=[]
    for ipar,par in enumerate(parlist):
        samplepar.append(random.uniform(par['lower_limit'], par['upper_limit']))
   # p=pars_to_dict(samplepar,parlist)
    return np.array(samplepar)


def getfixedpar(par):
    par['n_RED']=2.0
    par['n_ahl']=2.0
    par['n_ahl2']=2.0
    par['delta_ahl']=1
    par['delta_ahl2']=1
    par['D_ahl']=0.01
    par['D_ahl2']=1

    return par

def diffusion(u0,d,D, oneD=False):
    d2 = d*d
    dt = d2 * d2 / (2 * maxD * (d2 + d2))
    u = u0.copy()
    uxx = (u0[2:, 1:-1] - 2*u0[1:-1, 1:-1] + u0[:-2, 1:-1])/d2
    uyy = (u0[1:-1, 2:] - 2*u0[1:-1, 1:-1] + u0[1:-1, :-2])/d2
    if oneD:
        uxx= uxx*0
    u[1:-1, 1:-1] =  D*(uxx + uyy)
    return u

def model(AHLi,AHL2i,d,density,par, isdiffusion=True,oneD =False):
    #list of fixed par
    par=getfixedpar(par)

    if isdiffusion:
        AHL = (par['beta_ahl']*np.power(AHLi*par['K_ahl'],par['n_ahl']))/(1+np.power(AHLi*par['K_ahl'],par['n_ahl']))
        AHL= AHL / (1 + np.power(AHL2i*par['K_RED'],par['n_RED']))   
        AHLdif = diffusion(AHLi,d,par['D_ahl'],oneD)
        AHL = AHL*density + AHLdif - par['delta_ahl']*(AHLi)
        
        AHL2 = (par['beta_ahl2']*np.power(AHLi*par['K_ahl2'],par['n_ahl2']))/(1+np.power(AHLi*par['K_ahl2'],par['n_ahl2']))
        AHL2dif = diffusion(AHL2i,d,par['D_ahl2'],oneD)
        AHL2 = AHL2*density + AHL2dif - par['delta_ahl2']*(AHL2i)
    
    if isdiffusion == False:
        AHL = (par['beta_ahl']*np.power(AHLi*par['K_ahl'],par['n_ahl']))/(1+np.power(AHLi*par['K_ahl'],par['n_ahl']))
        AHL= AHL / (1 + np.power(AHL2i*par['K_RED'],par['n_RED']))
        AHL = AHL*density  - par['delta_ahl']*(AHLi)
        AHL2 = (par['beta_ahl2']*np.power(AHLi*par['K_ahl2'],par['n_ahl2']))/(1+np.power(AHLi*par['K_ahl2'],par['n_ahl2']))
        AHL2 = AHL2*density  - par['delta_ahl2']*(AHL2i)


    return  AHL, AHL2



def Integration(AHL0,AHL20,density,par,totaltime,dt,d, oneD =False, dimensionless=False,isdiffusion=True):

    t=0
    i=0

    if dimensionless:
            
        AHL=[]
        AHL2=[]
        AHLi=AHL0
        AHL2i=AHL20
        
        while t < totaltime:
            a,a2 = model(AHLi,AHL2i,d,density,par,isdiffusion,oneD)    
            AHLi = AHLi  + a*dt
            AHL2i = AHL2i  + a2*dt
            AHL.append(AHLi)
            AHL2.append(AHL2i)       
            t=t+dt

    else:
      
        
        AHL = np.zeros((round(totaltime/dt)+1,round(w/d),round(h/d)))
        AHL2 = np.zeros((round(totaltime/dt)+1,round(w/d),round(h/d)))   
        AHLi=AHL0.copy()
        AHL2i=AHL20.copy() 

        while t < totaltime:
            a,a2 = model(AHLi,AHL2i,d,density,par,isdiffusion,oneD)    
            AHLi[1:-1, 1:-1] = AHLi[1:-1, 1:-1]  + a[1:-1, 1:-1]*dt
            AHL2i[1:-1, 1:-1] = AHL2i[1:-1, 1:-1]  + a2[1:-1, 1:-1]*dt
            AHL[i]=AHLi.copy()
            AHL2[i]=AHL2i.copy()        
            t=t+dt
            i=i+1
            
    return AHL,AHL2



def jacobianMatrix(AHL,AHL2,par):
    par=getfixedpar(par)
    # [ DX/dx   DX/dy ]
    # [ Dy/dx   Dy/dy ]
    DXdx = (par['beta_ahl']*par['n_ahl']*np.power((par['K_ahl']*AHL),par['n_ahl']))/((np.power((par['K_RED']*AHL2),par['n_RED'])+1)*AHL*(np.power((par['K_ahl']*AHL),par['n_ahl'])+1))
    DXdx = DXdx - (par['beta_ahl']*par['n_ahl']*np.power((par['K_ahl']*AHL),2*par['n_ahl']))/((np.power((par['K_RED']*AHL2),par['n_RED'])+1)*AHL*(np.power((np.power((par['K_ahl']*AHL),par['n_ahl'])+1),2)))
    DXdx = DXdx - par['delta_ahl']

    DXdy = - (par['beta_ahl']*par['n_RED']*np.power((par['K_ahl']*AHL),par['n_ahl'])*np.power((par['K_RED']*AHL2),par['n_RED']))/((np.power((par['K_ahl']*AHL),par['n_ahl'])+1)*AHL2*(np.power((np.power((par['K_RED']*AHL2),par['n_RED'])+1),2)))
                 
    DYdx =  (par['beta_ahl2']*par['n_ahl2']*np.power((par['K_ahl2']*AHL),par['beta_ahl2']))/(AHL*(np.power((np.power((par['K_ahl2']*AHL),par['n_ahl2'])+1),2)))

    DYdy = - par['delta_ahl2']
    
    A = np.array([[DXdx,DXdy],[DYdx,DYdy]])
    
    return A

def approxJacob(AHL,AHL2,par):
    delta=10e-5
    density=1

    H,H2= model(AHL,AHL2,d,density,par, isdiffusion=False,oneD =True)
    Dxdx= (model(AHL+delta,AHL2,d,density,par, isdiffusion=False,oneD =True)[0] - H) /delta
    Dxdy= (model(AHL,AHL2+delta,d,density,par, isdiffusion=False,oneD =True)[0] - H) /delta
    Dydx= (model(AHL+delta,AHL2,d,density,par, isdiffusion=False,oneD =True)[1] - H2) /delta
    Dydy= (model(AHL,AHL2+delta,d,density,par, isdiffusion=False,oneD =True)[1] - H2) /delta

    A = np.array([[DXdx,DXdy],[DYdx,DYdy]])
    
    return A

def solvedfunction(AHLi,par):
    #rewrite the system equation to have only one unknow and to be call with scipy.optimze.brentq
    #the output give a function where when the line reach 0 are a steady states
    #par=getfixedpar(par)
    AHL2ss = (par['beta_ahl2']*np.power(AHLi*par['K_ahl2'],par['n_ahl2']))/(1+np.power(AHLi*par['K_ahl2'],par['n_ahl2']))
    AHL2ss = AHL2ss/par['delta_ahl2']
    AHLss= (par['beta_ahl']*np.power(AHLi*par['K_ahl'],par['n_ahl']))/(1+np.power(AHLi*par['K_ahl'],par['n_ahl']))
    AHLss = AHLss / (1 + np.power(AHL2ss*par['K_RED'],par['n_RED']))
    AHLss = AHLss/par['delta_ahl']    
    funAHL= AHLss - AHLi
    return funAHL



def findss(par):
    #list of fixed par
    par=getfixedpar(par)  
    #function to find steady state
    #1. find where line reached 0
    AHLi=np.logspace(-10,5,200,base=10) # generate x axis with min and max of real biological value
    f=solvedfunction(AHLi,par)
    x=f[1:-1]*f[0:-2] #when the output give <0, where is a change in sign, meaning 0 is crossed
    index=np.where(x<0)
    ss=[]
    for i in index[0]:
        H=brentq(solvedfunction, AHLi[i], AHLi[i+1],args=par) #find the value of AHL at 0
        #now we have AHL we can find AHL2 ss
        H2=(par['beta_ahl2']*np.power(H*par['K_ahl2'],par['n_ahl2']))/(1+np.power(H*par['K_ahl2'],par['n_ahl2']))
        H2=H2/par['delta_ahl2']
        ss.append(np.array([H,H2]))
    return ss

def turinginstability(par):
    #step one find steady stateS
    par=getfixedpar(par)

    turing_type=[]
    q=np.arange(0,200,1) 
    #q=np.logspace(-4,4,500,base=10)
#    ss =findsteadystate(par,nstep)
    ss= findss(par)
    sse=[]
    if len(ss)==0:
        turing_type.append(np.nan)
    for i,s in enumerate(ss): 
        A=jacobianMatrix(s[0],s[1],par)
        eigvals, eigvecs =eig(A)
        sse.append(eigvals.real)
        pos=sse[i][sse[i]>0]

        if np.all(sse[i]<0): #if all neg = stable point, test turing instability
            # add diffusion as in scholes et al.
            eigens_1=[]
            eigens_2=[]
            for qi in q:
                A=jacobianMatrix(s[0],s[1],par)
                A[0][0] = A[0][0] - (qi**2)*par['D_ahl']
                A[1][1] = A[1][1] - (qi**2)*par['D_ahl2']
                eigvals, eigvecs =eig(A)
                #eigens=np.append(eigens,eigvals.real)
                eigens_1.append(eigvals.real[0])
                eigens_2.append(eigvals.real[1])
            eigens=np.array([eigens_1,eigens_2])

               # plt.plot(eigens)
               # plt.show()
            if np.any(eigens>0):
                print("Tu instability")
               # print(eigens)
                if np.all(eigens[:,-1]<0):
                    turing_type.append(1)
                else:
                    turing_type.append(2)
            else:
                turing_type.append(0)

        if len(pos)>1:
            if pos[0]-pos[1] == 0:
                print("oscillation")
                turing_type.append(4)  #need to check hpf instability , need to add line here
            else:
                turing_type.append(0)  
        else:
            turing_type.append(0)

    return ss, turing_type


def pars_to_dict(pars,parlist):
### This function is not necessary, but it makes the code a bit easier to read,
### it transforms an array of pars e.g. p[0],p[1],p[2] into a
### named dictionary e.g. p['k0'],p['B'],p['n'],p['x0']
### so it is easier to follow the parameters in the code
    dict_pars = {}
    for ipar,par in enumerate(parlist):
        dict_pars[par['name']] = pars[ipar] 
    return dict_pars


def GeneratePars(parlist, ncpus,Npars=1000):
    #@EO: to compare the 2 versions, set the seed
    #np.random.seed(0)

    ## Call to function GeneratePar in parallel until Npars points are accepted
    trials = 0
    start_time = time.time()
    results = []

    pool = multiprocessing.Pool(ncpus)
    results = pool.map(func=partial(calculatePar, parlist),iterable=range(Npars), chunksize=10)
    pool.close()
    pool.join()    
    end_time = time.time()
    print(f'>>>> Loop processing time: {end_time-start_time:.3f} sec on {ncpus} CPU cores.')    

    newparlist = [result[0] for result in results]
    turingtype = [result[1] for result in results]

    return(newparlist,turingtype)


def calculatePar(parlist, iter):
  #selectpar=[]
  turingtype=[]
  newpar=choosepar(parlist)    
  p=pars_to_dict(newpar,parlist)
  ss,tu = turinginstability(p)
  #if tu >0:
    #selectpar.append(newpar)
  turingtype.append(tu)
  return newpar,turingtype


def load(name,parlist):
    tutype= np.loadtxt(name+"_turingtype.out")
    p= np.loadtxt(name+"_par.out")

    namelist=[]
    for i,par in enumerate(parlist):
        namelist.append(parlist[i]['name'])
    df = pd.DataFrame(p, columns = namelist)
    df['tutype']=tutype
    return df

def run(name):
    par,tutype=GeneratePars(parlist, ncpus=40,Npars=5000)
    np.savetxt(name+'_turingtype.out', tutype)
    np.savetxt(name+'_par.out', par)

####################################################

def movieplot1d(a,a2):    
    fig, ax = plt.subplots()
    line1, = ax.plot(a2[-1][1],'r')
    line2, = ax.plot(a[-1][1],'g')
    ax.set_ylim([0,2.1])
    def animate(i):
        line1.set_ydata(a2[i][1])  # update the data.
        line2.set_ydata(a[i][1])  # update the data.
        return line1,line2,

    ani = animation.FuncAnimation(
        fig, animate, interval=0, blit=True, frames=round(tt/dt), save_count=50)

    # To save the animation, use e.g.
    #
    #ani.save("movie.mp4")
    #
    # or
    #
    #writer = animation.FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    #ani.save("movie.mp4", writer=writer)

    plt.show()

def plot1d(a,a2,t):
    fig, ax = plt.subplots()
    line1, = ax.plot(a2[t][1],'r')
    line2, = ax.plot(a[t][1],'g')
    ax.set_ylim([0,1])
    plt.show()
    

def plottime(a,a2,pos):    
    fig, ax = plt.subplots()
    if pos == 0:
        line1, = ax.plot(a2[:],'r')
        line2, = ax.plot(a[:],'g')
    else:
        line1, = ax.plot(a2[:,pos[0],pos[1]],'r') #might bug didn't test
        line2, = ax.plot(a[:,pos[0],pos[1]],'g')
    ax.set_ylim([0,2.1])
    plt.show()

def par_plot(name,df,parlist):

    fonts=2
    namelist=[]
    for i,par in enumerate(parlist):
        namelist.append(parlist[i]['name'])
     
    for i,par1 in enumerate(namelist):
        for j,par2 in enumerate(namelist):
            plt.subplot(len(namelist),len(namelist), i+j*len(namelist)+1)
            if i == j :
                plt.hist(df[par1])
                plt.xlim((parlist[i]['lower_limit'],parlist[i]['upper_limit']))
            else:
                plt.scatter(df[par1],df[par2], c=df['tutype'], s=0.001, cmap='viridis')# vmin=mindist, vmax=maxdist)
                plt.xlim((parlist[i]['lower_limit'],parlist[i]['upper_limit']))
                plt.ylim((parlist[j]['lower_limit'],parlist[j]['upper_limit']))
            if i > 0 and j < len(namelist)-1 :
                plt.xticks([])
                plt.yticks([])
            else:
                if i==0 and j!=len(namelist)-1:
                    plt.xticks([])
                    plt.ylabel(par2,fontsize=fonts)
                    plt.yticks(fontsize=fonts,rotation=90)
                if j==len(namelist)-1 and i != 0:
                    plt.yticks([])
                    plt.xlabel(par1,fontsize=fonts)
                    plt.xticks(fontsize=fonts)
                else:
                    plt.ylabel(par2,fontsize=fonts)
                    plt.xlabel(par1,fontsize=fonts)
                    plt.xticks(fontsize=fonts)
                    plt.yticks(fontsize=4,rotation=90)
            
      
    plt.savefig(name+'_full_par_plot.pdf', bbox_inches='tight')
    plt.close()




def niceplot(name):
    df=load(name,parlist)
    par_plot(name,df,parlist)


    turingpar=[]
    tu_df = df[df['tutype']>0]
    nrow=tu_df.shape[0]

    tt = 100 #totaltime
    h = 10 #10
    w= 0.3
    nx, ny = round(w/d), round(h/d)
    density=np.ones((nx, ny))
    AHL = np.ones((nx, ny))*0.2
    AHL2 = np.ones((nx, ny))*0.2
    #AHL[1,round(5/d)]=5
    #AHL2[1,round(5/d)]=5
    for i in np.arange(1,ny-1):
                AHL[1][i]=AHL[1][i]*random.randint(0,10)/10
                AHL2[1][i]=AHL2[1][i]*random.randint(0,10)/10

    print(nrow)
    for n in np.arange(0,nrow):
        par=tu_df.iloc[n].tolist()[:-1] #transform in list and remove turing type
        p=pars_to_dict(par,parlist)
        da, da2= Integration(AHL,AHL2,density,p,totaltime=tt,dt=dt,d=d,oneD=True, dimensionless=False,isdiffusion=True)
       # plot1d(da,da2,0)
        plt.subplot(round(np.sqrt(nrow)),round(np.sqrt(nrow)),n+1)
        plt.plot(da[-2][1],'g')
        plt.plot(da2[-2][1],'r')
        plt.yscale("log")
        print(n)
        
    plt.savefig(name+'_Tu_plot.pdf', bbox_inches='tight')
    #plt.show()


##########################################################

name='turing_001'
run(name)
niceplot(name)




'''
a, a2= Integration(AHL,AHL2,density,par,totaltime=tt,dt=dt,d=d,oneD=True, dimensionless=False,isdiffusion=False)
da, da2= Integration(AHL,AHL2,density,par,totaltime=tt,dt=dt,d=d,oneD=True, dimensionless=False,isdiffusion=True)
plot1d(a,a2,-2)
#movieplot1d(a,a2)
plot1d(da,da2,0)
plot1d(da,da2,-2)
#movieplot1d(da,da2)
#plottime(da,da2,pos=[1,round(5/d)])
'''



'''
par['K_RED']=20#2
par['beta_ahl2']=2
par['beta_ahl']=20

par['K_ahl']=10
par['K_ahl2']=10

par['delta_ahl']=1#1.0
par['delta_ahl2']=1#1.0
par['D_ahl2']=1#0.001
par['D_ahl']=0.01

tu,e = turinginstability(par,4)
print(tu)


selectpar=[]
par=choosepar(parlist)
for i in np.arange(0,10):
    newpar=choosepar(parlist)
    par=np.vstack((par,newpar))
    p=pars_to_dict(newpar,parlist)
    tu,e = turinginstability(p,4)
    if tu >0:
        if len(selectpar)==0:
            selectpar=newpar
        else:
            selectpar=np.vstack((selectpar,newpar))
    print(tu,i)



'''

'''
for k in np.arange(0,len(parlist)):
    plt.subplot(4,5,k+1)
    plt.hist(par[:,k],bins=50)
    plt.xlabel(parlist[k]['name'])
plt.show()
'''






##ss =findsteadystate(par,6)
##s=ss[0]



'''
c=0
for kr in [10,15,20]:
    for ba in [5,10,20]:
        for ba2 in [2]:
            for ka in [10]:
                for ka2 in [10]:
                    par['K_RED']=kr
                    par['K_ahl']=ka
                    par['K_ahl2']=ka2
                    par['beta_ahl']=ba
                    par['beta_ahl2']=ba2
                    tu,e = turinginstability(par,3)
                    if tu >0:
                        print(tu,kr,ba,ba2,ka,ka2)
                    c=c+1
                    print(c)

'''


'''
#0D

# plate size & tie
tt = 500 #totaltime
w = h = 10 #10
w= 0.3
nx, ny = round(w/d), round(h/d)

density=1
AHL=1
AHL2=1
i=1
'''
'''
for par1 in [7,10,15,20]:
    for par2 in [7,10,15,20]:
        par['beta_ahl']=par1
        par['beta_ahl2']=par2

        plt.subplot(4,4,i)
        a, a2= Integration(AHL,AHL2,density,par,totaltime=tt,dt=dt,d=d,oneD=True, dimensionless=True,isdiffusion=False)
        plt.plot(a2[:],'r')
        plt.plot(a[:],'g')

        i=i+1
       # q,eigens=turinginstability(AHL,AHL2,par)
       # plt.plot(q,eigens)
       # plt.ylim(-10,10)
plt.show()

i=1
for par1 in [7,10,15,20]:
    for par2 in [7,10,15,20]:
        par['beta_ahl']=par1
        par['beta_ahl2']=par2

        plt.subplot(4,4,i)
        q,eigens=turinginstability(AHL,AHL2,par)
        plt.plot(q,eigens)
        plt.ylim(-10,10)

        i=i+1

plt.show()
'''

'''
#1D

tt = 500 #totaltime
h = 10 #10
w= 0.3
nx, ny = round(w/d), round(h/d)

density=np.ones((nx, ny))
AHL = np.ones((nx, ny))*0
AHL2 = np.ones((nx, ny))*0
AHL[1,round(5/d)]=ss[0][0]
AHL2[1,round(5/d)]=ss[0][1]


AHL = np.ones((nx, ny))*ss[0][0]
AHL2 = np.ones((nx, ny))*ss[0][1]
for i in np.arange(1,ny-1):
            AHL[1][i]=AHL[1][i]*random.randint(0,10)/10
            AHL2[1][i]=AHL2[1][i]*random.randint(0,10)/10



a, a2= Integration(AHL,AHL2,density,par,totaltime=tt,dt=dt,d=d,oneD=True, dimensionless=False,isdiffusion=False)
da, da2= Integration(AHL,AHL2,density,par,totaltime=tt,dt=dt,d=d,oneD=True, dimensionless=False,isdiffusion=True)
plot1d(a,a2,-2)
#movieplot1d(a,a2)
plot1d(da,da2,0)
plot1d(da,da2,-2)
#movieplot1d(da,da2)
#plottime(da,da2,pos=[1,round(5/d)])
'''



#2d plot
'''

tt = 500 #totaltime
h= 6 #10
w= 6
nx, ny = round(w/d), round(h/d)


density=np.ones((nx, ny))
AHL = np.ones((nx, ny))*0
AHL2 = np.ones((nx, ny))*0
AHL[round(4/d),round(4/d)]=ss[0][0]
AHL2[round(4/d),round(4/d)]=ss[0][1]

for j in np.arange(1,ny-1):
    for i in np.arange(1,nx-1):
            AHL[i][j]=AHL[i][j]+random.randint(0,10)/10*ss[0][0]

a, a2= Integration(AHL,AHL2,density,par,totaltime=tt,dt=dt,d=d)

plt.subplot(1,2,1)
plt.imshow(a[-2][:,:],cmap='Greens',vmin=0,vmax=0.5)
plt.subplot(1,2,2)
plt.imshow(a2[-2][:,:],cmap='Reds',vmin=0,vmax=0.5)
plt.show()

'''

'''
fig=plt.figure()
ims =[]
for i in range(int(tt/dt)):
    im=plt.imshow(a[i][:,:],animated=True,cmap='Greens')
    ims.append([im])
#print(ims)
ani = animation.ArtistAnimation(fig, ims, interval=0.0001, blit=True)
#ani.save('dynamic_images.mp4')
#plt.show()

fig=plt.figure()
ims =[]
for i in range(int(tt/dt)):
    im=plt.imshow(a2[i][:,:],animated=True,cmap='Reds')
    ims.append([im])
#print(ims)
ani = animation.ArtistAnimation(fig, ims, interval=0, blit=True)
# ani.save('dynamic_images.mp4')
plt.show()
'''



