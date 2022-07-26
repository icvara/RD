import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.linalg import eig
import random

from scipy import optimize
from scipy.optimize import brentq
import pandas as pd
import seaborn as sns

par={
    
    'alpha_red': 0,
    
    'beta_red': 1000,

    'K_IPTG':2.22,

    'K_GREEN':1,
    'K_RED':3.5,

    'n_GREEN':2,
    'n_RED':2,
    'delta_red':1,#8.3,
    'delta_green':1,#8.3,

    #need to be defined
    'alpha_green': 0,
    'beta_green': 100,
    'leak_green':0,

    'K_ahl_red':10,
    'K_ahl_green':233,
    'n_ahl_green':1.61,
    'n_ahl_red':1.61,

    #luxI par
    'beta_ahl':0.1,
    'K_ahl':10,
    'n_ahl':4.,
    'delta_ahl':1#1

}

# barbier et al. 2020
paper_par={
    
    'alpha_red': 364,
    
    'beta_red': 362,
    'K_IPTG':2.22,
    'K_ahl_red':133,
    'K_GREEN':4.17E-2,
    'K_RED':27.4E-2,
    'n_GREEN':2.17,
    'n_ahl_red':1.61,
    'n_RED':2.29,
    'delta_red':1,#8.3,
    'delta_green':1,#8.3,

    #need to be defined
    'alpha_green': 310,
    'beta_green': 438,

    'K_ahl_green':133,
    'n_ahl_green':1.61,

  


    }

def model_TSL(GREENi,REDi,AHLi,IPTG,par):
	#here to calculate steady state:  we do without diffusion and cell density
    GREENi = np.maximum(GREENi - par['alpha_green'],0) # fluorescence background on X
    REDi = np.maximum(REDi - par['alpha_red'],0) # fluorescence background on X

    free_laci= GREENi / ( 1 + par['K_IPTG']*IPTG)
    RED = (par['beta_red']*np.power(AHLi*par['K_ahl_red'],par['n_ahl_red']))/(1+np.power(AHLi*par['K_ahl_red'],par['n_ahl_red']))
    RED = RED / (1 + np.power(free_laci*par['K_GREEN'],par['n_GREEN']))
    RED = RED - par['delta_red']*REDi #+ par['alpha_red']


    GREEN = par['beta_green'] # 1 inducer first
    GREEN = GREEN / (1 + np.power(REDi*par['K_RED'],par['n_RED']))
    GREEN = GREEN - par['delta_green']*GREENi
   # GREEN = GREEN # + par['alpha_green']



    return GREEN,RED

def model_TSLT(GREENi,REDi,AHLi,IPTG,par):
    #here to calculate steady state:  we do without diffusion and cell density
    GREENi = np.maximum(GREENi - par['alpha_green'],0) # fluorescence background on X
    REDi = np.maximum(REDi - par['alpha_red'],0) # fluorescence background on X
    GREEN = (par['beta_green']*np.power(AHLi*par['K_ahl_green'],par['n_ahl_green']))/(1+np.power(AHLi*par['K_ahl_green'],par['n_ahl_green']))
    GREEN = GREEN / (1 + np.power(REDi*par['K_RED'],par['n_RED']))
    GREEN = GREEN - par['delta_green']*GREENi  + par['leak_green']
  #  GREEN = GREEN # + par['alpha_green']

    free_GREENi= GREENi / ( 1+ par['K_IPTG']*IPTG)

    RED = (par['beta_red']*np.power(AHLi*par['K_ahl_red'],par['n_ahl_red']))/(1+np.power(AHLi*par['K_ahl_red'],par['n_ahl_red']))
    RED = RED / (1 + np.power(free_GREENi*par['K_GREEN'],par['n_GREEN']))
    RED = RED - par['delta_red']*REDi #+ par['alpha_red']


    return GREEN,RED

def model_TSXLT(GREENi,REDi,AHLi,IPTG,par):
    #here to calculate steady state:  we do without diffusion and cell density
    GREENi = np.maximum(GREENi - par['alpha_green'],0) # fluorescence background on X
    REDi = np.maximum(REDi - par['alpha_red'],0) # fluorescence background on X
    GREEN = (par['beta_green']*np.power(AHLi*par['K_ahl_green'],par['n_ahl_green']))/(1+np.power(AHLi*par['K_ahl_green'],par['n_ahl_green']))
    GREEN = GREEN / (1 + np.power(REDi*par['K_RED'],par['n_RED']))
    GREEN = GREEN - par['delta_green']*GREENi + par['leak_green']
   # GREEN = GREEN #+ par['alpha_green']

    free_GREENi= GREENi / ( 1+ par['K_IPTG']*IPTG)

    RED = (par['beta_red']*np.power(AHLi*par['K_ahl_red'],par['n_ahl_red']))/(1+np.power(AHLi*par['K_ahl_red'],par['n_ahl_red']))
    RED = RED / (1 + np.power(free_GREENi*par['K_GREEN'],par['n_GREEN']))
    RED = RED - par['delta_red']*REDi # + par['alpha_red']

    AHL = (par['beta_ahl']*np.power(GREENi*par['K_ahl'],par['n_ahl']))/(1+np.power(GREENi*par['K_ahl'],par['n_ahl']))
    AHL = AHL - par['delta_ahl']*(AHLi) 

    return GREEN,RED,AHL


def Integration(G0,R0,U0,IPTG,p,totaltime=500,dt=0.1):   
    #U0 need to be of shape (totaltime,nx,ny)
    Gi=G0
    Ri=R0
    Ai=U0
    AHL=U0

    G = np.zeros(((round(totaltime/dt+0.5)+1),U0.shape[0],G0.shape[1]))
    R = np.zeros(((round(totaltime/dt+0.5)+1),U0.shape[0],G0.shape[1]))
    A= np.zeros(((round(totaltime/dt+0.5)+1),U0.shape[0],G0.shape[1]))
    G[0]=G0
    R[0]=R0
    A[0]=U0

    t=dt
    i=1

    while t < totaltime:
        g,r = model_TSL(Gi,Ri,U0,IPTG,p)
        g,r = model_TSLT(Gi,Ri,U0,IPTG,p)
        a=0
        g,r,a = model_TSXLT(Gi,Ri,Ai,IPTG,p)


        Gi = Gi + g*dt
        Ri = Ri + r*dt
        Ai= Ai + a*dt + AHL
        G[i]=Gi
        R[i]=Ri
        A[i]=Ai
        t=t+dt
        i=i+1
    return G, R,A


def jacobianMatrix(G,R,A,par):
    # [ DG/dg   DG/dr   DG/da ]
    # [ DR/dg   DR/dr   DR/da ]
    # [ DA/dg   DA/dr   DA/da ]    

    #need to double check , some mistake are here
    dGdg = - par['delta_green']    
    dGdr = -((par['beta_green'])*np.power((par['K_ahl_green']*A),par['n_ahl_green'])*par['n_RED']*np.power((par['K_RED']*R),par['n_RED']))
    dGdr =  dGdr/((np.power((par['K_ahl_green']*A),par['n_ahl_green'])+1)*R*np.power((np.power((par['K_RED']*R),par['n_RED'])+1),2))   
    dGda =((par['beta_green']) *par['n_ahl_green']*np.power((par['K_ahl_green']*A),par['n_ahl_green'])) 
    dGda =  dGda/((np.power((par['K_RED']*R),par['n_RED'])+1)*A*np.power((np.power((par['K_ahl_green']*A),par['n_ahl_green'])+1),2))
    
    dRdg = - ((par['beta_red'])*np.power((par['K_ahl_red']*A),par['n_ahl_red'])*par['n_GREEN']*np.power((par['K_GREEN']*G),par['n_GREEN']))
    dRdg =  dRdg/((np.power((par['K_ahl_red']*A),par['n_ahl_red'])+1)*G*np.power((np.power((par['K_GREEN']*G),par['n_GREEN'])+1),2))    
    dRdr = - par['delta_red']    
    dRda = ((par['beta_red']) *par['n_ahl_red']*np.power((par['K_ahl_red']*A),par['n_ahl_red']))
    dRda=dRda/((np.power((par['K_GREEN']*G),par['n_GREEN'])+1)*A*np.power((np.power((par['K_ahl_red']*A),par['n_ahl_red'])+1),2))
    
    dAdg = (par['beta_ahl']*par['n_ahl']*(np.power((par['K_ahl']*G),par['n_ahl'])))
    dAdg = dAdg/ (G*np.power((np.power((G*par['K_ahl']),par['n_ahl'])+1),2))
    dAdr = 0
    dAda = -par['delta_ahl']
    A = np.array([[dGdg,dGdr,dGda],[dRdg,dRdr,dRda],[dAdg,dAdr,dAda]])  
    return A


def solvedfunction(Gi,AHLe,par):
    #rewrite the system equation to have only one unknow and to be call with scipy.optimze.brentq
    #the output give a function where when the line reach 0 are a steady states
    A= (par['beta_ahl']*np.power(Gi*par['K_ahl'],par['n_ahl']))/(1+np.power(Gi*par['K_ahl'],par['n_ahl']))
    A= (A/par['delta_ahl'])
    A= A + AHLe
    A=np.array(A)
    A[A<0]=0


    R = (par['beta_red'])*np.power(A*par['K_ahl_red'],par['n_ahl_red'])/(1+np.power(A*par['K_ahl_red'],par['n_ahl_red']))
    R = R / (1 + np.power(Gi*par['K_GREEN'],par['n_GREEN']))
    R = R/ par['delta_red']

    G = (par['beta_green'])*np.power(A*par['K_ahl_green'],par['n_ahl_green'])/(1+np.power(A*par['K_ahl_green'],par['n_ahl_green']))
    G = G / (1 + np.power(R*par['K_RED'],par['n_RED']))
    G = G / par['delta_green']
    func = G - Gi

    return func
  



def findss(par,AHLe):
    #list of fixed par
    #function to find steady state
    #1. find where line reached 0
    Gi=np.logspace(-10,5,1000,base=10)
    Gi[0]=0
    ss=[]
    nNode=3 # number of nodes : X,Y,Z
    nStstate= 5
    ss=np.ones((len(AHLe),nStstate,nNode))*np.nan

    for ai,a in enumerate(AHLe):
        f=solvedfunction(Gi,a,par)
        x=f[1:-1]*f[0:-2] #when the output give <0, where is a change in sign, meaning 0 is crossed
        index=np.where(x<0)

        for it,i in enumerate(index[0]):
            G=brentq(solvedfunction, Gi[i], Gi[i+1],args=(a,par)) #find the value of AHL at 0
            #now we have AHL we can find AHL2 ss
            A= (par['beta_ahl']*np.power(G*par['K_ahl'],par['n_ahl']))/(1+np.power(G*par['K_ahl'],par['n_ahl']))
            A= A/par['delta_ahl'] + a

            R = par['alpha_red']+((par['beta_red']-par['alpha_red'])*np.power(A*par['K_ahl_red'],par['n_ahl_red']))/(1+np.power(A*par['K_ahl_red'],par['n_ahl_red']))
            R = R / (1 + np.power(G*par['K_GREEN'],par['n_GREEN']))
            R = R/ par['delta_red']        
           # ss.append(np.array([G,R,A]))

       
            ss[ai][it]=np.array([G,R,A])



    return ss


def getEigen(ARA,par,s):
    A=meq.jacobianMatrix(ARA,s[0],s[1],s[2],par)
    eigvals, eigvecs =np.linalg.eig(A)
    sse=eigvals.real
    return sse 
#################################################################33
#######################################################################

AHLe=np.logspace(-10,-1,100,base=10)
AHLe[0]=0
#AHLe=np.zeros(5)

ss=findss(par,AHLe)


st=np.ones(ss.shape)*np.nan
os=np.ones(ss.shape)*np.nan
un=np.ones(ss.shape)*np.nan

for i,s1 in enumerate(ss):
    for k,s in enumerate(s1):
        if np.any(np.isnan(s)==False): 
            j = jacobianMatrix(s[0],s[1],s[2],par) 
            eigvals, eigvecs =np.linalg.eig(j)
            sse=eigvals.real
            if np.all(sse<0):
                st[i][k]=s
            else:
                pos=sse[sse>0]
                if len(pos)==2:
                    if pos[0]-pos[1]==0:
                        os[i][k]=s
                else:
                    un[i][k]=s


plt.subplot(1,3,1)
plt.plot(AHLe,os[:,:,0],'*m')
plt.plot(AHLe,st[:,:,0],'og')
plt.plot(AHLe,un[:,:,0],'.k')
plt.yscale('log')
plt.xscale('log')

plt.subplot(1,3,2)
plt.plot(AHLe,os[:,:,1],'*m')
plt.plot(AHLe,st[:,:,1],'or')
plt.plot(AHLe,un[:,:,1],'.k')
plt.yscale('log')
plt.xscale('log')

plt.subplot(1,3,3)
plt.plot(AHLe,os[:,:,2],'*m')
plt.plot(AHLe,st[:,:,2],'ob')
plt.plot(AHLe,un[:,:,2],'.k')
plt.yscale('log')
plt.xscale('log')

plt.show()



IPTG=0


'''
AHLrange= np.logspace(-10,-1,100,base=10)
Grange= np.logspace(-6,1,100,base=10)
Grange= np.array([0,100])

Rrange= np.logspace(-6,1,100,base=10)
Rrange= np.array([100,0])



A0=np.ones((len(AHLrange),len(Grange)))*Grange
G0=np.ones((len(AHLrange),len(Grange)))*AHLrange[:,None]
R0=np.ones((len(AHLrange),len(Grange)))*AHLrange[:,None]




g,r,a=Integration(G0,R0,A0,IPTG,par)
print(a.shape)
#plt.plot(y[-2])
plt.plot(AHLrange,g[-2])
plt.yscale('log')
plt.xscale('log')

plt.show()
#sns.heatmap(g[-2]/r[-2],cmap="RdYlGn")# norm=LogNorm())
#plt.show()
'''


'''
IPTGrange= np.array([0,7,10])
AHLrange= np.logspace(-6,1,100,base=10)



IPTG=np.ones((len(AHLrange),len(IPTGrange)))*IPTGrange
AHL=np.ones((len(AHLrange),len(IPTGrange)))*AHLrange[:,None]


#g,r = model_TSL(0,0,AHLrange,IPTGrange,paper_par)
g,r,a=Integration(np.ones((len(AHLrange),len(IPTGrange)))*100,np.ones((len(AHLrange),len(IPTGrange)))*0,AHL,IPTG,par)
g2,r2,a2=Integration(np.ones((len(AHLrange),len(IPTGrange)))*0,np.ones((len(AHLrange),len(IPTGrange)))*100,AHL,IPTG,par)
g3,r3,a3=Integration(np.ones((len(AHLrange),len(IPTGrange)))*0,np.ones((len(AHLrange),len(IPTGrange)))*0,AHL,IPTG,par)






plt.subplot(3,3,1)
plt.plot(AHLrange,g[-2,:,0],'g')
plt.plot(AHLrange,g2[-2,:,0],'g--')
#plt.plot(AHLrange,g3[-2,:,0],'b')

plt.xscale("log")

plt.subplot(3,3,2)
plt.plot(AHLrange,g[-2,:,1],'g')
plt.plot(AHLrange,g2[-2,:,1],'g--')
#plt.plot(AHLrange,g3[-2,:,1],'b')

plt.xscale("log")

plt.subplot(3,3,3)
plt.plot(AHLrange,g[-2,:,2],'g')
plt.plot(AHLrange,g2[-2,:,2],'g--')
#plt.plot(AHLrange,g3[-2,:,2],'b')

plt.xscale("log")

plt.subplot(3,3,4)

plt.plot(AHLrange,r[-2,:,0],'r')
plt.plot(AHLrange,r2[-2,:,0],'r--')
#plt.plot(AHLrange,r3[-2,:,0],'b')

plt.xscale("log")

plt.subplot(3,3,5)
plt.plot(AHLrange,r[-2,:,1],'r')
plt.plot(AHLrange,r2[-2,:,1],'r--')
#plt.plot(AHLrange,r3[-2,:,1],'b')

plt.xscale("log")

plt.subplot(3,3,6)
plt.plot(AHLrange,r[-2,:,2],'r')
plt.plot(AHLrange,r2[-2,:,2],'r--')
#plt.plot(AHLrange,r3[-2,:,2],'b')

plt.xscale("log")

plt.subplot(3,3,7)

plt.plot(AHLrange,a[-2,:,0],'b')
plt.plot(AHLrange,a2[-2,:,0],'b--')
#plt.plot(AHLrange,r3[-2,:,0],'b')

plt.xscale("log")

plt.subplot(3,3,8)
plt.plot(AHLrange,a[-2,:,1],'b')
plt.plot(AHLrange,a2[-2,:,1],'b--')
#plt.plot(AHLrange,r3[-2,:,1],'b')

plt.xscale("log")

plt.subplot(3,3,9)
plt.plot(AHLrange,a[-2,:,2],'b')
plt.plot(AHLrange,a2[-2,:,2],'b--')
#plt.plot(AHLrange,r3[-2,:,2],'b')

plt.xscale("log")

plt.show()
'''


















'''
def solvedfunction(Gi,Ai,par):
    #rewrite the system equation to have only one unknow and to be call with scipy.optimze.brentq
    #the output give a function where when the line reach 0 are a steady states
    R = par['alpha_red']+((par['beta_red']-par['alpha_red'])*np.power(Ai*par['K_ahl_red'],par['n_ahl_red']))/(1+np.power(Ai*par['K_ahl_red'],par['n_ahl_red']))
    R = R / (1 + np.power(Gi*par['K_GREEN'],par['n_GREEN']))
    R = R/ par['delta_red']

    G = par['alpha_green']+((par['beta_green']-par['alpha_green'])*np.power(Ai*par['K_ahl_green'],par['n_ahl_green']))/(1+np.power(Ai*par['K_ahl_green'],par['n_ahl_green']))
    G = G / (1 + np.power(R*par['K_RED'],par['n_RED']))
    G = G / par['delta_green']
    func = G - Gi
    return func 

def solvedfunction2(Gi,Ai,par):
    #rewrite the system equation to have only one unknow and to be call with scipy.optimze.brentq
    #the output give a function where when the line reach 0 are a steady states
    R = par['alpha_red']+((par['beta_red']-par['alpha_red'])*np.power(Ai*par['K_ahl_red'],par['n_ahl_red']))/(1+np.power(Ai*par['K_ahl_red'],par['n_ahl_red']))
    R = R[None,:] / (1 + np.power(Gi*par['K_GREEN'],par['n_GREEN']))[:,None]
    R = R/ par['delta_red']

    G = par['alpha_green']+((par['beta_green']-par['alpha_green'])*np.power(Ai*par['K_ahl_green'],par['n_ahl_green']))/(1+np.power(Ai*par['K_ahl_green'],par['n_ahl_green']))
    G = G / (1 + np.power(R*par['K_RED'],par['n_RED']))
    G = G / par['delta_green']
    func = G - Gi[:,None]
    return func 

def findss(AHL,par):
    #list of fixed par
    #function to find steady state
    #1. find where line reached 0
    Gi=np.arange(0,100,1)
    Gi=np.logspace(-20,5,500,base=10)

    f=solvedfunction2(Gi,AHL,par)

    x=f[1:,:]*f[0:-1,:] #when the output give <0, where is a change in sign, meaning 0 is crossed

    index=np.where(x<0)


    ss=[]
    nNode=2 # number of nodes : X,Y,Z
    nStstate= 4
    ss=np.ones((len(AHL),nStstate,nNode))*np.nan  

    for i,ai in enumerate(index[1]):
    	G=brentq(solvedfunction, Gi[index[0][i]], Gi[index[0][i]+1],args=(AHL[ai],par)) 
    	R = par['alpha_red']+((par['beta_red']-par['alpha_red'])*np.power(AHL[ai]*par['K_ahl_red'],par['n_ahl_red']))/(1+np.power(AHL[ai]*par['K_ahl_red'],par['n_ahl_red']))
    	R = R / (1 + np.power(G*par['K_GREEN'],par['n_GREEN']))
    	R = R/ par['delta_red']        
    	ss= addSSinGoodOrder(ss,np.array([G,R]),ai,0,beforeXvalues=[])

    return ss
    
def addSSinGoodOrder(vector,values,ai,pos,beforeXvalues=[]):

    if ai ==0:
        if np.isnan(vector[ai,pos,0]):
            vector[ai,pos]=values
        else:
            pos=pos+1
            vector=addSSinGoodOrder(vector,values,ai,pos)
    else:
        if len(beforeXvalues) == 0:
            beforeXvalues = vector[ai-1,:,0].copy()
            #print(beforeXvalues)
        if np.all(np.isnan(beforeXvalues)):
            if np.isnan(vector[ai,pos,0]):
                vector[ai,pos]=values
            else:
                 pos=pos+1
                 vector=addSSinGoodOrder(vector,values,ai,pos,beforeXvalues)
        else:
            v1=abs(values[0] - beforeXvalues)
            pos=np.nanargmin(v1)
            if np.isnan(vector[ai,pos,0]):
                vector[ai,pos]=values
            else:
                previousvalues= vector[ai,pos].copy()
               
                v2=abs(previousvalues[0] - beforeXvalues)
                if np.nanmin(v1)<np.nanmin(v2): #if new values closest, takes the place and replace the previous one
                    vector[ai,pos]=values
                    vector=addSSinGoodOrder(vector,previousvalues,ai,pos)
                else:
                    adjusted_beforeXvalues= beforeXvalues.copy()
                    adjusted_beforeXvalues[pos]= np.nan  # remove placed element
                    vector=addSSinGoodOrder(vector,values,ai,pos,adjusted_beforeXvalues)
    return vector 






def jacobianMatrix(ss,A,par):
    JM=np.ones((ss.shape[0],ss.shape[1],2,2))*np.nan 
    G=ss[:,:,0]
    R=ss[:,:,1]
    A=A[:,None]
    dGdg = - par['delta_green'] *G/G
    dGdr = -((par['alpha_green'] - par['beta_green'])*np.power((par['K_ahl_green']*A),par['n_ahl_green'])*par['n_RED']*np.power((par['K_RED']*R),par['n_RED']))
    dGdr =  dGdr/((np.power((par['K_ahl_green']*A),par['n_ahl_green'])+1)*R*np.power((np.power((par['K_RED']*R),par['n_RED'])+1),2))   
   
    dRdg = - ((par['alpha_red'] - par['beta_red'])*np.power((par['K_ahl_red']*A),par['n_ahl_red'])*par['n_GREEN']*np.power((par['K_GREEN']*G),par['n_GREEN']))
    dRdg =  dRdg/((np.power((par['K_ahl_red']*A),par['n_ahl_red'])+1)*G*np.power((np.power((par['K_GREEN']*G),par['n_GREEN'])+1),2))    
    dRdr = - par['delta_red']  *R/R  


    JM[:,:,0,0]=dGdg
    JM[:,:,0,1]=dGdr

    JM[:,:,1,0]=dRdg
    JM[:,:,1,1]=dRdr

    
    return JM



def calculateALL(ARA,parUsed, dummy):
    #sort ss according to their stabilitz
    #create stability list of shape : arabinose x steady x X,Y,Z 
    nNode=2 # number of nodes : X,Y,Z
    nStstate= 4 # number of steady state accepted by. to create the storage array
  #  ss=np.ones((len(ARA),nStstate,nNode))*np.nan 
    eig= np.ones((len(ARA),nStstate,nNode))*np.nan 
    unstable=np.ones((len(ARA),nStstate,nNode))*np.nan
    stable=np.ones((len(ARA),nStstate,nNode))*np.nan
    oscillation=np.ones((len(ARA),nStstate,nNode))*np.nan
    homoclincic=np.ones((len(ARA),nStstate,nNode))*np.nan
    M=np.ones((len(ARA),nStstate,nNode))*np.nan
    m=np.ones((len(ARA),nStstate,nNode))*np.nan


    delta=10e-10 #perturbation from ss

    ss=findss(ARA,parUsed) 
   # print(ss)
    A=jacobianMatrix(ss,ARA,parUsed)

    for i in np.arange(0,ss.shape[0]):
        for j in np.arange(0,ss.shape[1]):
            if np.any(np.isnan(A[i][j]))==False:
                eigvals, eigvecs =np.linalg.eig(A[i][j])
                eig[i,j]=eigvals.real

                if any(eig[i,j]>0):
                    pos=eig[i,j][eig[i,j]>0]
                    if len(pos)==2:
                            if pos[0]-pos[1] == 0:                                
                                init=[ss[i,j,0]+delta,ss[i,j,1]+delta,ss[i,j,2]+delta]
                               # M[i,j],m[i,j] = limitcycle(i,ss,ARA,init,parUsed,dummy)###
                                if np.isnan(M[i,j][0]):
                                    homoclincic[i][j]=ss[i,j] 

                                else:
                                    oscillation[i][j]=ss[i,j]
                            else:
                                unstable[i,j]=ss[i,j]
                    else:
                        unstable[i,j]=ss[i,j]
                else:
                    if np.all(eig[i,j]<0):
                        stable[i,j]=ss[i,j]
                    else:
                       unstable[i,j]=ss[i,j]
    return ss,eig,unstable,stable,oscillation,homoclincic,M,m


def bifurcation_plot(ARA,filename,pars,c):
    sizex=round(np.sqrt(len(pars)))
    sizey=round(np.sqrt(len(pars))+0.5)

    for pi,p in enumerate(pars):
        s,eig,un,st,os,hc,M,m=calculateALL(ARA,p,dummy=pi) 
        plt.subplot(sizex,sizey,pi+1)
        #plt.tight_layout()
        for i in np.arange(0,un.shape[1]):
            plt.plot(ARA,un[:,i,0],'--',c='g',linewidth=1)
            plt.plot(ARA,st[:,i,0],'-g',linewidth=1)
            plt.plot(ARA,un[:,i,1],'--',c='r',linewidth=1)
            plt.plot(ARA,st[:,i,1],'-r',linewidth=1)
            plt.plot(ARA,os[:,i,0],'--b',linewidth=1)
            plt.plot(ARA,hc[:,i,0],'--b',linewidth=1)
            plt.plot(ARA,M[:,i,0],'-b',linewidth=1)
            plt.plot(ARA,m[:,i,0],'-b',linewidth=1)
            plt.fill_between(ARA,M[:,i,0],m[:,i,0],alpha=0.2,facecolor='blue')
        plt.tick_params(axis='both', which='major', labelsize=2)
      #  plt.yscale("log")
        plt.xscale("log")
       # plt.xlim(-4,2)
        plt.show()
  		  #  plt.savefig(filename+"/"+c+'_Bifurcation.pdf', bbox_inches='tight')



AHL=np.logspace(-4,2,1000,base=10)

#print(ss.shape)
#ss,eig,unstable,stable,oscillation,homoclincic,M,m=calculateALL(AHL,par,0)
bifurcation_plot(AHL,'filename',[par],'test')
'''