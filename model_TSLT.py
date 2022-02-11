
## equation for TSLT

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import argrelextrema
import numpy as np
from scipy import optimize
from scipy.optimize import brentq



dtt=0.1
tt=120 #totaltime



parlist = [

    {'name':'alpha_red', 'lower_limit':0.0,'upper_limit':10000.0},
    {'name':'beta_red', 'lower_limit':100.0,'upper_limit':10000.0},
    {'name':'K_RED', 'lower_limit':-5.0,'upper_limit':5.0},
    {'name':'n_RED', 'lower_limit':1.0,'upper_limit':4.0},
   # {'name':'delta_red', 'lower_limit':0.0,'upper_limit':1.0},
    {'name':'K_ahl_red', 'lower_limit':-2.0,'upper_limit':5.0},
    {'name':'n_ahl_red', 'lower_limit':0.0,'upper_limit':4.0},
 #   {'name':'cell_red', 'lower_limit':0.0,'upper_limit':1000.0},


    {'name':'alpha_green', 'lower_limit':0.0,'upper_limit':10000.0},
    {'name':'beta_green', 'lower_limit':100.0,'upper_limit':10000.0},
    {'name':'K_GREEN', 'lower_limit':-5.0,'upper_limit':5.0},
    {'name':'n_GREEN', 'lower_limit':1.0,'upper_limit':4.0},
   # {'name':'delta_green', 'lower_limit':0.0,'upper_limit':1.0},
    {'name':'K_ahl_green', 'lower_limit':-2.0,'upper_limit':5.0},
    {'name':'n_ahl_green', 'lower_limit':1.0,'upper_limit':4.0},
    {'name':'K_IPTG', 'lower_limit':-2.0,'upper_limit':5.0}
  #  {'name':'cell_green', 'lower_limit':0.0,'upper_limit':1000.0}


#    {'name':'beta_ahl', 'lower_limit':0.0,'upper_limit':0.0},
#    {'name':'K_ahl', 'lower_limit':0.0,'upper_limit':0.0},
#    {'name':'n_ahl', 'lower_limit':0.0,'upper_limit':0.0},
#    {'name':'delta_ahl', 'lower_limit':0.0,'upper_limit':0.0}

]








########################################################
'''
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
'''
def model_TSLT(GREENi,REDi,AHLi,IPTG,par):

    par['delta_green']=1
    par['delta_red']=1

    #here to calculate steady state:  we do without diffusion and cell density
 #   GREENi = np.maximum(GREENi - par['cell_green'],0) # fluorescence background on X
 #   REDi = np.maximum(REDi - par['cell_red'],0) # fluorescence background on X

    GREEN = (par['beta_green']*np.power(AHLi*10**par['K_ahl_green'],par['n_ahl_green']))/(1+np.power(AHLi*10**par['K_ahl_green'],par['n_ahl_green']))
    GREEN = GREEN[:,None] / (1 + np.power(REDi*10**par['K_RED'],par['n_RED']))
    GREEN = GREEN - par['delta_green']*GREENi  + par['alpha_green']

    free_GREENi= GREENi / ( 1+ 10**par['K_IPTG']*IPTG)

    RED = (par['beta_red']*np.power(AHLi*10**par['K_ahl_red'],par['n_ahl_red']))/(1+np.power(AHLi*10**par['K_ahl_red'],par['n_ahl_red']))
    RED = RED[:,None] / (1 + np.power(free_GREENi*10**par['K_GREEN'],par['n_GREEN']))
    RED = RED - par['delta_red']*REDi + par['alpha_red']

    return GREEN,RED
'''
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
'''


def Integration(G0,R0,A0,IPTG,p,totaltime=500,dt=0.1):   
    #U0 need to be of shape (totaltime,nx,ny)
    Gi=G0
    Ri=R0

    Ai=np.ones((len(A0),len(IPTG)))*np.array(A0)[:,None]

    G = np.zeros(((round(totaltime/dt+0.5)+1),len(A0),len(IPTG)))
    R = np.zeros(((round(totaltime/dt+0.5)+1),len(A0),len(IPTG)))
    A= np.zeros(((round(totaltime/dt+0.5)+1),len(A0),len(IPTG)))

    G[0]=G0
    R[0]=R0
    A[0]=np.ones((len(A0),len(IPTG)))*np.array(A0)[:,None]

    t=dt
    i=1
    while t < totaltime:
      #  g,r = model_TSL(Gi,Ri,U0,IPTG,p)
        g,r = model_TSLT(Gi,Ri,np.array(A0),np.array(IPTG),p)
        a=np.zeros((len(A0),len(IPTG)))

      # g,r,a = model_TSXLT(Gi,Ri,Ai,IPTG,p)


        Gi = Gi + g*dt
        Ri = Ri + r*dt
        Ai= Ai + a*dt 
        G[i]=Gi
        R[i]=Ri
        A[i]=Ai
        t=t+dt
        i=i+1
    return G, R,A


def model(pars,totaltime=tt, dt=dtt):
    #init green state
    Gi=np.ones((len(AHL),len(IPTG)))*init_GREEN[0]
    Ri=np.ones((len(AHL),len(IPTG)))*init_GREEN[1]
    #Ai=np.ones(len(AHL))*init_GREEN[2]
    Ai = AHL
    GG,GR,GA = Integration(Gi,Ri,Ai,IPTG,pars,totaltime,dt)

    #init red state
    Gi=np.ones((len(AHL),len(IPTG)))*init_RED[0]
    Ri=np.ones((len(AHL),len(IPTG)))*init_RED[1]
    #Ai=np.ones(len(AHL))*init_RED[2]
    Ai = AHL
    RG,RR,RA = Integration(Gi,Ri,Ai,IPTG,pars,totaltime,dt)

    return GG,GR,GA,RG,RR,RA



################################3
### dynamic analysis
########################################


   
def solvedfunction(Gi,A,I,par):
    #rewrite the system equation to have only one unknow and to be call with scipy.optimze.brentq
    #the output give a function where when the line reach 0 are a steady states

   # Gii = np.maximum(Gi - par['cell_green'],0) # fluorescence background on X

    Gii=Gi

    Gf = Gii / ( 1+ 10**par['K_IPTG']*I)

    R = (par['beta_red'])*np.power(A*10**par['K_ahl_red'],par['n_ahl_red'])/(1+np.power(A*10**par['K_ahl_red'],par['n_ahl_red'])) 
    R = R / (1 + np.power(Gf*10**par['K_GREEN'],par['n_GREEN']))
    R = ( R + par['alpha_red'] ) / par['delta_red']  
 #   R = np.minimum(R + par['cell_red'],par['cell_red'])
   # R = np.maximum(R - par['cell_red'],0) # fluorescence background on X

    G = (par['beta_green'])*np.power(A*10**par['K_ahl_green'],par['n_ahl_green'])/(1+np.power(A*10**par['K_ahl_green'],par['n_ahl_green']))
    G = G / (1 + np.power(R*10**par['K_RED'],par['n_RED']))
    G = (G + par['alpha_green']) / par['delta_green']
  #  G =  np.minimum(G + par['cell_green'],par['cell_green'])

   # G =G - np.maximum(Gi - par['cell_green'],0) # fluorescence background on X


    func = G - Gi

    return func 


def findss(A,I,par):

    #list of fixed par
    #function to find steady state
    #1. find where line reached 0

    ss=[]
    nNode=2 # number of nodes : X,Y,Z
    nStstate= 5
    nAHL= len(A)
    nIPTG=len(I)
    ss=np.ones((nAHL,nIPTG,nStstate,nNode))*np.nan  
    for ai,a in enumerate(A):
        for iptgi,iptg in enumerate(I):
            Gi=np.arange(0,100,1)
            Gi=np.logspace(-50,5,1000,base=10)
            f=solvedfunction(Gi,a,iptg,par)
            x=f[1:-1]*f[0:-2] #when the output give <0, where is a change in sign, meaning 0 is crossed
            index=np.where(x<0)
            for it,i in enumerate(index[0]):
                G = brentq(solvedfunction, Gi[i], Gi[i+1],args=(a,iptg,par)) #find the value of AHL at 0
                #now we have AHL we can find AHL2 ss
              #  Gii = np.maximum(G - par['cell_green'],0) # fluorescence background on X
                Gf = G / ( 1+ 10**par['K_IPTG']*iptg)
                R = (par['beta_red'])*np.power(a*10**par['K_ahl_red'],par['n_ahl_red'])/(1+np.power(a*10**par['K_ahl_red'],par['n_ahl_red'])) 
                R = R / (1 + np.power(Gf*10**par['K_GREEN'],par['n_GREEN']))
                R = ( R + par['alpha_red'] ) / par['delta_red']  
               # R = np.minimum(R + par['cell_red'],par['cell_red'])
  

       # ss.append(np.array([G,R,A]))

                ss[ai,iptgi,it]=np.array([G,R])

    return ss

def ssmodel(GREENi,REDi,AHLi,IPTG,par):
    #here to calculate steady state:  we do without diffusion and cell density

    #here to calculate steady state:  we do without diffusion and cell density
    GREENi = np.maximum(GREENi - par['cell_green'],0) # fluorescence background on X
    REDi = np.maximum(REDi - par['cell_red'],0) # fluorescence background on X

    GREEN = (par['beta_green']*np.power(AHLi*10**par['K_ahl_green'],par['n_ahl_green']))/(1+np.power(AHLi*10**par['K_ahl_green'],par['n_ahl_green']))
    GREEN = GREEN / (1 + np.power(REDi*10**par['K_RED'],par['n_RED']))
    GREEN = GREEN - par['delta_green']*GREENi  + par['alpha_green']

    free_GREENi= GREENi / ( 1+ 10**par['K_IPTG']*IPTG)

    RED = (par['beta_red']*np.power(AHLi*10**par['K_ahl_red'],par['n_ahl_red']))/(1+np.power(AHLi*10**par['K_ahl_red'],par['n_ahl_red']))
    RED = RED / (1 + np.power(free_GREENi*10**par['K_GREEN'],par['n_GREEN']))
    RED = RED - par['delta_red']*REDi + par['alpha_red']

    return GREEN,RED




def jacobianMatrix(G,R,A,I,par): #function nto finished
    # [ DG/dg   DG/dr   ]
    # [ DR/dg   DR/dr   ]

    G = np.maximum(G - par['cell_green'],0) # fluorescence background on X
    R = np.maximum(R - par['cell_red'],0)
  
    #need to double check , some mistake are here
    dGdg = - par['delta_green']    
    dGdr = -(par['beta_green']*np.power((10**par['K_ahl_green']*A),par['n_ahl_green'])*par['n_RED']*np.power((10**par['K_RED']*R),par['n_RED']))
    dGdr =  dGdr/((np.power((10**par['K_ahl_green']*A),par['n_ahl_green'])+1)*R*np.power((np.power((10**par['K_RED']*R),par['n_RED'])+1),2))   
 
    dRdg = - (par['beta_red']*np.power((10**par['K_ahl_red']*A),par['n_ahl_red'])*par['n_GREEN']*np.power((10**par['K_GREEN']*G)/(I*10**par['K_IPTG']+1),par['n_GREEN']))
    dRdg =  dRdg/((np.power((10**par['K_ahl_red']*A),par['n_ahl_red'])+1)*G*np.power((np.power((10**par['K_GREEN']*G)/(I*10**par['K_IPTG']+1),par['n_GREEN'])+1),2))    
    dRdr = - par['delta_red']     

    A = np.array([[dGdg,dGdr],[dRdg,dRdr]])
    
    return A


def approximateJacob(G,R,A,I,par): #function nto finished
    #allow to check if jacobian matrix derivate are correctly written
    
    delta=10e-5
    g,r = ssmodel(G,R,A,I,par)
    dgdg= (ssmodel(G+delta,R,A,I,par)[0] - g)/delta
    dgdr= (ssmodel(G,R+delta,A,I,par)[0] - g)/delta
    drdg= (ssmodel(G+delta,R,A,I,par)[1] - r)/delta
    drdr= (ssmodel(G,R+delta,A,I,par)[1] - r)/delta
    A=np.array(([dgdg,dgdr],[drdg,drdr]))

    return A

def getEigen(G,R,A,I,par):
    J= jacobianMatrix(G,R,A,I,par)
    eigvals, eigvecs =np.linalg.eig(J)
    sse=eigvals.real
    return sse #, np.trace(A), np.linalg.det(A)

###########################################3


def distance(pars,totaltime=tt, dt=dtt):
    GG,GR,GA,RG,RR,RA = model(pars,totaltime, dt)

    d_green_1= np.nansum(np.power(gg.to_numpy() - GG[-2,:,:],2))/(len(IPTG)*len(AHL))
    d_green_2= np.nansum(np.power(gr.to_numpy() - GR[-2,:,:],2))/(len(IPTG)*len(AHL))
    d_red_1= np.nansum(np.power(rg.to_numpy() - RG[-2,:,:],2))/(len(IPTG)*len(AHL))
    d_red_2= np.nansum(np.power(rr.to_numpy() - RR[-2,:,:],2))/(len(IPTG)*len(AHL))
    d_final= d_green_1 + d_green_2 + d_red_1 + d_red_2
    d_final=d_final/4
    return d_final



def distance2(pars):
   # GG,GR,GA,RG,RR,RA = model(pars,totaltime, dt)
    pars['delta_green']=1
    pars['delta_red']=1
    ss= findss(AHL,IPTG,pars)

    m=np.nanmax(ss[:,:,:,:],axis=2)

    d_green = np.nansum(np.power(gg.to_numpy() - m[:,:,0],2))/(len(IPTG)*len(AHL))
    d_red = np.nansum(np.power(rr.to_numpy() - m[:,:,1],2))/(len(IPTG)*len(AHL))

    d=(d_green+d_red)/2


    return d

    

def Get_data():
    path='data.txt'
    df = pd.read_csv(path,sep='\t' ,header=[0])
    df[df == ' NA'] = np.nan
    sub_df=df[["sample"," AHL"," IPTG"," m_GREEN"," m_RED"]]


    df_green=sub_df[sub_df["sample"]=="N"]

    gg=df_green.pivot(index=' AHL', columns=' IPTG', values=' m_GREEN').astype(float)
    gr=df_green.pivot(index=' AHL', columns=' IPTG', values=' m_RED').astype(float)
    df_red=sub_df[sub_df["sample"]=="R"]
    rg=df_red.pivot(index=' AHL', columns=' IPTG', values=' m_GREEN').astype(float)
    rr=df_red.pivot(index=' AHL', columns=' IPTG', values=' m_RED').astype(float)
   
 #   d=d1[" m_GREEN"]
  #  gg = d.to_numpy().reshape(8,6)


    return gg,gr,rg,rr

#############################################3


gg,gr,rg,rr=Get_data()
AHL=gg.index.values
IPTG=gg.columns.values
init_RED = [rg.iloc[7,5],rr.iloc[7,5],0]
init_GREEN= [gg.iloc[0,0],gr.iloc[0,0],0]





#print(gg-m_gg[-2])

'''
plt.subplot(2,2,1)
sns.heatmap(gg,cmap="Greens")
plt.subplot(2,2,2)
sns.heatmap(gr,cmap="Reds")
plt.subplot(2,2,3)
sns.heatmap(rg,cmap="Greens")
plt.subplot(2,2,4)
sns.heatmap(rr,cmap="Reds")
plt.show()
'''
'''
plt.subplot(2,2,1)
sns.heatmap(gg,cmap="Greens")
plt.subplot(2,2,2)
sns.heatmap(m_gg[-2],cmap="Reds")
plt.subplot(2,2,3)
sns.heatmap(gg-m_gg[-2],cmap="Greens")
plt.show()
'''

