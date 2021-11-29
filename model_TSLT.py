
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

    {'name':'alpha_red', 'lower_limit':0.0,'upper_limit':1000.0},
    {'name':'beta_red', 'lower_limit':100.0,'upper_limit':1000.0},
    {'name':'K_RED', 'lower_limit':0.0,'upper_limit':100.0},
    {'name':'n_RED', 'lower_limit':1.0,'upper_limit':4.0},
    {'name':'delta_red', 'lower_limit':0.0,'upper_limit':1.0},
    {'name':'K_ahl_red', 'lower_limit':0.0,'upper_limit':100.0},
    {'name':'n_ahl_red', 'lower_limit':0.0,'upper_limit':4.0},
    {'name':'cell_red', 'lower_limit':0.0,'upper_limit':500.0},


    {'name':'alpha_green', 'lower_limit':0.0,'upper_limit':1000.0},
    {'name':'beta_green', 'lower_limit':100.0,'upper_limit':1000.0},
    {'name':'K_GREEN', 'lower_limit':1.0,'upper_limit':100.0},
    {'name':'n_GREEN', 'lower_limit':1.0,'upper_limit':4.0},
    {'name':'delta_green', 'lower_limit':0.0,'upper_limit':1.0},
    {'name':'K_ahl_green', 'lower_limit':0.0,'upper_limit':100.0},
    {'name':'n_ahl_green', 'lower_limit':1.0,'upper_limit':4.0},
    {'name':'K_IPTG', 'lower_limit':0.0,'upper_limit':100.0},
    {'name':'cell_green', 'lower_limit':0.0,'upper_limit':500.0}


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

    #here to calculate steady state:  we do without diffusion and cell density
    GREENi = np.maximum(GREENi - par['cell_green'],0) # fluorescence background on X
    REDi = np.maximum(REDi - par['cell_red'],0) # fluorescence background on X

    GREEN = (par['beta_green']*np.power(AHLi*par['K_ahl_green'],par['n_ahl_green']))/(1+np.power(AHLi*par['K_ahl_green'],par['n_ahl_green']))
    GREEN = GREEN[:,None] / (1 + np.power(REDi*par['K_RED'],par['n_RED']))
    GREEN = GREEN - par['delta_green']*GREENi  + par['alpha_green']

    free_GREENi= GREENi / ( 1+ par['K_IPTG']*IPTG)

    RED = (par['beta_red']*np.power(AHLi*par['K_ahl_red'],par['n_ahl_red']))/(1+np.power(AHLi*par['K_ahl_red'],par['n_ahl_red']))
    RED = RED[:,None] / (1 + np.power(free_GREENi*par['K_GREEN'],par['n_GREEN']))
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



def distance(pars,totaltime=tt, dt=dtt):
    GG,GR,GA,RG,RR,RA = model(pars,totaltime, dt)

    d_green_1= np.nansum(np.power(gg.to_numpy() - GG[-2,:,:],2))/(len(IPTG)*len(AHL))
    d_green_2= np.nansum(np.power(gr.to_numpy() - GR[-2,:,:],2))/(len(IPTG)*len(AHL))
    d_red_1= np.nansum(np.power(rg.to_numpy() - RG[-2,:,:],2))/(len(IPTG)*len(AHL))
    d_red_2= np.nansum(np.power(rr.to_numpy() - RR[-2,:,:],2))/(len(IPTG)*len(AHL))
    d_final= d_green_1 + d_green_2 + d_red_1 + d_red_2
    d_final=d_final/4




    return d_final



def model(pars,totaltime=tt, dt=dtt):
    print(IPTG)
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

