
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
#datafile="data_percent.txt"


parlist = [

    {'name':'alpha_red', 'lower_limit':0.,'upper_limit':3.},
    {'name':'basal_red', 'lower_limit':1.,'upper_limit':3.},
    {'name':'beta_red', 'lower_limit':2.,'upper_limit':4.},
    {'name':'K_RED', 'lower_limit':-2.0,'upper_limit':3.0},
    {'name':'n_RED', 'lower_limit':0.0,'upper_limit':2.0},
    {'name':'K_ahl_red', 'lower_limit':-2.0,'upper_limit':2.0},
    {'name':'n_ahl_red', 'lower_limit':0.0,'upper_limit':2.0},
    
    
    {'name':'K_GREEN2', 'lower_limit':-5.0,'upper_limit':2.0},

    {'name':'alpha_green', 'lower_limit':1.,'upper_limit':3.},
    {'name':'basal_green', 'lower_limit':2.,'upper_limit':3.},
    {'name':'beta_green', 'lower_limit':1.,'upper_limit':4.},
    {'name':'K_GREEN', 'lower_limit':-2.0,'upper_limit':3.0},
    {'name':'n_GREEN', 'lower_limit':0.,'upper_limit':2.0},
    {'name':'K_ahl_green', 'lower_limit':0.0,'upper_limit':5.0},
    {'name':'n_ahl_green', 'lower_limit':.0,'upper_limit':2.0},
    #{'name':'F_green', 'lower_limit':-4.0,'upper_limit':2.0},
    {'name':'K_IPTG', 'lower_limit':0.0,'upper_limit':5.0},
    #{'name':'K_IPTG2', 'lower_limit':3.0,'upper_limit':8.0}#,
    
    #{'name':'beta_ahl', 'lower_limit':-1.0,'upper_limit':1.0},
    #{'name':'K_ahl', 'lower_limit':-4.0,'upper_limit':1.0},
    #{'name':'n_ahl', 'lower_limit':0.0,'upper_limit':2.0}
  
    
]






#first part on simualtion
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
'''
def model_TSLT(GREENi,REDi,AHLi,IPTG,par):

    par['delta_green']=1
    par['delta_red']=1

    #here to calculate steady state:  we do without diffusion and cell density
 #   GREENi = np.maximum(GREENi - par['cell_green'],0) # fluorescence background on X
 #   REDi = np.maximum(REDi - par['cell_red'],0) # fluorescence background on X

    GREEN = (10**par['alpha_green']+10**par['beta_green']*np.power(AHLi*10**par['K_ahl_green'],par['n_ahl_green']))/(1+np.power(AHLi*10**par['K_ahl_green'],par['n_ahl_green']))
   # GREEN = (10**par['alpha_green']+10**par['beta_green']*np.power(AHLi*10**par['K_ahl_green'],par['n_ahl_green']))/(1+np.power(AHLi*10**par['K_ahl_green'],par['n_ahl_green']))
    GREEN = GREEN[:,None] / (1 + np.power(REDi*10**par['K_RED'],par['n_RED']))
   # GREEN = GREEN - par['delta_green']*GREENi  + 10**par['alpha_green']
    GREEN = GREEN - par['delta_green']*GREENi  

    free_GREENi= GREENi / ( 1+ 10**par['K_IPTG']*IPTG)

    RED = (10**par['alpha_red'] + 10**par['beta_red']*np.power(AHLi*10**par['K_ahl_red'],par['n_ahl_red']))/(1+np.power(AHLi*10**par['K_ahl_red'],par['n_ahl_red']))
  #  RED = (10**par['beta_red']*np.power(AHLi*10**par['K_ahl_red'],par['n_ahl_red']))/(1+np.power(AHLi*10**par['K_ahl_red'],par['n_ahl_red']))
    RED = RED[:,None] / (1 + np.power(free_GREENi*10**par['K_GREEN'],par['n_GREEN']))
    #RED = RED - par['delta_red']*REDi + 10**par['alpha_red']
    RED = RED - par['delta_red']*REDi 

    return GREEN,RED
'''
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

'''

################################3
### dynamic analysis
########################################


   
def solvedfunction(Gi,A,I,par,model):
    #rewrite the system equation to have only one unknow and to be call with scipy.optimze.brentq
    #the output give a function where when the line reach 0 are a steady states
    par['delta_green']=1
    par['delta_red']=1 #1.2
    par['delta_ahl']=1 #1.2


    if model == 'TSLT':

        Gf = Gi / ( 1+ 10**par['K_IPTG']*I)

        R = 10**par['alpha_red'] + ( 10**par['beta_red']*np.power(A*10**par['K_ahl_red'],par['n_ahl_red']))/(1+np.power(A*10**par['K_ahl_red'],par['n_ahl_red']))
        R = R / (1 + np.power(Gf*10**par['K_GREEN'],par['n_GREEN']))  
        R = ( R ) / par['delta_red']  


        G = 10**par['alpha_green'] + ( 10**par['beta_green']*np.power(A*10**par['K_ahl_green'],par['n_ahl_green']))/(1+np.power(A*10**par['K_ahl_green'],par['n_ahl_green'])) 
        G = G / (1 + np.power(R*10**par['K_RED'],par['n_RED'])) 
        G = (G ) / par['delta_green']

        func = G - Gi


    elif model == 'TSXLT':

        Gf = Gi / ( 1+ 10**par['K_IPTG']*I)
        Gfluo = Gi # Gi*10**par['F_green']  + 10**par['basal_green'] #dissociate LacI form GFP

        AHL =  ( 10**par['beta_ahl']*np.power(Gfluo*10**par['K_ahl'],par['n_ahl']))/(1+np.power(Gfluo*10**par['K_ahl'],par['n_ahl'])) 
        AHL = (AHL) / par['delta_ahl']
        A = A + AHL

        R = 10**par['alpha_red'] + ( 10**par['beta_red']*np.power(A*10**par['K_ahl_red'],par['n_ahl_red']))/(1+np.power(A*10**par['K_ahl_red'],par['n_ahl_red']))
        R = R / (1 + np.power(Gf*10**par['K_GREEN'],par['n_GREEN']))  
        R = ( R ) / par['delta_red']  


        G = 10**par['alpha_green'] + ( 10**par['beta_green']*np.power(A*10**par['K_ahl_green'],par['n_ahl_green']))/(1+np.power(A*10**par['K_ahl_green'],par['n_ahl_green'])) 
        G = G / (1 + np.power(R*10**par['K_RED'],par['n_RED'])) 
        G = (G ) / par['delta_green']

        func = G - Gi

    else:
        print("wrong model type, try TSLT or TSXLT. you type: " +model)
        func=0

    return func 


def findss(A,I,par,model):
    #list of fixed par
    #function to find steady state
    #1. find where line reached 0
    ss=[]
    nNode=2 # number of nodes : X,Y,
    if model == 'TSXLT':
        nNode = 3
    nStstate= 5
    nAHL= len(A)
    nIPTG=len(I)
    ss=np.ones((nAHL,nIPTG,nStstate,nNode))*np.nan  
    for ai,a in enumerate(A):
        for iptgi,iptg in enumerate(I):
            Gi=np.arange(0,100,1)
            Gi=np.logspace(-50,10,10000,base=10)
            f=solvedfunction(Gi,a,iptg,par,model)         
            x=f[1:-1]*f[0:-2] #when the output give <0, where is a change in sign, meaning 0 is crossed
            index=np.where(x<0)
            for it,i in enumerate(index[0]):
                if model == 'TSLT':
                    G = brentq(solvedfunction, Gi[i], Gi[i+1],args=(a,iptg,par,model)) #find the value of AHL at 0
                    Gf = G / ( 1+ 10**par['K_IPTG']*iptg)
                    R = 10**par['alpha_red'] + ( 10**par['beta_red']*np.power(a*10**par['K_ahl_red'],par['n_ahl_red']))/(1+np.power(a*10**par['K_ahl_red'],par['n_ahl_red'])) 
                    #K_GREEN change to F_GREEN, try to change dynamic from TETR and mcherry repression by LacI
                    R = R / (1 + np.power(Gf*10**par['K_GREEN'],par['n_GREEN']))
                    #R = R / (1 + np.power(Gf*10**par['K_GREEN2'],par['n_GREEN'])) 
                    R = ( R ) / par['delta_red'] 


                    R = R + 10**par['basal_red'] #autofluo
                    G = G + 10**par['basal_green'] #autofluo
                    ss[ai,iptgi,it]=np.array([G,R])


                if model == 'TSXLT':
                    G = brentq(solvedfunction, Gi[i], Gi[i+1],args=(a,iptg,par,model)) #find the value of AHL at 0
                    Gf = G / ( 1+ 10**par['K_IPTG']*iptg)
                    #Gfluo = G*10**par['F_green'] + 10**par['basal_green'] 

                    AHL =  ( 10**par['beta_ahl']*np.power(Gfluo*10**par['K_ahl'],par['n_ahl']))/(1+np.power(Gfluo*10**par['K_ahl'],par['n_ahl'])) 
                    AHL = (AHL) / par['delta_ahl']
                    a = a + AHL

                    R = 10**par['alpha_red'] + ( 10**par['beta_red']*np.power(a*10**par['K_ahl_red'],par['n_ahl_red']))/(1+np.power(a*10**par['K_ahl_red'],par['n_ahl_red'])) 
                    R = R / (1 + np.power(Gf*10**par['K_GREEN'],par['n_GREEN'])) #+ 10**par['basal_red']
                    R = ( R ) / par['delta_red'] 

                    R = R*10**par['F_red'] + 10**par['basal_red'] #dissociate TetR form mcherry
                    #G = Gfluo

                    R = R + 10**par['basal_red'] #dissociate TetR form mcherry
                    G = G + 10**par['basal_green'] #dissociate LacI form GFP

                    ss[ai,iptgi,it]=np.array([G,R,AHL])

             #   ss[ai,iptgi,it]=np.array([G+par['basal_green'],R+par['basal_red']])
    return ss




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

###########################################



def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

def Get_data(dataname, st='median'):
    path=dataname
    df = pd.read_csv(path,sep='\t' ,header=[0])
    df[df == ' NA'] = np.nan

    t=find_between(dataname, '_','.')
    #t=find_between(dataname, '_','_')
   

    if t=="gated":
        names=df.columns
        df_G=df[df['fluo'] == "GREEN"]
        df_gg=df_G[df_G["minmax"] == "max"]
        df_gr=df_G[df_G["minmax"] == "min"]
        gg = df_gg.pivot(index='AHL', columns='IPTG', values='mean').astype(float)
        gr = df_gr.pivot(index='AHL', columns='IPTG', values='mean').astype(float)

        df_R=df[df['fluo'] == "RED"]
        df_rg=df_R[df_R["minmax"] == "min"]
        df_rr=df_R[df_R["minmax"] == "max"]
        rg = df_rg.pivot(index='AHL', columns='IPTG', values='mean').astype(float)
        rr = df_rr.pivot(index='AHL', columns='IPTG', values='mean').astype(float)

    elif t=="percent":
        names=df.columns
        df_G=df[df['gate'] == 1]
        df_gg=df_G[df_G["sample"] == "G"]
        df_gr=df_G[df_G["sample"] == "R"]
        gg = df_gg.pivot(index='AHL', columns='IPTG', values='mean').astype(float)
        gr = df_gr.pivot(index='AHL', columns='IPTG', values='mean').astype(float)

        df_R=df[df['gate'] == 2]
        df_rg=df_R[df_R["sample"] == "G"]
        df_rr=df_R[df_R["sample"] == "R"]
        rg = df_rg.pivot(index='AHL', columns='IPTG', values='mean').astype(float)
        rr = df_rr.pivot(index='AHL', columns='IPTG', values='mean').astype(float)

    #elif t=="":
    else:
        names=df.columns
        df_G=df[df['fluo'] == "GREEN"]
        df_gg=df_G[df_G["sample"] == "G"]
        df_gr=df_G[df_G["sample"] == "R"]
        gg = df_gg.pivot(index='AHL', columns='IPTG', values=st).astype(float)
        gr = df_gr.pivot(index='AHL', columns='IPTG', values=st).astype(float)

        df_R=df[df['fluo'] == "RED"]
        df_rg=df_R[df_R["sample"] == "G"]
        df_rr=df_R[df_R["sample"] == "R"]
        rg = df_rg.pivot(index='AHL', columns='IPTG', values=st).astype(float)
        rr = df_rr.pivot(index='AHL', columns='IPTG', values=st).astype(float)

    return  gg,gr,rr,rg # gg = green fluo, green sample


###########################################3


def distance(pars,path,modeltype):
    
   # GG,GR,GA,RG,RR,RA = model(pars,totaltime, dt)
    gg,gr,rg,rr=Get_data(path)
    AHL=gg.index.values
    IPTG=gg.columns.values
    
    pars['delta_green']=1
    pars['delta_red']=1
    ss= findss(AHL,IPTG,pars,modeltype)

    M=np.nanmax(ss[:,:,:,:],axis=2)
    m=np.nanmin(ss[:,:,:,:],axis=2)
    
    
    d_green = np.sqrt(np.power(gg.to_numpy() - M[:,:,0],2))
    d_red = np.sqrt(np.power(rr.to_numpy() - M[:,:,1],2))

    d_green2 = np.sqrt(np.power(gr.to_numpy() - m[:,:,0],2))
    d_red2 = np.sqrt(np.power(rg.to_numpy() - m[:,:,1],2))
    
    d=np.nansum((d_green,d_red,d_red2,d_green2))
    #d=np.nansum((d_green,d_green2))
    d=np.nansum((d_red,d_red2))

    return d


#############################################3



