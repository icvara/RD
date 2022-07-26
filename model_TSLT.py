
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


'''
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
'''

parlist = [

    #{'name':'alpha_red', 'lower_limit':-2.,'upper_limit':4.},
    #{'name':'basal_red', 'lower_limit':-2.,'upper_limit':4.},
    #{'name':'beta_red', 'lower_limit':-2.,'upper_limit':4.},
    {'name':'alpha_red', 'lower_limit':0.,'upper_limit':5.},
    {'name':'basal_red', 'lower_limit':-4.,'upper_limit':3.},
    {'name':'beta_red', 'lower_limit':0.,'upper_limit':5.},
    {'name':'K_RED', 'lower_limit':-8.0,'upper_limit':5.0},
    {'name':'n_RED', 'lower_limit':0.0,'upper_limit':4.0},
   # {'name':'delta_red', 'lower_limit':0.0,'upper_limit':1.0},
    {'name':'K_ahl_red', 'lower_limit':-5.0,'upper_limit':4.0},
    {'name':'n_ahl_red', 'lower_limit':0.0,'upper_limit':4.0},
 #   {'name':'cell_red', 'lower_limit':0.0,'upper_limit':1000.0},


    #{'name':'alpha_green', 'lower_limit':-2.,'upper_limit':4.0},
    #{'name':'basal_green', 'lower_limit':-2.,'upper_limit':4.},
    #{'name':'beta_green', 'lower_limit':-2.,'upper_limit':4.0},
    {'name':'alpha_green', 'lower_limit':-1.,'upper_limit':5.},
    {'name':'basal_green', 'lower_limit':-4.,'upper_limit':3.},
    {'name':'beta_green', 'lower_limit':-1.,'upper_limit':5.},
    {'name':'K_GREEN', 'lower_limit':-8.0,'upper_limit':5.0},
    {'name':'n_GREEN', 'lower_limit':0.,'upper_limit':4.0},
   # {'name':'delta_green', 'lower_limit':0.0,'upper_limit':1.0},
    {'name':'K_ahl_green', 'lower_limit':-5.0,'upper_limit':4.0},
    {'name':'n_ahl_green', 'lower_limit':.0,'upper_limit':4.0},
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

   # Gii = np.maximum(Gi - par ['cell_green'],0) # fluorescence background on X
    par['delta_green']=1
    par['delta_red']=1 #1.2

    Gii=Gi
    Gii = np.maximum(Gi - 10**par['basal_green'],0)

    Gf = Gii / ( 1+ 10**par['K_IPTG']*I)

    R = 10**par['alpha_red'] + ( 10**par['beta_red']*np.power(A*10**par['K_ahl_red'],par['n_ahl_red']))/(1+np.power(A*10**par['K_ahl_red'],par['n_ahl_red']))
    #R = (10**par['beta_red'])*np.power(A*10**par['K_ahl_red'],par['n_ahl_red'])/(1+np.power(A*10**par['K_ahl_red'],par['n_ahl_red'])) 
    R = R / (1 + np.power(Gf*10**par['K_GREEN'],par['n_GREEN']))  #+ 10**par['basal_red']
  #  R = ( R + 10**par['alpha_red'] ) / par['delta_red']  
    R = ( R ) / par['delta_red']  #################
 #   R = np.minimum(R + par['cell_red'],par['cell_red'])
   # R = np.maximum(R - par['cell_red'],0) # fluorescence background on X
    R=  np.maximum(R - 10**par['basal_red'],0)
    G = 10**par['alpha_green'] + ( 10**par['beta_green']*np.power(A*10**par['K_ahl_green'],par['n_ahl_green']))/(1+np.power(A*10**par['K_ahl_green'],par['n_ahl_green'])) 
    #G = (10**par['beta_green'])*np.power(A*10**par['K_ahl_green'],par['n_ahl_green'])/(1+np.power(A*10**par['K_ahl_green'],par['n_ahl_green']))
    G = G / (1 + np.power(R*10**par['K_RED'],par['n_RED'])) #+ 10**par['basal_green']
    G = (G ) / par['delta_green']
    #G = (G + 10**par['alpha_green']) / par['delta_green']
    #G =  np.minimum(G + par['cell_green'],par['cell_green'])

    #G =G - np.maximum(Gi - par['cell_green'],0) # fluorescence background on X

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
            Gi=np.logspace(-50,10,10000,base=10)
            f=solvedfunction(Gi,a,iptg,par)         
            x=f[1:-1]*f[0:-2] #when the output give <0, where is a change in sign, meaning 0 is crossed
            index=np.where(x<0)
            for it,i in enumerate(index[0]):
                G = brentq(solvedfunction, Gi[i], Gi[i+1],args=(a,iptg,par)) #find the value of AHL at 0
                #now we have AHL we can find AHL2 ss
              #  Gii = np.maximum(G - par['cell_green'],0) # fluorescence background on X
                #Gii = np.maximum(Gi - par['basal_green'],0)
                Gf = G / ( 1+ 10**par['K_IPTG']*iptg)
                R = 10**par['alpha_red'] + ( 10**par['beta_red']*np.power(a*10**par['K_ahl_red'],par['n_ahl_red']))/(1+np.power(a*10**par['K_ahl_red'],par['n_ahl_red'])) 
                #R = (10**par['beta_red'])*np.power(a*10**par['K_ahl_red'],par['n_ahl_red'])/(1+np.power(a*10**par['K_ahl_red'],par['n_ahl_red'])) 
                R = R / (1 + np.power(Gf*10**par['K_GREEN'],par['n_GREEN'])) #+ 10**par['basal_red']
                #R = ( R + 10**par['alpha_red'] ) / par['delta_red']  
                R = ( R ) / par['delta_red'] 
                R=  np.maximum(R - 10**par['basal_red'],0)
               # R = np.minimum(R + par['cell_red'],par['cell_red'])
  

       # ss.append(np.array([G,R,A]))

                ss[ai,iptgi,it]=np.array([G,R])
             #   ss[ai,iptgi,it]=np.array([G+par['basal_green'],R+par['basal_red']])

    return ss



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



def distance2(pars,path):
    
   # GG,GR,GA,RG,RR,RA = model(pars,totaltime, dt)
    gg,gr,rg,rr=Get_data2(path)
    AHL=gg.index.values
    IPTG=gg.columns.values
    pars['delta_green']=1
    pars['delta_red']=1
    ss= findss(AHL,IPTG,pars)

    M=np.nanmax(ss[:,:,:,:],axis=2)
    m=np.nanmin(ss[:,:,:,:],axis=2)


    d_green = np.nansum(np.power(gg.to_numpy() - M[:,:,0],2))/(len(IPTG)*len(AHL))
    d_red = np.nansum(np.power(rr.to_numpy() - M[:,:,1],2))/(len(IPTG)*len(AHL))

    d_green2 = np.nansum(np.power(rg.to_numpy() - m[:,:,0],2))/(len(IPTG)*len(AHL))
    d_red2 = np.nansum(np.power(gr.to_numpy() - m[:,:,1],2))/(len(IPTG)*len(AHL))

    d=(d_green+d_red+d_green2+d_red2)/4


    return d

def distance4(pars,path,split=False):
    
   # GG,GR,GA,RG,RR,RA = model(pars,totaltime, dt)
    #gmin,gmax,rmin,rmax=Get_data4(path,pars)
    gmin,gmax,rmin,rmax=Get_data4(path)

    AHL=gmin.index.values
    IPTG=gmin.columns.values
    pars['delta_green']=1
    pars['delta_red']=1
    ss= findss(AHL,IPTG,pars)

    M=np.nanmax(ss[:,:,:,:],axis=2)
    m=np.nanmin(ss[:,:,:,:],axis=2)
    '''
    hyst= gmax.to_numpy()-gmin.to_numpy()
    hystss= M[:,:,0]-m[:,:,0]
    d_hyst = np.sqrt(np.nansum(np.power(hyst -hystss,2)))# /(len(IPTG)*len(AHL))
    '''
    
    hyst= gmax.to_numpy()-gmin.to_numpy()
    hyst[hyst<100]=0
    hyst[hyst>1]=1
    
    hyst2= rmax.to_numpy()-rmin.to_numpy()
    hyst2[hyst2<100]=0
    hyst2[hyst2>1]=1
    hysttable=np.array([[0,   0,  0,  0,  0,  0],
                        [0,   0,  0,  1,  1,  1],
                        [0,   0,  1,  1,  1,  1],
                        [0,   0,  1,  1,  1,  1],
                        [0,   1,  1,  1,  1,  1],
                        [0,   1,  1,  1,  0,  0],
                        [1,   1,  1,  1,  0,  0],
                        [1,   1,  1,  0,  0,  0] ] )

                                                         
    hystss = np.count_nonzero(~np.isnan(ss[:,:,:,0]),axis=2)#(ss[:,:,:,:],axis=2)
    hystss[hystss<2]=0
    hystss[hystss>0]=1
   
    hystss2 = np.count_nonzero(~np.isnan(ss[:,:,:,1]),axis=2)#(ss[:,:,:,:],axis=2)
    hystss2[hystss2<2]=0
    hystss2[hystss2>0]=1
    
    d_hyst =np.nansum(1000*np.power(hysttable-hystss2,2)) + np.nansum(1000*np.power(hysttable-hystss,2))
    #hystss2= M[:,:,1]-m[:,:,1]
    #d_hyst2 = np.sqrt(np.nansum(np.power(hyst2 -hystss2,2)))# /(len(IPTG)*len(AHL))
    
    t=np.sqrt(np.power(gmax.to_numpy() - M[:,:,0],2))
    #t[t<25]=t[t<25]*.1
    d_green=np.nansum(t)# /(len(IPTG)*len(AHL))  
    
    t=np.sqrt(np.power(gmin.to_numpy() - m[:,:,0],2))
    #t[t<25]=t[t<25]*.1
    d_green2=np.nansum(t)# /(len(IPTG)*len(AHL))  
    
    t=np.sqrt(np.power(rmax.to_numpy() - M[:,:,1],2))
    #t[t<25]=t[t<25]*.1
    d_red=np.nansum(t)# /(len(IPTG)*len(AHL))  
    
    t=np.sqrt(np.power(rmin.to_numpy() - m[:,:,1],2))
    #t[t<25]=t[t<25]*.1
    d_red2=np.nansum(t)# /(len(IPTG)*len(AHL))  
 
    d=(d_green+d_red+d_green2+d_red2)
    dtot=d #+d_hyst

    if split:
      d_hystgreen = 1000*np.power(hysttable-hystss2,2)
      d_hystred= 1000*np.power(hysttable-hystss,2)
      d_green=np.sqrt(np.power(gmax.to_numpy() - M[:,:,0],2))
      d_green2=np.sqrt(np.power(gmin.to_numpy() - m[:,:,0],2))
      d_red=np.sqrt(np.power(rmax.to_numpy() - M[:,:,1],2))# /(len(IPTG)*len(AHL))  
      d_red2=np.sqrt(np.power(rmin.to_numpy() - m[:,:,1],2))
      dtot=np.array([d_green,d_green2,d_hystgreen,d_red,d_red2,d_hystred])
      
    
    return dtot # +4*(d_hyst+d_hyst2)
    
   

def distance3(pars,path):
    
   # GG,GR,GA,RG,RR,RA = model(pars,totaltime, dt)
    Ggg,Ggr,Grg,Grr, Rgg,Rgr,Rrg,Rrr = Get_data3(path)
   
    #gate,fluo,sample
    AHL=Ggg.index.values
    IPTG=Ggg.columns.values
    pars['delta_green']=1
    pars['delta_red']=1
    ss= findss(AHL,IPTG,pars)

    M=np.nanmax(ss[:,:,:,:],axis=2)
    m=np.nanmin(ss[:,:,:,:],axis=2)

    d_green = np.nansum(np.power(Ggg.to_numpy() - M[:,:,0],2))/(len(IPTG)*len(AHL))
    
    d_red = np.nansum(np.power(Rrr.to_numpy() - M[:,:,1],2))/(len(IPTG)*len(AHL))

    d_green2 = np.nansum(np.power(Rgg.to_numpy() - m[:,:,0],2))/(len(IPTG)*len(AHL))
    d_red2 = np.nansum(np.power(Grr.to_numpy() - m[:,:,1],2))/(len(IPTG)*len(AHL))


    d_green3 = np.nansum(np.power(Rrg.to_numpy() - m[:,:,0],2))/(len(IPTG)*len(AHL))
    d_red3 = np.nansum(np.power(Rgr.to_numpy() - M[:,:,1],2))/(len(IPTG)*len(AHL))

    d_green4 = np.nansum(np.power(Grg.to_numpy() - M[:,:,0],2))/(len(IPTG)*len(AHL))
    d_red4 = np.nansum(np.power(Ggr.to_numpy() - m[:,:,1],2))/(len(IPTG)*len(AHL))

    d=(d_green+d_red+d_green2+d_red2+d_green3+d_red3+d_green4+d_red4)/8


    return d
   

def Get_data(path):
    path=datafile
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

def Get_data2(dataname):
    path=dataname
    df = pd.read_csv(path,sep='\t' ,header=[0])
    df[df == ' NA'] = np.nan

    df_green=df[df["sample"] == "G"]
    df_gg = df_green[df_green.iloc[:,3] == ' GREEN']
    df_gr = df_green[df_green.iloc[:,3] == ' RED']

    df_red=df[df["sample"] == "R"]
    df_rg = df_red[df_red.iloc[:,3] == ' GREEN']
    df_rr = df_red[df_red.iloc[:,3] == ' RED']

    gg=df_gg.pivot(index=' AHL', columns=' IPTG', values=' mean').astype(float)
    gr=df_gr.pivot(index=' AHL', columns=' IPTG', values=' mean').astype(float)
    rr=df_rr.pivot(index=' AHL', columns=' IPTG', values=' mean').astype(float)
    rg=df_rg.pivot(index=' AHL', columns=' IPTG', values=' mean').astype(float)
    return gg,gr,rg,rr

def Get_data4(dataname):
    path=dataname
    df = pd.read_csv(path,sep='\t' ,header=[0])
    df[df == ' NA'] = np.nan


    df_green=df[df[" fluo"] == " GREEN"]
    df_gmin = df_green[df_green.iloc[:,3] == ' minimun']
    df_gmax = df_green[df_green.iloc[:,3] == ' maximun']

    df_red=df[df[" fluo"] == " RED"]
    df_rmin = df_red[df_red.iloc[:,3] == ' minimun']
    df_rmax = df_red[df_red.iloc[:,3] == ' maximun']



    gmin=df_gmin.pivot(index='AHL', columns=' IPTG', values=' median').astype(float)
    gmax=df_gmax.pivot(index='AHL', columns=' IPTG', values=' median').astype(float)
    rmin=df_rmin.pivot(index='AHL', columns=' IPTG', values=' median').astype(float)
    rmax=df_rmax.pivot(index='AHL', columns=' IPTG', values=' median').astype(float)

    '''
    gmin = gmin-par['basal_green']
    gmax= gmax-par['basal_green']
    rmin = rmin-par['basal_red']
    rmax= rmax-par['basal_red']
    '''
    return gmin,gmax,rmin,rmax
    
def Get_data5(dataname):
    path=dataname
    df = pd.read_csv(path,sep='\t' ,header=[0])
    df[df == ' NA'] = np.nan
    
    

    df_green=df[df[" fluo"] == " GREEN"]
 
    df_gmin = df_green[df_green.iloc[:,0] == 'R']
    df_gmax = df_green[df_green.iloc[:,0] == 'G']
  
    df_red=df[df[" fluo"] == " RED"]
    df_rmin = df_red[df_red.iloc[:,0] == 'G']
    df_rmax = df_red[df_red.iloc[:,0] == 'R']

    gmin=df_gmin.pivot(index=' AHL', columns=' IPTG', values=' maximun').astype(float)
    gmax=df_gmax.pivot(index=' AHL', columns=' IPTG', values=' maximun').astype(float)
    rmin=df_rmin.pivot(index=' AHL', columns=' IPTG', values=' maximun').astype(float)
    rmax=df_rmax.pivot(index=' AHL', columns=' IPTG', values=' maximun').astype(float)

    '''
    gmin = gmin-par['basal_green']
    gmax= gmax-par['basal_green']
    rmin = rmin-par['basal_red']
    rmax= rmax-par['basal_red']
    '''
    return gmin,gmax,rmin,rmax

def Get_data3(dataname):
    path=dataname
    df = pd.read_csv(path,sep='\t' ,header=[0])
    df[df == ' NA'] = np.nan


    df_G=df[df[" gate"] == 1]
    df_greenG=df_G[df_G["sample"] == "G"]
    df_Ggg = df_greenG[df_greenG.iloc[:,3] == ' GREEN']
    df_Ggr = df_greenG[df_greenG.iloc[:,3] == ' RED']

    df_redG=df_G[df_G["sample"] == "R"]
    df_Grg = df_redG[df_redG.iloc[:,3] == ' GREEN']
    df_Grr = df_redG[df_redG.iloc[:,3] == ' RED']


    df_R=df[df[" gate"] == 2]
    df_greenR=df_R[df_R["sample"] == "G"]
    df_Rgg = df_greenR[df_greenR.iloc[:,3] == ' GREEN']
    df_Rgr = df_greenR[df_greenR.iloc[:,3] == ' RED']

    df_redR=df_R[df_R["sample"] == "R"]
    df_Rrg = df_redR[df_redR.iloc[:,3] == ' GREEN']
    df_Rrr = df_redR[df_redR.iloc[:,3] == ' RED']

    Ggg=df_Ggg.pivot(index=' AHL', columns=' IPTG', values=' mean').astype(float)
    Ggr=df_Ggr.pivot(index=' AHL', columns=' IPTG', values=' mean').astype(float)
    Grr=df_Grr.pivot(index=' AHL', columns=' IPTG', values=' mean').astype(float)
    Grg=df_Grg.pivot(index=' AHL', columns=' IPTG', values=' mean').astype(float)

    Rgg=df_Rgg.pivot(index=' AHL', columns=' IPTG', values=' mean').astype(float)
    Rgr=df_Rgr.pivot(index=' AHL', columns=' IPTG', values=' mean').astype(float)
    Rrr=df_Rrr.pivot(index=' AHL', columns=' IPTG', values=' mean').astype(float)
    Rrg=df_Rrg.pivot(index=' AHL', columns=' IPTG', values=' mean').astype(float)



    return Ggg,Ggr,Grg,Grr, Rgg,Rgr,Rrg,Rrr


#############################################3


#print(Get_data5("data_median_gated_maxmin2.txt"))

#init_RED = [rg.iloc[7,5],rr.iloc[7,5],0]
#init_GREEN= [gg.iloc[0,0],gr.iloc[0,0],0]


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

