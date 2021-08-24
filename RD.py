#RD system

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.linalg import eig
import random
#from scipy.stats import norm, uniform, multivariate_normal
import multiprocessing
import time
from functools import partial
from scipy import optimize
from scipy.optimize import brentq
import pandas as pd


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
 #   {'name' : 'delta_green', 'lower_limit':1.0,'upper_limit':1.0},
    {'name' : 'K_GREEN', 'lower_limit':0.0,'upper_limit':100.0},
    {'name' : 'n_GREEN', 'lower_limit':0.5,'upper_limit':2.0},

    #RED
    {'name' : 'K_ahl_red', 'lower_limit':0.0,'upper_limit':100.0},
    {'name' : 'n_ahl_red', 'lower_limit':0.5,'upper_limit':2.0},
    {'name' : 'alpha_red', 'lower_limit':0.0,'upper_limit':50.0},
    {'name' : 'beta_red', 'lower_limit':0.0,'upper_limit':100.0},
 #   {'name' : 'delta_red', 'lower_limit':1.0,'upper_limit':1.0},
    {'name' : 'K_RED', 'lower_limit':0.0,'upper_limit':100.0},
    {'name' : 'n_RED', 'lower_limit':0.5,'upper_limit':2.0},

    #AHL
    {'name' : 'K_ahl', 'lower_limit':0.0,'upper_limit':100.0},
    {'name' : 'n_ahl', 'lower_limit':0.5,'upper_limit':2.0},
    {'name' : 'beta_ahl', 'lower_limit':0.0,'upper_limit':100.0},
  #  {'name' : 'delta_ahl', 'lower_limit':1.0,'upper_limit':1.0},
  #  {'name' : 'D_ahl', 'lower_limit':0.01,'upper_limit':1.0},
]

tt = 500 #totaltime
h = 10 #10
w= 0.3
maxD= 1.1
d = 0.1 #interval size
d2 = d*d
dt = d2 * d2 / (2 * maxD * (d2 + d2)) #interval time according to diffusion and interval dist
nx, ny = round(w/d), round(h/d)


def addfixedpar(par):
    #list of fixed par
    par['delta_red']=1
    par['delta_green']=1
    par['delta_ahl']=1
    par['D_ahl']=1



    return par

def choosepar(parlist):
    #choose random par in the defined range
    samplepar=[]
    for ipar,par in enumerate(parlist):
        samplepar.append(random.uniform(par['lower_limit'], par['upper_limit']))
   # p=pars_to_dict(samplepar,parlist)
    return np.array(samplepar)


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
    par=addfixedpar(par)
    
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


def solvedfunction(Gi,par):
    #rewrite the system equation to have only one unknow and to be call with scipy.optimze.brentq
    #the output give a function where when the line reach 0 are a steady states

    A= (par['beta_ahl']*np.power(Gi*par['K_ahl'],par['n_ahl']))/(1+np.power(Gi*par['K_ahl'],par['n_ahl']))
    A= A/par['delta_ahl']

    R = par['alpha_red']+((par['beta_red']-par['alpha_red'])*np.power(A*par['K_ahl_red'],par['n_ahl_red']))/(1+np.power(A*par['K_ahl_red'],par['n_ahl_red']))
    R = R / (1 + np.power(Gi*par['K_GREEN'],par['n_GREEN']))
    R = R/ par['delta_red']

    G = par['alpha_green']+((par['beta_green']-par['alpha_green'])*np.power(A*par['K_ahl_green'],par['n_ahl_green']))/(1+np.power(A*par['K_ahl_green'],par['n_ahl_green']))
    G = G / (1 + np.power(R*par['K_RED'],par['n_RED']))
    G = G / par['delta_green']

    func = G - Gi

    return func


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
    
    par=addfixedpar(par)
    

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


    A = np.array([[dGdg,dGdr,dGda],[dRdg,dRdr,dRda],[dAdg,dAdr,dAda]])
    
    return A

def approximateJacob(G,R,A,par):
    par=addfixedpar(par)
    delta=10e-5
    density=1
    g,r,a =model(G,R,A,d,density,par, isdiffusion=False,oneD =True)

    dGdg = (model(G+delta,R,A,d,density,par, isdiffusion=False,oneD =True)[0] - g ) /delta
    dGdr = (model(G,R+delta,A,d,density,par, isdiffusion=False,oneD =True)[0] - g ) /delta
    dGda = (model(G,R,A+delta,d,density,par, isdiffusion=False,oneD =True)[0] - g ) /delta
   
    dRdg = (model(G+delta,R,A,d,density,par, isdiffusion=False,oneD =True)[1] - r ) /delta
    dRdr = (model(G,R+delta,A,d,density,par, isdiffusion=False,oneD =True)[1] - r ) /delta
    dRda = (model(G,R,A+delta,d,density,par, isdiffusion=False,oneD =True)[1] - r ) /delta

    dAdg = (model(G+delta,R,A,d,density,par, isdiffusion=False,oneD =True)[2] - a ) /delta
    dAdr = (model(G,R+delta,A,d,density,par, isdiffusion=False,oneD =True)[2] - a ) /delta
    dAda = (model(G,R,A+delta,d,density,par, isdiffusion=False,oneD =True)[2] - a ) /delta

    A = np.array([[dGdg,dGdr,dGda],[dRdg,dRdr,dRda],[dAdg,dAdr,dAda]])
    
    return A

def findss(par):
    #list of fixed par
    par=addfixedpar(par)    
    #function to find steady state
    #1. find where line reached 0
    Gi=np.arange(0,100,1)
    Gi=np.logspace(-20,5,1000,base=10)

    f=solvedfunction(Gi,par)
    x=f[1:-1]*f[0:-2] #when the output give <0, where is a change in sign, meaning 0 is crossed
    index=np.where(x<0)

    ss=[]
    for i in index[0]:
        G=brentq(solvedfunction, Gi[i], Gi[i+1],args=par) #find the value of AHL at 0
        #now we have AHL we can find AHL2 ss
        A= (par['beta_ahl']*np.power(G*par['K_ahl'],par['n_ahl']))/(1+np.power(G*par['K_ahl'],par['n_ahl']))
        A= A/par['delta_ahl']

        R = par['alpha_red']+((par['beta_red']-par['alpha_red'])*np.power(A*par['K_ahl_red'],par['n_ahl_red']))/(1+np.power(A*par['K_ahl_red'],par['n_ahl_red']))
        R = R / (1 + np.power(G*par['K_GREEN'],par['n_GREEN']))
        R = R/ par['delta_red']        
        ss.append(np.array([G,R,A]))

    return ss

def turinginstability(par,q=np.arange(0,200,1)):
    #step one find steady stateS
    par=addfixedpar(par)
    turing_type=[] 
    #q=np.logspace(-4,4,500,base=10)
#    ss =findsteadystate(par,nstep)
    ss= findss(par)
    sse=[]
    eigens=np.ones((len(ss),3,len(q)))*np.nan
    if len(ss)==0:
        turing_type.append(np.nan)
    for i,s in enumerate(ss): 
        A=jacobianMatrix(s[0],s[1],s[2],par)
        eigvals, eigvecs =eig(A)
        sse.append(eigvals.real)
        pos=sse[i][sse[i]>0]
        if np.all(sse[i]<0): #if all neg = stable point, test turing instability
            # add diffusion as in scholes et al.
            eigens_1=[]
            eigens_2=[]
            eigens_3=[]
            for qi in q:
                A=jacobianMatrix(s[0],s[1],s[2],par)
                A[2][2] = A[2][2] - (qi**2)*par['D_ahl']
                eigvals, eigvecs =eig(A)
                idx = eigvals.argsort()[::-1]   
                eigvals = eigvals[idx]
                eigens_1.append(eigvals.real[0])
                eigens_2.append(eigvals.real[1])
                eigens_3.append(eigvals.real[2])           
            eigens[i]= [eigens_1,eigens_2,eigens_3]
            for ei,e in enumerate(eigens[i]):
              if np.any(e>0):                
                  if np.all(e[-1]<0):
                      turing_type.append(1)
                      print(1)
                  else:
                      turing_type.append(2)
                      print(2)
                  print("Tu instability" )
              else:
                  turing_type.append(0)
                  
        if len(pos)>1:
            if pos[0]-pos[1] == 0:
                print("oscillation")
                eigens_1=[]
                eigens_2=[]
                eigens_3=[]
                for qi in q:
                    A=jacobianMatrix(s[0],s[1],s[2],par)
                    A[2][2] = A[2][2] - (qi**2)*par['D_ahl']
                    eigvals, eigvecs =eig(A)
                    idx = eigvals.argsort()[::-1]   
                    eigvals = eigvals[idx]
                    eigens_1.append(eigvals.real[0])
                    eigens_2.append(eigvals.real[1])
                    eigens_3.append(eigvals.real[2])
                eigens[i]= [eigens_1,eigens_2,eigens_3]
                turing_type.append(10)  #need to check hpf instability , need to add line here
            else:
                turing_type.append(0)  
        else:
            turing_type.append(0)
    return ss, np.nansum(turing_type), eigens
'''
def turinginstability(par):
    #step one find steady stateS
    turing_type=[]
    q=np.arange(0,100,0.1) 
#    ss =findsteadystate(par,nstep)
    ss= findss(par)
    sse=[]
    for i,s in enumerate(ss): 
        A=jacobianMatrix(s[0],s[1],s[2],par)
        eigvals, eigvecs =eig(A)
        sse.append(eigvals.real)
        pos=sse[i][sse[i]>0]
        if np.all(sse[i]<0): #if all neg = stable point, test turing instability
            # add diffusion as in scholes et al.
            eigens=[]
            for qi in q:
                A=jacobianMatrix(s[0],s[1],s[2],par)
                A[2][2] = A[2][2] - (qi**2)*par['D_ahl']
                eigvals, eigvecs =eig(A)
                #eigens=np.append(eigens,eigvals.real)
                eigens.append(eigvals.real)
            if np.any(eigens[i]>0):
                print("Tu instability")
                if eigens[i][-1]<0:
                    turing_type.append(1)
                else:
                    turing_type.append(2)
            else:
                turing_type.append(0)
        if len(pos)>1:
            if pos[0]-pos[1] == 0:
                print("oscillation")
        else:
            turing_type.append(0)

    return ss, sse, turing_type
'''

def choosepar(parlist):
    #choose random par in the defined range
    samplepar=[]
    for ipar,par in enumerate(parlist):
        samplepar.append(random.uniform(par['lower_limit'], par['upper_limit']))
   # p=pars_to_dict(samplepar,parlist)
    return np.array(samplepar)

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
  newpar=choosepar(parlist)    
  p=pars_to_dict(newpar,parlist)
  ss,tutype,e = turinginstability(p)
  #if tu >0:
    #selectpar.append(newpar)
  return newpar,tutype


def load(name,parlist):
    tutype= np.loadtxt(name+"_turingtype.out")
    p= np.loadtxt(name+"_par.out")

    namelist=[]
    for i,par in enumerate(parlist):
        namelist.append(parlist[i]['name'])
    df = pd.DataFrame(p, columns = namelist)
    df['tutype']=tutype
    df=df.sort_values(by='tutype', ascending=False)
    return df

#############################################################


def diffusionplot_1D(name,tu_df,parlist):
    nrow=tu_df.shape[0]
    density=np.ones((nx, ny))
    G = np.ones((nx, ny))*10
    R = np.ones((nx, ny))*10
    A = np.ones((nx, ny))*10
    #AHL[1,round(5/d)]=5
    #AHL2[1,round(5/d)]=5
    for i in np.arange(1,ny-1):
                R[1][i]=R[1][i]*random.randint(0,10)/10
                G[1][i]=G[1][i]*random.randint(0,10)/10
                A[1][i]=A[1][i]*random.randint(0,10)/10
    print(nrow)
    for n in np.arange(0,nrow):
        par=tu_df.iloc[n].tolist()[:-1] #transform in list and remove turing type
        p=pars_to_dict(par,parlist)
        r, g,a= Integration(G,R,A,density,p,totaltime=tt,dt=dt,d=d,oneD=True, dimensionless=False,isdiffusion=True)
       # plot1d(da,da2,0)
        plt.subplot(round(np.sqrt(nrow)),round(np.sqrt(nrow)),n+1)
        plt.plot(r[-2][1],'g')
        plt.plot(g[-2][1],'r')
        plt.plot(a[-2][1],'--b')
       # plt.ylim(0,1)
        print(n)
        
    plt.savefig(name+'_Tu_plot.pdf', bbox_inches='tight')
    #plt.show()

def instability_plot(name,tu_df,parlist):
    q=np.arange(0,5,0.1)
    nrow=tu_df.shape[0]
    sizex=round(np.sqrt(nrow)+0.5)
    sizey=round(np.sqrt(nrow))
    cl=['red','green','blue','orange']
    for n in np.arange(0,nrow):
        par=tu_df.iloc[n].tolist()[:-1] #transform in list and remove turing type
        p=pars_to_dict(par,parlist)
        ss,tutype,eigenpertub = turinginstability(p,q)
        plt.subplot(sizex,sizey,n+1)
        for ei,e in enumerate(eigenpertub): #for each ss
            for line in e: #for each eigen
              plt.plot(q,line,linewidth=0.5,c=cl[ei])
        plt.text((max(q)-0.1),0.01,str(tutype),color="pink")
        plt.axhline(y = 0.0, color = 'black', linestyle = '--',linewidth=0.1)
        plt.yscale('symlog', linthresh=0.001)
        plt.ylim(-1,1)
        plt.yticks(fontsize=2)
        plt.ylabel("Eigens")
        plt.xlabel("q")
        
    plt.tight_layout()
       # plt.ylim(-0.0001,0.0001)

        
    plt.savefig(name+'_Pertubation.pdf', bbox_inches='tight')

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
    plt.savefig(name+'_full_par_plot.png', bbox_inches='tight')
    plt.close()
####################################################


def run(name,Npars=5000):
    par,tutype=GeneratePars(parlist, ncpus=40,Npars=Npars)
    np.savetxt(name+'_turingtype.out', tutype)
    np.savetxt(name+'_par.out', par)


def niceplot(name):
    df=load(name,parlist)
    tu_df = df[df['tutype']>0]
    instability_plot(name,tu_df,parlist)
    #par_plot(name,tu_df,parlist)
    #diffusionplot_1D(name,tu_df,parlist)
    
    



####################################################################
#main function
####################################################################

name='TSRD_001'
#run(name,Npars=40000)
#niceplot(name)


############################################

tt=20
df=load(name,parlist)
tu_df = df[df['tutype']>0]
#print(tu_df)
n=2
density=np.ones((nx, ny))

    #AHL[1,round(5/d)]=5
    #AHL2[1,round(5/d)]=5


par=tu_df.iloc[n].tolist()[:-1] #transform in list and remove turing type
p=pars_to_dict(par,parlist)
ss,tutype,eigenpertup=turinginstability(p,q=np.arange(0,200,1))
ssn=2
A=jacobianMatrix(ss[ssn][0],ss[ssn][1],ss[ssn][2],p)
eigvals, eigvecs =eig(A)
print(eigvals)

#print(ss)
#print(eigenpertup.shape)
print(eigenpertup[:,:,0])

G = np.ones((nx, ny))*10e-100
R = np.ones((nx, ny))*10e-100
A = np.ones((nx, ny))*10e-100

G[1][:]=(ss[ssn][0]+10e-5)
R[1][:]=(ss[ssn][1]+10e-5)
A[1][:]=(ss[ssn][2]+10e-5)

#for i in np.arange(1,ny-1):
#                R[1][i]=R[1][i]*random.randint(0,10)/100
#                G[1][i]=G[1][i]*random.randint(0,10)/100
#                A[1][i]=A[1][i]*random.randint(0,10)/100

'''
r, g,a= Integration(G,R,A,density,p,totaltime=tt,dt=dt,d=d,oneD=True, dimensionless=False,isdiffusion=True)
# plot1d(da,da2,0)
#plt.subplot(round(np.sqrt(nrow)),round(np.sqrt(nrow)),n+1)
plt.subplot(2,2,1)
plt.plot(r[-2][1],'g')
plt.plot(g[-2][1],'r')
plt.plot(a[-2][1],'--b')
plt.yscale("log")
plt.subplot(2,2,2)
plt.plot(r[0][1],'g')
plt.plot(g[0][1],'r')
plt.plot(a[0][1],'--b')
plt.yscale("log")
plt.subplot(2,2,3)
print(a[:,1,5].shape)
plt.plot(r[:-2,1,50],'g')
plt.plot(g[:-2,1,50],'r')
plt.plot(a[:-2,1,50],'--b')
plt.yscale("log")
#plt.ylim(6*10e-5,8*10e-4)
plt.show()
'''