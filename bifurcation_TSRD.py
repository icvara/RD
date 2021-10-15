
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.linalg import eig
import random
#from scipy.stats import norm, uniform, multivariate_normal


from scipy import optimize
from scipy.optimize import brentq
import pandas as pd


par = {
	'K_ahl_green': 69.89802317210678,
	'n_ahl_green': 1.9008144425324653, 
	'alpha_green': 47.840674921103684, 
	'beta_green': 2.31557081283309, 
	'K_GREEN': 79.54987455560202, 
	'n_GREEN': 1.997288173047589, 
	'K_ahl_red': 65.77484986491167, 
	'n_ahl_red': 0.7940009415081422, 
	'alpha_red': 29.794492292099378, 
	'beta_red': 1.3597920120675178, 
	'K_RED': 63.235535679312704, 
	'n_RED': 1.8389599089663373, 
	'K_ahl': 48.88614491266986, 
	'n_ahl': 1.0718691537985365, 
	'beta_ahl': 89.08249490069538,




	'max_density':1,
	'D': 1.00, #diffusion rate
	'G': 0.001 #growth diffusion rate.
}

par['delta_red']=1
par['delta_green']=1
par['delta_ahl']=1


def ssmodel(GREENi,REDi,AHLi,par):
	#here to calculate steady state:  we do without diffusion and cell density

	GREEN = par['alpha_green']+((par['beta_green']-par['alpha_green'])*np.power(AHLi*par['K_ahl_green'],par['n_ahl_green']))/(1+np.power(AHLi*par['K_ahl_green'],par['n_ahl_green']))
	GREEN = GREEN / (1 + np.power(REDi*par['K_RED'],par['n_RED']))
	GREEN = GREEN - par['delta_green']*GREENi

	RED = par['alpha_red']+((par['beta_red']-par['alpha_red'])*np.power(AHLi*par['K_ahl_red'],par['n_ahl_red']))/(1+np.power(AHLi*par['K_ahl_red'],par['n_ahl_red']))
	RED = RED / (1 + np.power(GREENi*par['K_GREEN'],par['n_GREEN']))
	RED = RED - par['delta_red']*REDi
	AHL = (par['beta_ahl']*np.power(GREENi*par['K_ahl'],par['n_ahl']))/(1+np.power(GREENi*par['K_ahl'],par['n_ahl']))	
	AHL = AHL  - par['delta_ahl']*(AHLi)
	return GREEN,RED, AHL

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

    A = np.array([[dGdg,dGdr,dGda],[dRdg,dRdr,dRda],[dAdg,dAdr,dAda]])
    
    return A

def jacobianMatrix2(ss,par):
    JM=np.ones((ss.shape[0],3,3))*np.nan 
    G=ss[:,0]
    R=ss[:,1]
    A=ss[:,2]

   # A=A[:,None]

    #need to double check , some mistake are here
    dGdg = - par['delta_green'] *G/G
    dGdr = -((par['alpha_green'] - par['beta_green'])*np.power((par['K_ahl_green']*A),par['n_ahl_green'])*par['n_RED']*np.power((par['K_RED']*R),par['n_RED']))
    dGdr =  dGdr/((np.power((par['K_ahl_green']*A),par['n_ahl_green'])+1)*R*np.power((np.power((par['K_RED']*R),par['n_RED'])+1),2))   
    dGda =((par['alpha_green'] - par['beta_green']) *par['n_ahl_green']*np.power((par['K_ahl_green']*A),par['n_ahl_green'])) 
    dGda =  dGda/((np.power((par['K_RED']*R),par['n_RED'])+1)*A*np.power((np.power((par['K_ahl_green']*A),par['n_ahl_green'])+1),2))
    
    dRdg = - ((par['alpha_red'] - par['beta_red'])*np.power((par['K_ahl_red']*A),par['n_ahl_red'])*par['n_GREEN']*np.power((par['K_GREEN']*G),par['n_GREEN']))
    dRdg =  dRdg/((np.power((par['K_ahl_red']*A),par['n_ahl_red'])+1)*G*np.power((np.power((par['K_GREEN']*G),par['n_GREEN'])+1),2))    
    dRdr = - par['delta_red']  *R/R  
    dRda = ((par['alpha_red'] - par['beta_red']) *par['n_ahl_red']*np.power((par['K_ahl_red']*A),par['n_ahl_red']))
    dRda=dRda/((np.power((par['K_GREEN']*G),par['n_GREEN'])+1)*A*np.power((np.power((par['K_ahl_red']*A),par['n_ahl_red'])+1),2))
    
    dAdg = (par['beta_ahl']*par['n_ahl']*(np.power((par['K_ahl']*G),par['n_ahl'])))
    dAdg = dAdg/ (G*np.power((np.power((G*par['K_ahl']),par['n_ahl'])+1),2))
    dAdr = 0 *A/A
    dAda = -par['delta_ahl'] *A/A

    JM[:,0,0]=dGdg
    JM[:,0,1]=dGdr
    JM[:,0,2]=dGda

    JM[:,1,0]=dRdg
    JM[:,1,1]=dRdr
    JM[:,1,2]=dRda

    JM[:,2,0]=dAdg
    JM[:,2,1]=dAdr
    JM[:,2,2]=dAda
    
    return JM


   
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


def findss(par):
    #list of fixed par
    #function to find steady state
    #1. find where line reached 0
    Gi=np.arange(0,100,1)
    Gi=np.logspace(-20,5,1000,base=10)

    f=solvedfunction(Gi,par)
    x=f[1:-1]*f[0:-2] #when the output give <0, where is a change in sign, meaning 0 is crossed
    index=np.where(x<0)

    ss=[]
    nNode=3 # number of nodes : X,Y,Z
    nStstate= 5
    ss=np.ones((nStstate,nNode))*np.nan  
    for it,i in enumerate(index[0]):
        G=brentq(solvedfunction, Gi[i], Gi[i+1],args=par) #find the value of AHL at 0
        #now we have AHL we can find AHL2 ss
        A= (par['beta_ahl']*np.power(G*par['K_ahl'],par['n_ahl']))/(1+np.power(G*par['K_ahl'],par['n_ahl']))
        A= A/par['delta_ahl']

        R = par['alpha_red']+((par['beta_red']-par['alpha_red'])*np.power(A*par['K_ahl_red'],par['n_ahl_red']))/(1+np.power(A*par['K_ahl_red'],par['n_ahl_red']))
        R = R / (1 + np.power(G*par['K_GREEN'],par['n_GREEN']))
        R = R/ par['delta_red']        
       # ss.append(np.array([G,R,A]))

        ss[it]=np.array([G,R,A])

    return ss




    ##############################################3Bifurcation part 
################################################################################

def getminmax(X,Y,Z,transient):
    M=np.ones(3)*np.nan
    m=np.ones(3)*np.nan
    M[0]=max(X[transient:])
    M[1]=max(Y[transient:])
    M[2]=max(Z[transient:])
    m[0]=min(X[transient:])
    m[1]=min(Y[transient:])
    m[2]=min(Z[transient:])

    return M,m

def getpeaks(X,transient):
    max_list=argrelextrema(X[transient:], np.greater)
    maxValues=X[transient:][max_list]
    min_list=argrelextrema(X[transient:], np.less)
    minValues=X[transient:][min_list]

    return maxValues, minValues

def reachss(ssa,X,par,a):
    thr= 0.001
    out=False

    for ss in ssa:
        if np.all(np.isnan(ss)) == False:
            A=meq.jacobianMatrix(a,ss[0],ss[1],ss[2],par)
            eigvals, eigvecs =np.linalg.eig(A)
            sse=eigvals.real
            if np.all(sse<0):
                if abs((X[-2]-ss[0])/X[-2]) < thr:
                         out= True

    return out


def limitcycle(ai,ss,ARA,init,par,dummy,X=[],Y=[],Z=[],transient=500,count=0):
    threshold=0.01
    tt=200
    c=count
    #init=[init[0] + 10e-5,init[1] + 10e-5,init[2] + 10e-5]
    ssa=ss[ai]
    x,y,z=meq.model([ARA[ai]],par,totaltime=tt,init=init)
    X=np.append(X,x)
    Y=np.append(Y,y)
    Z=np.append(Z,z)

    M = m = np.nan

    maxValues, minValues = getpeaks(X,transient)

    if len(minValues)>4 and len(maxValues)>4:
        maximaStability = abs((maxValues[-2]-minValues[-2])-(maxValues[-3]-minValues[-3]))/(maxValues[-3]-minValues[-3]) #didn't take -1 and -2 because, i feel like -1 is buggy sometimes...
        if maximaStability > threshold:
            if reachss(ssa,X,par,ARA[ai])==False:
                #if we didn't reach the stability repeat the same for another 100 time until we reach it
                initt=[X[-2],Y[-2],Z[-2]] #take the -2 instead of -1 because sometimes the -1 is 0 because of some badly scripted part somewhere
                c=c+1
                if c<10:
               # if reachsteadystate(a,initt,par) == False:
                    M,m = limitcycle(ai,ss,ARA,initt,par,dummy,X,Y,Z,count=c)            
                if c==10:
                        #here the issue comes probably from 1) strange peak 2)very long oscillation
                        #here I try to get rid of strange peak , with multiple maxima and minima by peak. for this I take the local maximun and min of each..
                        #the issue here is to create artefact bc in the condition doesn't specify this kind of behaviour

                        maxValues2 = getpeaks(maxValues,0)[0]  
                        minValues2 = getpeaks(minValues,0)[1]
                        if len(minValues2)>4 and len(maxValues2)>4:
                          maximaStability2 = abs((maxValues2[-2]-minValues2[-2])-(maxValues2[-3]-minValues2[-3]))/(maxValues2[-3]-minValues2[-3]) #didn't take -1 and -2 because, i feel like -1 is buggy sometimes...
                          if maximaStability2 < threshold:
                              M,m = getminmax(X,Y,Z,transient=transient)
                          else:
                              #very long oscillation?
                              print("no limit cycle: probably encounter stable point at {} arabinose at p{}".format(ARA[ai],dummy))
                              '''
                              plt.plot(X[transient:])
                              plt.yscale("log")
                              plt.show() 
                              '''
                        else:
                              print("too long oscillation?? at {} arabinose at p{}".format(ARA[ai],dummy))
        else:

            M,m = getminmax(X,Y,Z,transient=transient)
         
    else:
       # print("no enough oscillation: " + str(len(minValues)))
        if reachss(ssa,X,par,ARA[ai])==False:
            #print("homoclinic")
            initt=[X[-2],Y[-2],Z[-2]]
            c=c+1
            if c<10:
                M,m = limitcycle(ai,ss,ARA,initt,par,dummy,X,Y,Z,count=c)  
            if c==10: 
                #very long oscillation?          
                print("error in limit cycle ara = {}, p{}".format(ARA[ai],dummy))
                '''
                plt.plot(X[transient:])
                plt.yscale("log")
                plt.show()
                '''


    return M,m

def getEigen(ARA,par,s):
    A=meq.jacobianMatrix(ARA,s[0],s[1],s[2],par)
    eigvals, eigvecs =np.linalg.eig(A)
    sse=eigvals.real
    return sse #, np.trace(A), np.linalg.det(A)

def getpar(i,df):
    return pars_to_dict(df.iloc[i].tolist())

def calculateALL2(ARA,parUsed, dummy):
    #sort ss according to their stabilitz
    #create stability list of shape : arabinose x steady x X,Y,Z 
    nNode=3 # number of nodes : X,Y,Z
    nStstate= 5 # number of steady state accepted by. to create the storage array
  #  ss=np.ones((len(ARA),nStstate,nNode))*np.nan 
    eig= np.ones((nStstate,nNode))*np.nan 
    unstable=np.ones((nStstate,nNode))*np.nan
    stable=np.ones((nStstate,nNode))*np.nan
    oscillation=np.ones((nStstate,nNode))*np.nan
    homoclincic=np.ones((nStstate,nNode))*np.nan
    M=np.ones((nStstate,nNode))*np.nan
    m=np.ones((nStstate,nNode))*np.nan


    delta=10e-10 #perturbation from ss
    ss=findss(parUsed) 
    A=jacobianMatrix2(ss,parUsed)
	
    for i in np.arange(0,nStstate):
            if np.any(np.isnan(A[i]))==False:
                eigvals, eigvecs =np.linalg.eig(A[i])
                eig[i]=eigvals.real

                if any(eig[i]>0):
                    pos=eig[i][eig[i]>0]
                    if len(pos)==2:
                            if pos[0]-pos[1] == 0:                                
                                init=[ss[i,0]+delta,ss[i,1]+delta,ss[i,2]+delta]
                              #  M[i],m[i] = limitcycle(i,ss,ARA,init,parUsed,dummy)###
                                if np.isnan(M[i][0]):
                                    homoclincic[i]=ss[i] 

                                else:
                                    oscillation[i]=ss[i]
                            else:
                                unstable[i]=ss[i]
                    else:
                        unstable[i]=ss[i]
                else:
                    if np.all(eig[i]<0):
                        stable[i]=ss[i]
                    else:
                       unstable[i]=ss[i]
    return ss,eig,unstable,stable,oscillation,homoclincic,M,m


ARA=np.array([1])
ss,eig,unstable,stable,oscillation,homoclincic,M,m = calculateALL2(ARA,par,0)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ss=findss(par)
xs=stable[:,2]
ys=stable[:,1]
zs= stable[:,0]
ax.scatter(xs, ys, zs,c='b')
xs=unstable[:,2]
ys=unstable[:,1]
zs= unstable[:,0]
ax.scatter(xs, ys, zs, c='r')
xs=oscillation[:,2]
ys=oscillation[:,1]
zs= oscillation[:,0]
ax.scatter(xs, ys, zs, c='g')
xs=homoclincic[:,2]
ys=homoclincic[:,1]
zs= homoclincic[:,0]
ax.scatter(xs, ys, zs, c='g')


ax.set_xlabel('H')
ax.set_ylabel('R')
ax.set_zlabel('G')

plt.show()

xs=unstable[:,2]
ys=unstable[:,1]
zs= unstable[:,0]
plt.scatter(xs,zs,c='r')
xs=stable[:,2]
ys=stable[:,1]
zs= stable[:,0]
plt.scatter(xs,zs,c='g')
plt.show()