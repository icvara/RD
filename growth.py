#diffusion + growth of cell

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm, Normalize
import pandas as pd

def get_dt(dxy,maxD):
	d2 = dxy*dxy
	dt = d2 * d2 / (2 * maxD * (d2 + d2)) 
	return dt

#########################################################3
#size of the plate
x=1
y=1

#parameter

par={

	'max_density':1,
	'D': 0.1, #diffusion rate
	'G': 0.01, #growth diffusion rate

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

#parameter for modeling
tt = 1 #totaltime
dxy = 0.1 #interval size
dt = get_dt(dxy,maxD=1.1)  #interval time according to diffusion and interval dist
nx, ny = round(x/dxy), round(y/dxy)

###########################################################
###########################################################


def model_diffusion2D(u0,par,dxy):
	d2 = dxy*dxy
	u = u0.copy()
	uxx = (u0[2:, 1:-1] - 2*u0[1:-1, 1:-1] + u0[:-2, 1:-1])/d2
	uyy = (u0[1:-1, 2:] - 2*u0[1:-1, 1:-1] + u0[1:-1, :-2])/d2
	u[1:-1, 1:-1] =  par['D']*(uxx + uyy)
	return u

def model_growth1(u0,par,dxy):
	#work as diffusion with self activation
	d2 = dxy*dxy
	u = u0.copy()
	#uxx = (u0[2:, 1:-1]  + u0[1:-1, 1:-1] + u0[:-2, 1:-1])/d2
	#uyy = (u0[1:-1, 2:]  + u0[1:-1, 1:-1] + u0[1:-1, :-2])/d2
	uxx = (u0[2:, 1:-1]  + u0[:-2, 1:-1])/d2
	uyy = (u0[1:-1, 2:]  + u0[1:-1, :-2])/d2	
	u[1:-1, 1:-1] =  par['G']*(uxx + uyy)

	#u = par['max_density']*(u*par['G_K'])/(1+(u*par['G_K']))  - u0*par['G_delta']

	return u


def Integration(C0,G0,R0,U0,par,totaltime=10,dt=dt,dxy=dxy):   
	#U0 need to be of shape (totaltime,nx,ny)
	Ui=U0 
	Ci=C0
	Gi=G0
	Ri=R0

	U= np.zeros(((round(totaltime/dt+0.5)+1),nx,ny))
	C= np.zeros(((round(totaltime/dt+0.5)+1),nx,ny))
	G = np.zeros(((round(totaltime/dt+0.5)+1),nx,ny))
	R = np.zeros(((round(totaltime/dt+0.5)+1),nx,ny))


	U[0]=U0 
	G[0]=G0
	R[0]=R0
	C[0]=C0


	t=dt
	i=1

	while t < totaltime:
		u = model_diffusion2D(Ui,par,dxy)
		c = model_growth1(Ci,par,dxy)
		g,r = model(Gi,Ri,Ui,Ci,par)
		Ci = Ci + c*dt
		Ui = Ui + u*dt
		Gi = Gi + g*dt
		Ri = Ri + r*dt
		Ci[Ci>par['max_density']]=par['max_density']
		G[i]=Gi
		U[i]=Ui
		C[i]=Ci
		R[i]=Ri
		t=t+dt
		i=i+1
	return U, C,  G, R



def model(GREENi,REDi,AHLi,density,par):
    
	GREEN = par['alpha_green']+((par['beta_green']-par['alpha_green'])*np.power(AHLi*par['K_ahl_green'],par['n_ahl_green']))/(1+np.power(AHLi*par['K_ahl_green'],par['n_ahl_green']))
	GREEN = GREEN / (1 + np.power(REDi*par['K_RED'],par['n_RED']))
	GREEN = GREEN - par['delta_green']*GREENi
                                                                 
	RED = par['alpha_red']+((par['beta_red']-par['alpha_red'])*np.power(AHLi*par['K_ahl_red'],par['n_ahl_red']))/(1+np.power(AHLi*par['K_ahl_red'],par['n_ahl_red']))
	RED = RED / (1 + np.power(GREENi*par['K_GREEN'],par['n_GREEN']))
	RED = RED - par['delta_red']*REDi

   # AHL = (par['beta_ahl']*np.power(GREENi*par['K_ahl'],par['n_ahl']))/(1+np.power(GREENi*par['K_ahl'],par['n_ahl']))
   # AHLdif = diffusion(AHLi,d,par['D_ahl'],oneD)

   # AHL = AHL*density + AHLdif - par['delta_ahl']*(AHLi)

	GREEN=GREEN*density
	RED = RED*density
        

	return GREEN,RED











def plot_diffusion(U,step,totaltime):
	st=round(tt/dt/step)
	sizex=round(np.sqrt(step))
	sizey=round(np.sqrt(step)+0.5)
	for i in np.arange(0,step):
		plt.subplot(sizex,sizey,i+1)
		sns.heatmap(U[st*i])# norm=LogNorm())
	plt.show()

def plot_line(U,w):
	for i in np.arange(0,20):
		plt.plot(U[i,w,:])
	for i in np.arange(20,U.shape[0],step=50):
		plt.plot(U[i,w,:])
	plt.yscale("log")
	plt.show()

def run():
	U0= np.zeros((nx,ny))
	R0= np.zeros((nx,ny)) 
	G0= np.zeros((nx,ny)) 
 
	C0= np.zeros((nx,ny))
	C0[5,5]=1
	U0[5,5]=1

	U,C,G,R = Integration(C0,G0,R0,U0,par,totaltime=tt)
	plot_diffusion(U,step=20,totaltime=tt)
	plot_diffusion(G,step=20,totaltime=tt)

	plot_line(U,w=5)
	plot_line(G,w=5)






#########################################################3
################################################################
####################################################################


run()
	