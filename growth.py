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
x=2
y=2

#parameter
'''
par={

	'max_density':1,
	'D': 0.001, #diffusion rate
	'G': 0.1, #growth diffusion rate

	'K_ahl_green':100.0,
    'n_ahl_green':2.0,
    'alpha_green':0.10,
    'beta_green':1,
    'K_GREEN':0,
    'n_GREEN':2.0,
    'delta_green':1,

    'K_ahl_red':1.0,
    'n_ahl_red':2.0,
    'alpha_red':0,
    'beta_red':1,
    'K_RED':10,
    'n_RED':2.0,
    'delta_red':1,

    'K_ahl':100,#100,
    'n_ahl':2.0,
    'beta_ahl':1.,
    'delta_ahl':2,#1.,
    
    'D_ahl':1.

}
'''
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
	'G': 0.1 #growth diffusion rate.
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

#parameter for modeling
tt = 10 #totaltime
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
	uxx = (u0[2:, 1:-1] - 2*u0[1:-1, 1:-1] + u0[:-2, 1:-1])/d2
	uyy = (u0[1:-1, 2:] - 2*u0[1:-1, 1:-1] + u0[1:-1, :-2])/d2	
	u[1:-1, 1:-1] =  par['G']*(uxx + uyy)

	growth= par['K_growth']*u0 * (1 - u0 / par['max_density'])
	#print(growth)
	#print(u)

	growth = growth  + u

	return growth


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
		c = model_growth1(Ci,par,dxy)
		g,r,u = model(Gi,Ri,Ui,Ci,par,dxy)
		#u = model_diffusion2D(u1,par,dxy)

		Ci = Ci + c*dt
		Ui = Ui + u*dt
		Gi = Gi + g*dt
		Ri = Ri + r*dt
		#Ci[Ci>par['max_density']]=par['max_density']
		G[i]=Gi
		U[i]=Ui
		C[i]=Ci
		R[i]=Ri
		t=t+dt
		i=i+1
	return U, C,  G, R



def model(GREENi,REDi,AHLi,density,par,dxy=dxy):

	d2=density.copy()
	#d2[d2==1]=0
    
	GREEN = par['alpha_green']+((par['beta_green']-par['alpha_green'])*np.power(AHLi*par['K_ahl_green'],par['n_ahl_green']))/(1+np.power(AHLi*par['K_ahl_green'],par['n_ahl_green']))
	GREEN = GREEN / (1 + np.power(REDi*par['K_RED'],par['n_RED']))
	GREEN = GREEN *d2 - par['delta_green']*GREENi
                                                                 
	RED = par['alpha_red']+((par['beta_red']-par['alpha_red'])*np.power(AHLi*par['K_ahl_red'],par['n_ahl_red']))/(1+np.power(AHLi*par['K_ahl_red'],par['n_ahl_red']))
	RED = RED / (1 + np.power(GREENi*par['K_GREEN'],par['n_GREEN']))
	RED = RED *d2 - par['delta_red']*REDi

	
	AHL = (par['beta_ahl']*np.power(GREENi*par['K_ahl'],par['n_ahl']))/(1+np.power(GREENi*par['K_ahl'],par['n_ahl']))
	AHLdif = model_diffusion2D(AHLi,par,dxy) 

	AHL = AHL*d2 + AHLdif - par['delta_ahl']*(AHLi)   







	return GREEN,RED,AHL





def plot_diffusion(C,U,G,R,step,totaltime):
	st=round(tt/dt/step)
	sizey=step
	for i in np.arange(0,step):
		plt.subplot(step,4,i*4+1)
		sns.heatmap(C[st*i],cmap="Purples")# norm=LogNorm())
		plt.subplot(step,4,i*4+2)
		sns.heatmap(U[st*i],cmap="Blues")# nor,cmap="Yellows"m=LogNorm())
		plt.subplot(step,4,i*4+3)
		sns.heatmap(G[st*i],cmap="Greens")# norm=LogNorm())
		plt.subplot(step,4,i*4+4)
		sns.heatmap(R[st*i],cmap="Reds")# norm=LogNorm())
	plt.show()

def plot_line(U,w):
	X=U.copy()
	X[X==0]=np.nan
	for i in np.arange(0,20):
		plt.plot(X[i,w,:])
	for i in np.arange(20,U.shape[0],step=50):
		plt.plot(X[i,w,:])
	#plt.yscale("log")
	plt.show()

	plt.plot(X[:,w,w])
	plt.show()

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

def run():
	U0= np.ones((nx,ny))
	R0= np.zeros((nx,ny)) 
	G0= np.zeros((nx,ny)) 
 
	C0= np.zeros((nx,ny))
	#C0[1:-1,1:-1]=1

	middle=round(x/dxy/2)

	C0[middle,middle]=0.001
#	C0[middle,middle+1]=0.1

	#C0[1,1]=1
	#U0[middle,middle]=1
	#G0[1,1]=100
	G0[middle,middle]=1


	
	name='TSRD_001'
	df=load(name,parlist)
	tu_df = df[df['tutype']>0]

	par= tu_df.iloc[2]
	par['delta_red']=1
	par['delta_green']=1
	par['delta_ahl']=1
	par['max_density']=1#1,
	par['D']= 1.00, #diffusion rate
	par['G']= 0.001#0.01 #growth diffusion rate.
	par['K_growth']=1.5


	U,C,G,R = Integration(C0,G0,R0,U0,par,totaltime=tt)
	plot_diffusion(C,U,G,R,step=6,totaltime=tt)

	#plot_line(U,w=5)
	plot_line(C,w=middle)
#





#########################################################3
################################################################
####################################################################



run()
	