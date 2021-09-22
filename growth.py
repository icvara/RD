#diffusion + growth of cell

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm, Normalize
import pandas as pd
import matplotlib.animation as animation



def get_dt(dxy,maxD):
	d2 = dxy*dxy
	dt = d2 * d2 / (2 * maxD * (d2 + d2)) 
	return dt

#########################################################3




par={
    
    
    'beta_red':100,#,100,

    'K_IPTG':2.22,

    'K_GREEN':15,
    'K_RED':20,

    'n_GREEN':2,
    'n_RED':2,
    'delta_red':1,#8.3,
    'delta_green':1,#1,#8.3,

    #need to be defined
    'beta_green':100,# 100,
    'leak_green':0,

    'K_ahl_red':133,
    'K_ahl_green':1133,
    'n_ahl_green':1.61,
    'n_ahl_red':1.61,

     #luxI par
    'beta_ahl':0.1,
    'K_ahl':1,
    'n_ahl':8.,
    'delta_ahl':1,
    'leak_ahl':10e-7,


    'max_density':15, #stationary
	'D': 10,# 4.9e-6 *60*60, #diffusion rate
	'G': 1e-10,# 45e-4,  #colonies horizontal expantion
	 #growth diffusion rate.
	'K_growth':2.0, #speed to reach stationary
	'growth_speed':2#2#2 # doubling time in h 



}

#growth speed 45 um/h warren et al  2019 eLife
# AHL diffusion 6.7*10^5 µm2/min payne et al 2013 or 4.9 x 10-6 cm2/s Stewart P.S., 2003

#size of the plate
x=2 #cm
y=2 #cm
tt = 72#totaltime hour
dxy = 0.02 #interval size in cm
#dt= 1 # 1h
dt = get_dt(dxy,maxD=1.1)  #interval time according to diffusion and interval dist
dt=0.01
nx, ny = round(x/dxy), round(y/dxy)
middle=round(x/dxy/2)
###########################################################
###########################################################


def model_diffusion2D(u0,par,dxy):
	d2 = dxy*dxy
	u = u0.copy()
	uxx = (u0[2:, 1:-1] - 2*u0[1:-1, 1:-1] + u0[:-2, 1:-1])/1#d2
	uyy = (u0[1:-1, 2:] - 2*u0[1:-1, 1:-1] + u0[1:-1, :-2])/1#d2
	u[1:-1, 1:-1] =  par['D']*(uxx + uyy)
	return u

'''
def model_growth1(u0,g0,r0,par,dxy):
	#work as diffusion with self activation
	d2 = dxy*dxy

	u = u0.copy()
	uxx = (u0[2:, 1:-1] - 2*u0[1:-1, 1:-1] + u0[:-2, 1:-1])/4#d2
	uyy = (u0[1:-1, 2:] - 2*u0[1:-1, 1:-1] + u0[1:-1, :-2])/4#d2	
	u[1:-1, 1:-1] =  par['G']*(uxx + uyy)
	growth= par['K_growth']*u0 * (1 - u0 / par['max_density'])
	growth = growth  + u

	u = g0.copy()
	uxx = (g0[2:, 1:-1] - 2*g0[1:-1, 1:-1] + g0[:-2, 1:-1])/4#d2
	uyy = (g0[1:-1, 2:] - 2*g0[1:-1, 1:-1] + g0[1:-1, :-2])/4#d2	
	u[1:-1, 1:-1] =  par['G']*(uxx + uyy)
	green=u

	u = r0.copy()
	uxx = (r0[2:, 1:-1] - 2*r0[1:-1, 1:-1] + r0[:-2, 1:-1])/4#d2
	uyy = (r0[1:-1, 2:] - 2*r0[1:-1, 1:-1] + r0[1:-1, :-2])/4#d2	
	u[1:-1, 1:-1] =  par['G']*(uxx + uyy)
	red=u

	return growth,green,red


def model_growth2(c0,g0,r0,par,dxy):
	#this model implies that "cell" are the size of the dxy , the growth speed can only be int
	#only when a distance unit reach 1 , it can go to the next ones at a certain speed
	d2 = dxy*dxy
	c=c0.copy()
	c2=c0.copy()
	c2[c2<1]=0
	c2[c2>1]=1
	cxx = (c2[2:, 1:-1]  + c2[:-2, 1:-1])/4#d2
	cyy = (c2[1:-1, 2:] + c2[1:-1, :-2])/4#d2
	c[1:-1, 1:-1] =  par['G']*(cxx + cyy)
	c= c +  par['K_growth']*c0 * (1 - c0 / par['max_density'])

	c1=c0.copy()
	c1[c1>0]=1 #old cell 
	c3=(cxx + cyy)  #potential new cell
	c3[c3>0]=1  
	c4= c3-c1[1:-1, 1:-1]# position of new cell
	c4[c4<0]=0

	g2=g0*c2
	gxx = (g2[2:, 1:-1] + g2[:-2, 1:-1])/4#/d2
	gyy = (g2[1:-1, 2:] + g2[1:-1, :-2])/4#/d2
	g=g0.copy()
	g[1:-1, 1:-1] =  (gxx + gyy) 
	g[1:-1, 1:-1]=g[1:-1, 1:-1]*c4

	r2=r0*c2
	rxx = (r2[2:, 1:-1]+ r2[:-2, 1:-1])/4#/d2
	ryy = (r2[1:-1, 2:] + r2[1:-1, :-2])/4#/d2
	r=r0.copy()
	r[1:-1, 1:-1] =  (rxx + ryy)
	r[1:-1, 1:-1]=r[1:-1, 1:-1]*c4

	return c,g,r

'''
def growth(c0,g0,r0,u0,d0,par,time_count):

	if time_count  >= par['growth_speed'] :
		c=c0.copy()
		s1_x = c0[2:, 1:-1] + c0[:-2, 1:-1]
		s1_y = c0[1:-1,2:] + c0[1:-1,:-2]
		s1= s1_x+s1_y
		s2=s1.copy()
		s2[s2>0]=1
		s3 = s2 - c0[1:-1,1:-1] 
		s3[s3<0]=0
		c[1:-1,1:-1] = s3
		time_count=0


		g=g0.copy()
		g[1:-1,1:-1] = dilution(g0,c0,s2,s1)
		r=r0.copy()
		r[1:-1,1:-1] = dilution(r0,c0,s2,s1)
		u=u0.copy()
		u[1:-1,1:-1] = dilution(u0,c0,s2,s1)
		d=d0.copy()
		d[1:-1,1:-1] = d0[1:-1,1:-1] + s1

	else:
		c=c0*0
		g=0
		r=0
		u=0
		d=d0

	return c,g,r,u,d, time_count

def dilution(x0,c0,s2,s1):
	c=c0.copy()
	new_cell = s2.copy()-c[1:-1, 1:-1]# position of new cell
	new_cell[new_cell<0]=0 #all negative values are old cell with neighbourg
	xxx = (x0[2:, 1:-1] -2*x0[1:-1, 1:-1] + x0[:-2, 1:-1])/4#/d2
	xyy = (x0[1:-1, 2:] -2*x0[1:-1, 1:-1] + x0[1:-1, :-2])/4#/d2
	x = xxx+ xyy 
	dil = 1/(new_cell*s1)
	dil[dil>1]=0
	x= x *dil
	return x
	#g=g0.copy()
	#g[1:-1, 1:-1] =  (gxx + gyy) 
	#g[1:-1, 1:-1]=g[1:-1, 1:-1]*c4

def Integration(C0,G0,R0,U0,d0,IPTG,par,totaltime=10,dt=dt,dxy=dxy):   
	#U0 need to be of shape (totaltime,nx,ny)
	Ui=U0 
	di=d0
	Ci=C0
	Gi=G0
	Ri=R0

	U= np.zeros(((round(totaltime/dt+0.5)+1),nx,ny))
	d= np.zeros(((round(totaltime/dt+0.5)+1),nx,ny))
	C= np.zeros(((round(totaltime/dt+0.5)+1),nx,ny))
	G = np.zeros(((round(totaltime/dt+0.5)+1),nx,ny))
	R = np.zeros(((round(totaltime/dt+0.5)+1),nx,ny))

	U[0]=U0 
	d[0]=d0 
	G[0]=G0
	R[0]=R0
	C[0]=C0

	t=dt
	time_count=dt
	i=1

	while t < totaltime:
		
		g,r,u = model(Gi,Ri,Ui,Ci,di,IPTG,par)
		c,gc,rc,uc,di,time_count= growth(Ci,Gi,Ri,Ui,di,par,time_count)
		#u = model_diffusion2D(u1,par,dxy)
		Ci = Ci + c #*dt
		

		Ui = Ui + u*dt #+uc
		#print(Ui[middle-2:middle,:])
		#print(".................")
	#	Uoi = Uoi + uo*dt
		Gi = Gi + g*dt +gc
		Ri = Ri + r*dt +rc
		#Ci[Ci>par['max_density']]=par['max_density']
		G[i]=Gi
		U[i]=Ui
		d[i]=di
		C[i]=Ci
		R[i]=Ri
		t=t+dt
		time_count = time_count+dt
		i=i+1
	return U,d, C,  G, R




def model(GREENi,REDi,AHLi,CELLi,density,IPTG,par):
    


    st_thr=0.90  
    stationary=density.copy()
    #stationary[stationary>st_thr*par['max_density']]=0
    stationary[stationary>0]=1
    #stationary = (1 - CELLi / par['max_density'])
 
    #growth,green_growth, red_growth = model_growth1(CELLi,GREENi, REDi, par,dxy)

   # CELL,green_growth, red_growth,time_count =growth(CELLi,GREENi, REDi,par,time_count)

    GREEN = (par['beta_green']*np.power(AHLi*par['K_ahl_green'],par['n_ahl_green']))/(1+np.power(AHLi*par['K_ahl_green'],par['n_ahl_green']))
    GREEN = GREEN / (1 + np.power(REDi*par['K_RED'],par['n_RED']))
    GREEN = GREEN - par['delta_green']*GREENi  + par['leak_green']
    GREEN = GREEN*stationary# + green_growth



    free_GREENi= GREENi / ( 1+ par['K_IPTG']*IPTG)
    RED = (par['beta_red']*np.power(AHLi*par['K_ahl_red'],par['n_ahl_red']))/(1+np.power(AHLi*par['K_ahl_red'],par['n_ahl_red']))
    RED = RED / (1 + np.power(free_GREENi*par['K_GREEN'],par['n_GREEN']))
    RED = RED - par['delta_red']*REDi # + par['alpha_red']
    RED=RED*stationary # + red_growth


    AHL = (par['beta_ahl']*np.power(GREENi*par['K_ahl'],par['n_ahl']))/(1+np.power(GREENi*par['K_ahl'],par['n_ahl']))
    AHLdif = model_diffusion2D(AHLi,par,dxy) 

    AHL = AHL +par['leak_ahl'] 
    AHL = AHL*stationary - par['delta_ahl']*(AHLi) 
    AHL= AHL + AHLdif 

   # AHL= AHL +  



    return GREEN,RED,AHL


#####################################################


def plot_diffusion(C,U,Uo,G,R,step,totaltime,dt):
	st=round(totaltime/dt/step)
	sizey=step
	for i in np.arange(0,step):
		plt.subplot(step,5,i*5+1)
		sns.heatmap(C[st*i],cmap="Greys")# norm=LogNorm())
		plt.subplot(step,5,i*5+2)
		sns.heatmap(Uo[st*i],cmap="viridis",vmin=0,vmax=10)# norm=LogNorm())
		plt.subplot(step,5,i*5+3)
		sns.heatmap(U[st*i],cmap="Blues")# nor,cmap="Yellows"m=LogNorm())
		plt.subplot(step,5,i*5+4)
		sns.heatmap(G[st*i],cmap="Greens")# norm=LogNorm())
		plt.subplot(step,5,i*5+5)
		sns.heatmap(R[st*i],cmap="Reds")# norm=LogNorm())
	plt.show()

def plot_line(U,w,step,tt,dt):
	X=U.copy()
	#X[X==0]=np.nan
	st=round(tt/dt/step)
	for i in np.arange(0,step):
		plt.plot(X[st*i,w,:])
	plt.yscale("log")
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
	U0= np.ones((nx,ny))*0
	d0= np.ones((nx,ny))*0

	R0= np.zeros((nx,ny)) 
	G0= np.zeros((nx,ny)) 
	C0= np.zeros((nx,ny))
	IPTG=10
	#C0[1:-1,1:-1]=1
	middle=round(x/dxy/2)

	C0[middle,middle]=1#0.01
	G0[middle,middle]=100#0.01
#	C0[middle-1:middle+2,middle]=1#0.01
#	G0[middle-1:middle+2,middle]=100#0.01
	d0=C0

#	C0[middle+1,middle+1]=1#0.01
#	R0[middle+1,middle+1]=100#0.01


	#C0[middle-2:middle+1,middle-1:middle+2]=1
	#G0[middle-2:middle+1,middle-1:middle+2]=100



	#d0[middle,middle]=1#0.01


	#U0[middle,middle]=100#0.01
	#R0[middle,middle]=0#0.01


	#print(c)

#	C0[middle,middle+1]=0.1

	#C0[1,1]=1
	#U0[middle,middle]=1
	#G0[1,1]=100

	'''
	name='TSRD_001'
	df=load(name,parlist)
	tu_df = df[df['tutype']>0]

	par= tu_df.iloc[2]
	par['delta_red']=1
	par['delta_green']=1
	par['delta_ahl']=1
	par['max_density']=1#1,
	'''
	#c,g,r=model_growth2(C0,G0,R0,par,dxy)
	#print(r[:,middle])

	U,d,C,G,R = Integration(C0,G0,R0,U0,d0,IPTG,par,totaltime=tt)

	plot_diffusion(C,U,d,G,R,step=6,totaltime=tt,dt=dt)
#	plt.plot(np.ones(round(nx)),'black')
	plt.plot(C[-2,:,middle],'m')
	plt.plot(U[-2,:,middle],'b--')

	plt.plot(R[-2,:,middle],'r')
	plt.plot(G[-2,:,middle],'g')
	plt.plot(d[-2,:,middle],'m--')



	plt.yscale("log")
	plt.show()

#	plt.plot(np.ones(round(tt/dt)),'black')
	plt.plot(C[:-1,middle+1,middle],'m')
	plt.plot(U[:-1,middle+1,middle],'b--')
	plt.plot(d[:-1,middle+1,middle],'m--')
	plt.plot(R[:-1,middle+1,middle],'r')
	plt.plot(G[:-1,middle+1,middle],'g')
	plt.yscale("log")
	plt.show()

	plt.plot(C[:-1,middle+8,middle],'m')
	plt.plot(U[:-1,middle+8,middle],'b--')
	plt.plot(d[:-1,middle+8,middle],'m--')
	plt.plot(R[:-1,middle+8,middle],'r')
	plt.plot(G[:-1,middle+8,middle],'g')
	plt.yscale("log")
	plt.show()

#	animateFig(d,'viridis','AHL_grow_cell_stationary.gif',min=0,max=10)
#	animateFig(U,'Blues','AHL_grow_stationary.gif')
	animateFig(R,'Reds','R_oscillation.gif')#,min=0,max=100)
	animateFig(G,'Greens','test.gif')#,min=0,max=100)
	animateFig(U,'Blues','test.gif')#,min=0,max=100)
	plt.show()






	






def animateFig(X,color,filename,min=None,max=None):
	fig=plt.figure()

	plt.ylabel('Space [mm]')
	plt.xlabel('Space [mm]')

	ims =[]
	for i in range(int(tt/dt)):
		if min==None:
	   		im=plt.imshow(X[i][:,:],animated=True,cmap=color)
		else:
	   		im=plt.imshow(X[i][:,:],animated=True,cmap=color, vmin=min, vmax=max)
		ims.append([im])
	#print(ims)
	ani = animation.ArtistAnimation(fig, ims, interval=1, blit=True)
	#ani.save(filename, writer=animation.PillowWriter(fps=100))
	#plt.show()
	#plt.close() 


#########################################################3
################################################################
####################################################################





run()







	