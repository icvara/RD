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

    'K_ahl_red':10,#0.1,
    'K_ahl_green':233,
    'n_ahl_green':1.61,
    'n_ahl_red':1.61,

    #luxI par
    'beta_ahl':0.1,
    'K_ahl':10,
    'n_ahl':4.,
    'delta_ahl':1,

    'leak_ahl':0,

	'K_growth':0.5, #speed to reach stationary
    'max_density':100, #stationary

	'D': 10,# 4.9e-6 *60*60, #diffusion rate
	#'delta_ahle':0.15,#0.15,#1,#0.07,

	'G': 0.05#0.005, #cm/h ??    #1e-10,# 45e-4,  #colonies horizontal expantion
	 #growth diffusion rate.
	#'growth_speed':2#2#2 # doubling time in h 


}





#growth speed 45 um/h warren et al  2019 eLife
# AHL diffusion 6.7*10^5 µm2/min payne et al 2013 or 4.9 x 10-6 cm2/s Stewart P.S., 2003

#size of the plate
x=4 #cm
y=4 #cm
tt = 35#totaltime hour
dxy = 0.1 #interval size in cm
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

def growth3(c0,c1,d0,par,t,dxy):

	r= par['G']*t /dxy
	lenx = np.arange(nx)
	leny = np.arange(ny)
	cell=c1.copy()
	center = np.where(c0>0)
	for x in lenx:
		for y in leny:
			if (x-center[0])**2 + (y-center[1])**2 <= r**2 :
					cell[x,y]=1

	density =  par['K_growth']*d0 * (1 - d0 / par['max_density'])
	density = density + (cell-c1) #add new cell at 1 density for the moment


	return cell,density

	#test if point inside circle.. from pythagore.. could has more computationaly optimize method


def heredity(x0,c0,s2,s1):
	x=0
	#todo
	return x




def Integration(C0,G0,R0,U0,Ue0,d0,IPTG,par,totaltime=10,dt=dt,dxy=dxy):   
	#U0 need to be of shape (totaltime,nx,ny)
	Ui=U0 
	Uei=Ue0
	Di=d0
	Ci=C0
	Gi=G0
	Ri=R0

	U= np.zeros(((round(totaltime/dt+0.5)+1),nx,ny))
	Ue= np.zeros(((round(totaltime/dt+0.5)+1),nx,ny))

	D= np.zeros(((round(totaltime/dt+0.5)+1),nx,ny))
	C= np.zeros(((round(totaltime/dt+0.5)+1),nx,ny))
	G = np.zeros(((round(totaltime/dt+0.5)+1),nx,ny))
	R = np.zeros(((round(totaltime/dt+0.5)+1),nx,ny))

	U[0]=U0 
	Ue[0]=Ue0
	D[0]=d0 
	G[0]=G0
	R[0]=R0
	C[0]=C0

	t=dt
	time_count=dt
	i=1

	while t < totaltime:
		
		ue=0
		g,r,u= model(Gi,Ri,Ui,Ci,Di,IPTG,par)
		#c,gc,rc,uc,di,time_count= growth(Ci,Gi,Ri,Ui,di,par,time_count)
		c,d = growth3(C0,Ci,Di,par,t=t,dxy=dxy)

		#u = model_diffusion2D(u1,par,dxy)
		Ci = c 
		Di= Di + d*dt
		Uei = Uei + ue*dt
		Ui = Ui + u*dt #+uc
		#print(Ui[middle-2:middle,:])
		#print(".................")
	#	Uoi = Uoi + uo*dt
		Gi = Gi + g*dt #+gc
		Ri = Ri + r*dt #+rc
		#Ci[Ci>par['max_density']]=par['max_density']
		G[i]=Gi
		U[i]=Ui
		Ue[i]=Uei
		D[i]=Di
		C[i]=Ci
		R[i]=Ri
		t=t+dt
		time_count = time_count+dt
		i=i+1
	return U,Ue,D, C,  G, R




def model(GREENi,REDi,AHLi,CELLi,density,IPTG,par):
    

    stationary = (1 - density/par['max_density']) * CELLi
    stationary = CELLi

    GREEN = (par['beta_green']*np.power(AHLi*par['K_ahl_green'],par['n_ahl_green']))/(1+np.power(AHLi*par['K_ahl_green'],par['n_ahl_green']))
    GREEN = GREEN / (1 + np.power(REDi*par['K_RED'],par['n_RED']))
    GREEN = GREEN - par['delta_green']*GREENi 
    GREEN = GREEN*stationary# + green_growth

    free_GREENi= GREENi / ( 1+ par['K_IPTG']*IPTG)
    RED = (par['beta_red']*np.power(AHLi*par['K_ahl_red'],par['n_ahl_red']))/(1+np.power(AHLi*par['K_ahl_red'],par['n_ahl_red']))
    RED = RED / (1 + np.power(free_GREENi*par['K_GREEN'],par['n_GREEN']))
    RED = RED - par['delta_red']*REDi # + par['alpha_red']
    RED=RED*stationary # + red_growth

    AHL = (par['beta_ahl']*np.power(GREENi*par['K_ahl'],par['n_ahl']))/(1+np.power(GREENi*par['K_ahl'],par['n_ahl']))
    AHL = AHL +par['leak_ahl']
    AHLdif = model_diffusion2D(AHLi,par,dxy) 
    AHL= AHL*stationary + AHLdif - par['delta_ahl']*(AHLi)

   # AHL= AHL +  

    return GREEN,RED,AHL

def model2(GREENi,REDi,AHLi,AHLei,CELLi,density,IPTG,par):
    
    AHLii = AHLei

    stationary = (1 - density/par['max_density']) * CELLi
    #stationary = CELLi

    GREEN = (par['beta_green']*np.power(AHLii*par['K_ahl_green'],par['n_ahl_green']))/(1+np.power(AHLii*par['K_ahl_green'],par['n_ahl_green']))
    GREEN = GREEN / (1 + np.power(REDi*par['K_RED'],par['n_RED']))
    GREEN = GREEN - par['delta_green']*GREENi 
    GREEN = GREEN*stationary# + green_growth

    free_GREENi= GREENi / ( 1+ par['K_IPTG']*IPTG)
    RED = (par['beta_red']*np.power(AHLii*par['K_ahl_red'],par['n_ahl_red']))/(1+np.power(AHLii*par['K_ahl_red'],par['n_ahl_red']))
    RED = RED / (1 + np.power(free_GREENi*par['K_GREEN'],par['n_GREEN']))
    RED = RED - par['delta_red']*REDi # + par['alpha_red']
    RED=RED*stationary # + red_growth

    AHL = (par['beta_ahl']*np.power(GREENi*par['K_ahl'],par['n_ahl']))/(1+np.power(GREENi*par['K_ahl'],par['n_ahl']))
    AHL = AHL +par['leak_ahl']
    AHL = AHL - par['delta_ahl']*(AHLi) #+ AHLei
    AHL= AHL*stationary 

    AHLdif = model_diffusion2D(AHLei,par,dxy) 
    AHLe = AHLi + AHLdif -par['delta_ahle']*AHLei

   # AHL= AHL +  

    return GREEN,RED,AHL,AHLe

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
	Ue0= np.ones((nx,ny))*0

	d0= np.ones((nx,ny))*0

	R0= np.zeros((nx,ny)) 
	G0= np.zeros((nx,ny)) 
	C0= np.zeros((nx,ny))
	IPTG=10
	#C0[1:-1,1:-1]=1
	middle=round(x/dxy/2)

	C0[middle,middle]=1#0.01
	G0[middle,middle]=100#0.01
	U0[middle,middle]=0#0.01

#	C0[middle-1:middle+2,middle]=1#0.01
#	G0[middle-1:middle+2,middle]=100#0.01
	d0=C0.copy()
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

	U,Ue,d,C,G,R = Integration(C0,G0,R0,U0,Ue0,d0,IPTG,par,totaltime=tt)
	for t in np.arange(1,U.shape[0]):
		sns.heatmap(C[t,:,:],cmap='Greys')
		plt.savefig('cell/'+str(t)+'.png', bbox_inches='tight',dpi=300)
		plt.close()
		sns.heatmap(d[t,:,:],cmap='viridis',norm=LogNorm())
		plt.savefig('density/'+str(t)+'.png', bbox_inches='tight',dpi=300)
		plt.close()

	#animateFig(C,'Greys','test.gif')#,min=0,max=100)
	#animateFig(d,'viridis','test.gif',min=0,max=15)
	#animateFig(R,'Reds','test.gif',min=0,max=50)
	#animateFig(G,'Greens','test.gif')#,min=0,max=100)
	#animateFig(G/R,'RdYlGn','stripe_stationary.gif',min=0,max=100)


	#animateFig(U,'Blues','test.gif')#,min=0,max=100)

	plt.show()
	
#	plot_diffusion(C,U,d,G,R,step=6,totaltime=tt,dt=dt)
#	plt.plot(np.ones(round(nx)),'black')
	plt.plot(C[-2,:,middle],'m')
	plt.plot(U[-2,:,middle],'b-')
	plt.plot(Ue[-2,:,middle],'b--')


	plt.plot(R[-2,:,middle],'r')
	plt.plot(G[-2,:,middle],'g')
	plt.plot(d[-2,:,middle],'m--')

	plt.yscale("log")
	plt.show()
	'''
#	plt.plot(np.ones(round(tt/dt)),'black')
	plt.plot(C[:-1,middle+0,middle],'m')
	plt.plot(U[:-1,middle+0,middle],'b--')
	plt.plot(d[:-1,middle+0,middle],'m--')
	plt.plot(R[:-1,middle+0,middle],'r')
	plt.plot(G[:-1,middle+0,middle],'g')
	plt.yscale("log")
	plt.show()

	plt.plot(C[:-1,middle+4,middle],'m')
	plt.plot(U[:-1,middle+4,middle],'b--')
	plt.plot(d[:-1,middle+4,middle],'m--')
	plt.plot(R[:-1,middle+4,middle],'r')
	plt.plot(G[:-1,middle+4,middle],'g')
	plt.yscale("log")
	plt.show()
	'''
	plt.subplot(2,2,1)
	sns.heatmap(G[:-1,middle:,middle],cmap='Greens',norm=LogNorm())
	plt.subplot(2,2,2)
	sns.heatmap(R[:-1,middle:,middle],cmap='Reds',norm=LogNorm())
	#sns.heatmap(Ue[:-1,middle:,middle],cmap='Reds')#,norm=LogNorm())


	plt.subplot(2,2,3)

	sns.heatmap(d[:-1,middle:,middle],cmap='viridis',norm=LogNorm())
	plt.subplot(2,2,4)
	#U2[U2<10e-10]=0
	sns.heatmap(U[:-1,middle:,middle],cmap='Blues')#,norm=LogNorm())

	plt.show()


#	animateFig(d,'viridis','AHL_grow_cell_stationary.gif',min=0,max=10)
#	animateFig(U,'Blues','AHL_grow_stationary.gif')
#	animateFig(R,'Reds','R_oscillation.gif')#,min=0,max=100)
#	animateFig(G,'Greens','test.gif')#,min=0,max=100)
#	animateFig(U,'Blues','test.gif')#,min=0,max=100)







	






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
	#ani.save(filename, writer=animation.PillowWriter(fps=1000))
	#plt.show()
	#plt.close() 


#########################################################3
################################################################
####################################################################





run()







	