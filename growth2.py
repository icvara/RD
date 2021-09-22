import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm, Normalize
import pandas as pd

#agent model one


par={
	
	'growth_speed':1,
	'K_growth':0
}


x=1 #cm
y=1 #cm
dxy=0.1 #cm
dt= 0.1 #h
tt=2 #h

nx=round(x/dxy)
ny=round(y/dxy)


def growth(c0,par,time):

	#1 check space availaibility
	if time%par['growth_speed'] == 0 :
		s1_x = c0[2:, 1:-1] + c0[:-2, 1:-1]
		s1_y = c0[1:-1,2:] + c0[1:-1,:-2]
		s1= s1_x+s1_y
		s1[s1>0]=1
		s2 = s1 - c0[1:-1,1:-1] 
		s2[s2<0]=0
		c=c0[1:-1,1:-1]  + s2
	else:
		c=c0
	return c


ef Integration(C0,G0,R0,U0,IPTG,par,totaltime=10,dt=dt,dxy=dxy):   
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
		
		#g,r,u,c = model(Gi,Ri,Ui,Ci,IPTG,par,dxy)
		#c,gd,rd= model_growth2(Ci,Gi,Ri,par,dxy)
		#u = model_diffusion2D(u1,par,dxy)
		g = u = r = 0
		c= growth(Ci,par,t)

		Ci = Ci + c*dt
		

		Ui = Ui + u*dt
		Gi = Gi + (g)*dt
		Ri = Ri + (r)*dt
		#Ci[Ci>par['max_density']]=par['max_density']
		G[i]=Gi
		U[i]=Ui
		C[i]=Ci
		R[i]=Ri
		t=t+dt
		i=i+1
	return U, C,  G, R

########################################3
#########################################


def run():
	U0= np.ones((nx,ny))*0
	R0= np.zeros((nx,ny)) 
	G0= np.zeros((nx,ny)) 
	C0= np.zeros((nx,ny))
	IPTG=0
	#C0[1:-1,1:-1]=1
	middle=round(x/dxy/2)

	C0[middle,middle]=1#0.01
	G0[middle,middle]=100#0.01
	U0[middle,middle]=0#0.01
	R0[middle,middle]=0#0.01



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

	U,C,G,R = Integration(C0,G0,R0,U0,IPTG,par,totaltime=tt)

	plot_diffusion(C,U,G,R,step=6,totaltime=tt,dt=dt)
	plt.plot(np.ones(round(nx)),'black')
	plt.plot(C[-2,:,middle],'purple')
	plt.plot(U[-2,:,middle],'b')
	plt.plot(R[-2,:,middle],'r')
	plt.plot(G[-2,:,middle],'g')


	plt.yscale("log")
	plt.show()

	plt.plot(np.ones(round(tt/dt)),'black')
	plt.plot(C[:-1,middle+2,middle],'purple')
	plt.plot(U[:-1,middle+2,middle],'b')
	plt.plot(R[:-1,middle+2,middle],'r')
	plt.plot(G[:-1,middle+2,middle],'g')


	plt.yscale("log")
	plt.show()
                              
	#plt.plot(R[-2,:,middle],'r')

	#plt.yscale("log")
	#plt.show()


	#plot_line(U,w=5)
	#plot_line(G,w=middle)
	plot_line(U,w=middle,step=6,tt=tt,dt=dt)

#


run()