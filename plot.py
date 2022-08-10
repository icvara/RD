#ploting parameter

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics
import os
from collections import Counter
import sys
from scipy.signal import argrelextrema
from matplotlib.colors import LogNorm, Normalize
import multiprocessing
import time
from functools import partial



modeltype="TSLT"
datatype="_gated_025IPTG"#"_percent"
filename="FIT009TSLT" +datatype #_percent"


data="data"+datatype+".txt"
datafile = 'data/'+modeltype + '/' +data




n=['11']#'34','30','25','20','15','10']#['20','15','10','5']#,'20','10']
#n=['100','150','175']
#n=['15']
#
sys.path.insert(0, '/users/ibarbier/RD/'+filename)
#sys.path.insert(0, 'C:/Users/Administrator/Desktop/Modeling/AC-DC/'+filename)
import model as meq

parlist=meq.parlist


######################################################################33
#########################################################################
###########################################################################

def load(number= n,filename=filename,parlist=parlist):
    namelist=[]
    for i,par in enumerate(parlist):
        namelist.append(parlist[i]['name'])
    
    path = filename+'/smc/pars_' + number + '.out'
    dist_path = filename+'/smc/distances_' + number + '.out'

    raw_output= np.loadtxt(path)
    dist_output= np.loadtxt(dist_path)
    df = pd.DataFrame(raw_output, columns = namelist)
    df['dist']=dist_output
    df=df.sort_values('dist',ascending=False)
    distlist= sorted(df['dist'])
    p=[]
    for dist in distlist:
        
        p_0=df[df['dist']==dist]
        p0=[]
        for n in namelist:
          p0.append(p_0[n].tolist()[0])
        p0=pars_to_dict(p0,parlist)
        p.append(p0)

    
    return p, df 




def pars_to_dict(pars,parlist):
### This function is not necessary, but it makes the code a bit easier to read,
### it transforms an array of pars e.g. p[0],p[1],p[2] into a
### named dictionary e.g. p['k0'],p['B'],p['n'],p['x0']
### so it is easier to follow the parameters in the code
    dict_pars = {}
    for ipar,par in enumerate(parlist):
        dict_pars[par['name']] = pars[ipar] 
    return dict_pars
##plotting part

def plot(ARA,p,name,nb,tt=120):
    #ARA=np.logspace(-4.5,-2.,1000,base=10)
    for i,par in enumerate(p):
        

        X,Y,Z = meq.model(ARA,par,totaltime=tt)
        df_X=pd.DataFrame(X,columns=ARA)
        df_Y=pd.DataFrame(Y,columns=ARA)
        df_Z=pd.DataFrame(Z,columns=ARA)


        plt.subplot(len(p),3,(1+i*3))
        sns.heatmap(df_X, cmap="Reds", norm=LogNorm())
        plt.xticks([])
        plt.ylabel('time')
        plt.subplot(len(p),3,(2+i*3))
        sns.heatmap(df_Y, cmap ='Blues', norm=LogNorm())
        plt.xticks([])
        plt.yticks([])
        plt.subplot(len(p),3,(3+i*3))
        sns.heatmap(df_Z, cmap ='Greens', norm=LogNorm())
        plt.xticks([])
        plt.yticks([])



    #plt.savefig(name+"/plot/"+nb+'_heatmap'+'.pdf', bbox_inches='tight')
    plt.savefig(name+"/plot/heatmap/"+nb+'_heatmap'+'.png', bbox_inches='tight')
    #plt.show()
    plt.close()


def plotALLX(ARA,p,name,nb):
    #ARA=np.logspace(-4.5,-2.,1000,base=10)
    sizex=np.sqrt(len(p))
    sizey=np.sqrt(len(p))
    for i,par in enumerate(p):        
        X,Y,Z = meq.model(ARA,par)
        df_X=pd.DataFrame(X,columns=ARA)
        #df_Y=pd.DataFrame(Y,columns=ARA)
        #df_Z=pd.DataFrame(Z,columns=ARA)
        plt.subplot(sizex,sizey,i+1)
        sns.heatmap(df_X, cmap="Reds", norm=LogNorm())

   # plt.savefig(name+"/plot/"+nb+'ALLALL_heatmap'+'.pdf', bbox_inches='tight')
    #plt.savefig(name+"/plot/"+nb+'_heatmap'+'.png', bbox_inches='tight')
    plt.show()
    plt.close()


def par_plot(df,name,nb,parlist,namelist):
    #plt.plot(df['K_ARAX'],df['K_ARAY'],'ro')
    fonts=5
 
    for i,par1 in enumerate(namelist):
        for j,par2 in enumerate(namelist):
            plt.subplot(len(namelist),len(namelist), i+j*len(namelist)+1)#, figsize=(len(namelist)*3, len(namelist)*3))
            if i == j :
                plt.hist(df[par1])
                plt.xlim((parlist[i]['lower_limit'],parlist[i]['upper_limit']))
            else:
                plt.scatter(df[par1],df[par2], c=df['dist'], s=0.1, cmap='viridis')# vmin=mindist, vmax=maxdist)
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
                if i==0 and j==len(namelist)-1:
                    plt.ylabel(par2,fontsize=fonts)
                    plt.xlabel(par1,fontsize=fonts)
                    plt.xticks(fontsize=fonts)
                    plt.yticks(fontsize=fonts,rotation=90)                 
    #plt.savefig(name+"/plot/"+nb+'_par_plot.pdf', bbox_inches='tight')
    plt.savefig(name+"/plot/"+nb+'_par_plot.png', bbox_inches='tight', dpi=300)

    plt.close()
    #plt.show()
    
def splitted_parplot(n,filename,parlist):
    namelist=[]
    for i,par in enumerate(parlist):
       namelist.append(parlist[i]['name'])
    namelist=np.array(namelist) 
    parlist=np.array(parlist)   
    p, pdf= load(n,filename,parlist)

    namelist2=namelist[[2,4,9,12]] #only K par
    parlist2=parlist[[2,4,9,12]]
    namelist3=namelist[[0,1,6,7,8,11]] #only B and activation
    parlist3=parlist[[0,1,6,7,8,11]]
    namelist4=namelist[[7,8,9,10,11]] #only Y par
    parlist4=parlist[[7,8,9,10,11]]
    
    namelist5=namelist[[0,1,6,7,8,11]] #only ARA par
    parlist5=parlist[[0,1,6,7,8,11]]
    
    
    par_plot(pdf,filename,(str(n)+'ALL'),parlist,namelist)
    par_plot(pdf,filename,(str(n)+'K'),parlist2,namelist2)
    par_plot(pdf,filename,(str(n)+'B'),parlist3,namelist3)
    par_plot(pdf,filename,(str(n)+'Y'),parlist4,namelist4)
    par_plot(pdf,filename,(str(n)+'ARA'),parlist5,namelist5)
        
def plot_alltime(n,filename,parlist):
    namelist=[]
    for i,par in enumerate(parlist):
        namelist.append(parlist[i]['name'])
    parl = np.append(namelist,'dist')
    index=1
    size=round(np.sqrt(len(parl)))
    for i,name in enumerate(parl):
        plt.subplot(size,size,index)
        plt.tight_layout()
        for ni,nmbr in enumerate(n):
            p,df= load(nmbr,filename,parlist)
            sns.kdeplot(df[name],bw_adjust=.8,label=nmbr)
        #plt.ylim(0,1)
        if i < (len(parl)-2):
            plt.xlim((parlist[i]['lower_limit'],parlist[i]['upper_limit']))
        if index==size:       
          plt.legend(bbox_to_anchor=(1.05, 1))
        index=index+1
    plt.savefig(filename+"/plot/"+'ALLround_plot.pdf', bbox_inches='tight')
    plt.close()


def par_plot2(df,df2,name,nb,parlist,namelist):

    fonts=6
    
    for i,par1 in enumerate(namelist):
        for j,par2 in enumerate(namelist):
            plt.subplot(len(namelist),len(namelist), i+j*len(namelist)+1)
            if i == j :
                sns.kdeplot(df[par1],color='black',bw_adjust=.8,linewidth=0.5)
                #sns.kdeplot(c[par1],color='gray',bw_adjust=.8,linewidth=0.5)
                #sns.kdeplot(a[par1],color='green', bw_adjust=.8,linewidth=0.5)
                sns.kdeplot(df2[par1],color='red',bw_adjust=.8,linewidth=0.5)
                #sns.kdeplot(d[par1],color='orange',bw_adjust=.8,linewidth=0.5)
                plt.ylabel("")
                plt.xlabel("")
                plt.xlim((parlist[i]['lower_limit'],parlist[i]['upper_limit']))
            else:
                plt.scatter(df[par1],df[par2], c='black', s=0.01)# vmin=mindist, vmax=maxdist)
               # plt.scatter(c[par1],c[par2], color='black', s=0.0001)
               # plt.scatter(a[par1],a[par2], color='green', s=0.0001)
                plt.scatter(df2[par1],df2[par2], color='red', s=0.01)                
                #plt.scatter(d[par1],d[par2], color='orange', s=0.0001)
                #plt.scatter(df2[par1],df2[par2], c='blue', s=0.001)
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
    plt.savefig(name+"/"+nb+'_compar_plot.png', bbox_inches='tight',dpi=300)
    plt.close()
    #plt.show()


def plotselectedparoverall(n,filename,parlist):
     selected_index = np.loadtxt(filename+'/ACDC_par_index.out')
     criteria = np.loadtxt(filename +'/criteria.out')
     selected_index =[int(x) for x in selected_index]      
     ARA=np.logspace(-4.5,-2.,20,base=10)
     p, pdf= load(n,filename,parlist)
     pdf2=pdf.iloc[selected_index]
     p_selected =  np.take(p,selected_index) 
     pdf2['up']=criteria[:,0]
     pdf2['down']=criteria[:,1]
     pdf2['idk']=criteria[:,2]
         
     namelist=[]
     for i,par in enumerate(meq.parlist):
       namelist.append(parlist[i]['name'])
     namelist=np.array(namelist)

     namelist2=namelist[[2,4,9,12]] #only K par
     parlist2=parlist[[2,4,9,12]] 
     namelist3=namelist[[0,1,6,7,8,11]] #only B and activation
     parlist3=parlist[[0,1,6,7,8,11]]
     namelist4=namelist[[7,8,9,10,11]] #only Y par
     parlist4=parlist[[7,8,9,10,11]]
     
     par_plot2(pdf,pdf2,filename,n,parlist,namelist)
     par_plot2(pdf,pdf2,filename,'K',parlist2,namelist2)
     par_plot2(pdf,pdf2,filename,'B',parlist3,namelist3)
     par_plot2(pdf,pdf2,filename,'Y',parlist4,namelist4)
     

def compare_plot(p,filename,nb,datafile,modeltype):
       # gmin,gmax,rmin,rmax=meq.Get_data4(datafile,p[0])
        gmax,gmin,rmax,rmin=meq.Get_data(datafile)
        A=gmin.index.values
        I=gmin.columns.values
        maxi= np.nanmax([ np.nanmax(rmax.to_numpy()),np.nanmax(gmax.to_numpy())])
        mini= np.nanmin([ np.nanmin(rmin.to_numpy()),np.nanmin(gmin.to_numpy())])

        fig, axs = plt.subplots(6, 2)
#        ss=meq.findss(A,I,p[0])
#     Mmindist=np.nanmax(ss[:,:,:,:],axis=2)
#     mmindist=np.nanmin(ss[:,:,:,:],axis=2)
        for pi in p:
            #gmin,gmax,rmin,rmax=meq.Get_data4(datafile,pi)
            ss=meq.findss(A,I,pi,modeltype)
            M=np.nanmax(ss[:,:,:,:],axis=2)
            m=np.nanmin(ss[:,:,:,:],axis=2)
          
            
            for ii,i in enumerate(I):
                axs[ii,0].plot(M[:,ii,0],'g-',linewidth=0.4)
                axs[ii,1].plot(M[:,ii,1],'r-',linewidth=0.4)    
                axs[ii,0].plot(m[:,ii,0],'b--',linewidth=0.4)
                axs[ii,1].plot(m[:,ii,1],'b--',linewidth=0.4)
                

                
                axs[ii,0].set_ylim(ymin=mini-0.15*mini,ymax=maxi+.15*maxi)
                axs[ii,1].set_ylim(ymin=mini-0.15*mini,ymax=maxi+.15*maxi)
               # axs[ii,0].set_ylim(ymin=mini-0.15*mini,ymax=maxi+.15*maxi)
               # axs[ii,1].set_ylim(ymin=mini-0.15*mini,ymax=maxi+.15*maxi)
               
        for ii,i in enumerate(I):

                axs[ii,0].plot(gmax.to_numpy()[:,ii],'go', markersize=4.)
                axs[ii,0].plot(gmin.to_numpy()[:,ii],'go', markersize=4., mfc='none')

                axs[ii,1].plot(rmax.to_numpy()[:,ii],'ro', markersize=4.)
                axs[ii,1].plot(rmin.to_numpy()[:,ii],'ro', markersize=4., mfc='none')

                '''
                axs[ii,0].plot(Mmindist[:,ii,0],'r',linewidth=0.2)
                axs[ii,0].plot(Mmindist[:,ii,1],'r',linewidth=0.2)    
                axs[ii,1].plot(mmindist[:,ii,0],'r',linewidth=0.2)
                axs[ii,1].plot(mmindist[:,ii,1],'r',linewidth=0.2)  
                '''    
        #plt.show()  
        plt.savefig(filename+"/plot/"+nb+'_compare_plot.png', bbox_inches='tight',dpi=300)


def plot_distance(p,filename,nb,datafile):
      gmin,gmax,rmin,rmax=meq.Get_data4(datafile)
      A=gmin.index.values
      I=gmin.columns.values
      fig, axs = plt.subplots(6, 6)
      dall=np.empty(shape=(len(p),6,len(A),len(I)))
      print(dall.shape)
      for index,pi in enumerate(p):
          d=meq.distance4(pi,datafile,split=True)
          dall[index]=d
          
          for ii,i in enumerate(I):
                axs[ii,0].plot(d[0,:,ii],'g-',linewidth=0.4,alpha=0.1)
                axs[ii,1].plot(d[1,:,ii],'g--',linewidth=0.4,alpha=0.1)
                axs[ii,2].plot(d[2,:,ii],'b-',linewidth=0.4,alpha=0.1)
                axs[ii,3].plot(d[3,:,ii],'r-',linewidth=0.4,alpha=0.1) 
                axs[ii,4].plot(d[4,:,ii],'r--',linewidth=0.4,alpha=0.1)
                axs[ii,5].plot(d[5,:,ii],'b-',linewidth=0.4,alpha=0.1)   

          
      plt.savefig(filename+"/plot/"+nb+'_test.png', bbox_inches='tight',dpi=300)
        
      fig, axs = plt.subplots(6, 6)
      for ii,i in enumerate(I):
                axs[ii,0].boxplot(dall[:,0,:,ii], showfliers=False)
                axs[ii,1].boxplot(dall[:,1,:,ii], showfliers=False)
                axs[ii,2].boxplot(dall[:,2,:,ii], showfliers=False)
                axs[ii,3].boxplot(dall[:,3,:,ii], showfliers=False) 
                axs[ii,4].boxplot(dall[:,4,:,ii], showfliers=False)
                axs[ii,5].boxplot(dall[:,5,:,ii], showfliers=False)
         
               # axs[ii,0].set_yscale('log', base=10)
              #  axs[ii,1].set_yscale('log', base=10)
               # axs[ii,2].set_yscale('log', base=10)
                #axs[ii,3].set_yscale('log', base=10)
           
                
      plt.savefig(filename+"/plot/"+nb+'_test2.png', bbox_inches='tight',dpi=300)
      
def plot_distribution(p,filename,nb,datafile,modeltype):
      gmin,gmax,rmin,rmax=meq.Get_data(datafile)
      A=gmin.index.values
      I=gmin.columns.values
      fig, axs = plt.subplots(6, 2, figsize=(30,20))
      #A=[0]
      #I=[0]

      
      pTOT= np.empty(shape=5) * np.nan
      #shape=(len(A),len(I),1000,2,2))   
      i=0

             # pTOT= np.empty(shape=(len(A),len(I),1000,2,2))   :
      for index,pi in enumerate(p):
              ss=meq.findss(A,I,pi,modeltype)
              M=np.nanmax(ss[:,:,:,:],axis=2)
              m=np.nanmin(ss[:,:,:,:],axis=2)
              for ai,a in enumerate(A):
                  for ii,i in enumerate(I):
                    pTOT= np.vstack([pTOT,np.array([a,i,"min",m[ai,ii,0],'G'])])
                    pTOT= np.vstack([pTOT,np.array([a,i,"min",m[ai,ii,1],'R'])])
                    pTOT= np.vstack([pTOT,np.array([a,i,"max",M[ai,ii,0],'G'])])
                    pTOT= np.vstack([pTOT,np.array([a,i,"max",M[ai,ii,1],'R'])])
                #pTOT[ai,ii,index,0]=M
                #pTOT[ai,ii,index,1]=m

      print(pTOT.shape)
      df = pd.DataFrame(pTOT[1:], columns=['A','I','minmax','p','C'])
      df.loc[:,'p']=df['p'].astype(float)
      df.loc[:,'A']=df['A'].astype(float)
      df.loc[:,'I']=df['I'].astype(float)

      for ii,i in enumerate(I):
      
        for ci,c in enumerate(['G','R']):
          subdf = df[(df["C"] == c) & (df["I"] == i)]
          sns.violinplot(ax=axs[ii, ci],x='A', y='p', hue="minmax", data=subdf, palette="Set2", split=True, inner="quartile")  
          axs[ii,ci].legend([],[], frameon=False) 
          
        axs[ii,0].plot(gmax.to_numpy()[:,ii],'-go', markersize=4.)
        axs[ii,0].plot(gmin.to_numpy()[:,ii],'-go', markersize=4., mfc='none')
        axs[ii,1].plot(rmax.to_numpy()[:,ii],'-ro', markersize=4.)
        axs[ii,1].plot(rmin.to_numpy()[:,ii],'-ro', markersize=4., mfc='none')
          
           
      plt.savefig(filename+"/plot/"+nb+'_distribution.png', bbox_inches='tight',dpi=300)
     # sns.catplot(x='A', y='p', hue="minmax",col='C', row='I', data=df, kind="violin", palette="Set2", split=True) 
     # plt.savefig(filename+"/plot/"+nb+'_testtest3.png', bbox_inches='tight',dpi=300)


   
   
def compare_plot_mode(p,filename,nb,datafile):

    p_mode=pdf.mode(axis=0).to_dict(orient='index')[0]
    name="mode_"+nb
    compare_plot2([p_mode],filename,name,datafile)

def bifu_heatmap(p_mode):
    p0=p_mode

    A=np.logspace(-4,1,40)
    I=np.logspace(1,-4,40)

    ss=meq.findss(A,I,p0)

    
    hyst_matrix = np.count_nonzero(~np.isnan(ss[:,:,:,0]),axis=2)
    col_matrix = np.nanmax(ss[:,:,:,:],axis=2)

   # hyst_matrix[hyst_matrix==1] = np.NaN
    hyst_matrix = hyst_matrix.astype("float")
    col_matrix = col_matrix.astype("float")

   # col_matrix[col_matrix < .05] = np.NaN
    #hyst_matrix[hyst_matrix==1] = np.NaN
   # hyst_matrix[hyst_matrix==1] = np.NaN
    plt.subplot(1,3,1)
    sns.heatmap(col_matrix[:,:,1], cmap='Reds')
    plt.subplot(1,3,2)
    sns.heatmap(col_matrix[:,:,0], cmap='Greens')
    plt.subplot(1,3,3)
    sns.heatmap(hyst_matrix, cmap='Blues')
    #plt.show()
    plt.savefig(filename+"/plot/"+'bifurcation.png', bbox_inches='tight',dpi=300)


##############################################################################################################3   

if __name__ == "__main__":
   
    if os.path.isdir(filename+'/plot') is False: ## if 'smc' folder does not exist:
        os.mkdir(filename+'/plot') ## create it, the output will go there


    namelist=[]
    for i,par in enumerate(parlist):
        namelist.append(parlist[i]['name'])
    
    #n=["25"]

    A=[10]#np.logspace(-4,1,200)
    I=[10]#np.logspace(-1,0,15)


    for i in n:

        p, pdf= load(i,filename,meq.parlist)
        
        
        par_plot(pdf,filename,i,meq.parlist,namelist)
        compare_plot([p[24],p[499],p[974]],filename,i+"sub",datafile,modeltype)
        compare_plot(p,filename,i,datafile,modeltype)      
        plot_distribution(p,filename,i,datafile,modeltype)
        


    '''
    i=n[0]
    p, pdf= load(i,filename,meq.parlist)
    p=p[0]
    
    p['alpha_red']= -3.#2.496783409731153
    p[ 'beta_red']= 2.4#3.114471765921569

    p[ 'K_ahl_red']= 1.5#-1.1621886779174828
    p[ 'n_ahl_red']= 2# 0.27463516060617854

    p[ 'alpha_green']= -3.#2.72500158113845
    p[  'beta_green']= 2.7# -0.8161096974564463

    p[ 'K_ahl_green'] = 2.#-1.555685369218159
    p[ 'n_ahl_green'] =  2#3.4831414888659915

    p['F_red']=0
    p['F_green']=0

    p[ 'basal_red']= 2.5#3.114471765921569
    p[ 'basal_green']= 2.5#3.114471765921569

    p[ 'K_RED'] = 1#-4.047529380939851
    p[ 'K_GREEN']= 1#-4.545681717524607


    p[ 'n_RED'] = 2# 2.8230220470434815
    p[ 'n_GREEN' ]= 2# 2.2761084701257603

    p[ 'K_IPTG'] = 2.8#4.207597662328212

    #d=meq.distance(p,datafile,modeltype)
    #print(d)

    A= np.array([0.05,0.025,0.0125,0.00625,0.003125,0]) # np.logspace(-4,1,2)
    A=np.flipud(A)
    A=np.logspace(-4,-1,20)
    I=[0,0.0625,0.125,0.25,0.5,1]#np.logspace(-1,0,2)
    J=np.linspace(-1,1,6)

    fig, axs = plt.subplots(6, 6, figsize=(7,7))
    for jj,j in enumerate(J):
        ss=meq.findss(A,I,p,modeltype)
        for ii,i in enumerate(I):
                    axs[ii,jj].plot(ss[:,ii,:,0],'o')
    #plt.plot(ss[:,0,:,0],'-r')
    plt.show()


    

    compare_plot([p],filename,"test",datafile,modeltype)
    
    '''






