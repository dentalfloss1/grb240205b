import matplotlib.pyplot as plt
import pandas as pd 
import datetime
import numpy as np

trigger = datetime.datetime(2024, 2, 5, 22, 15, 8, 00)

plotdata = pd.read_csv("grbmeas.csv")
startdate = [(datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S.%f") - trigger).total_seconds()/3600/24 for d in plotdata['start']]
stopdate = [(datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S.%f") - trigger).total_seconds()/3600/24 for d in plotdata['stop']]
obsdur = [(t2-t1) for t1,t2 in zip(startdate,stopdate)]
plotdata['obsdate'] = [t1+(dur/2) for t1,dur in zip(startdate,obsdur)]
plotdata = plotdata[np.isin(plotdata['freq'],[5.5,9.0])]
fig,axs = plt.subplots(2,1,figsize=(10,15),sharex=True,sharey=True)

for freq,ax in zip(np.sort(np.unique(plotdata['freq'])),axs):
    curdata = plotdata[plotdata['freq']==freq]
    xdata = curdata['obsdate']
    print(xdata)
    ydata = curdata['flux']*1e-6
    yerr = np.sqrt(curdata['err']**2 + curdata['rms']**2)*1e-6
    ax.scatter(xdata,ydata,label=f'{freq} GHz')
    ax.errorbar(xdata,ydata,yerr=yerr,fmt=' ')
    ax.set_title(f'{freq} GHz')
    ax.set_xscale('log')
    ax.set_yscale('log')
    if freq==17.0:
        ax.set_xlabel("Days post trigger")
    ax.set_ylabel("Flux Density (Jy)")
    ax.set_ylim(1e-5,3e-3)
# plt.legend()
plt.tight_layout()
plt.savefig("lightcurve.png")
plt.close()
    
