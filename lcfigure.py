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
plotdata['startdate'] = startdate
plotdata['stopdate'] = stopdate
# plotdata = plotdata[np.isin(plotdata['freq'],[5.5,9.0])]
fig,axs = plt.subplots(5,1,figsize=(7,15),sharex=True,sharey=True)

for freq,ax in zip(np.sort(np.unique(plotdata['freq'])),axs):
    curdata = plotdata[plotdata['freq']==freq]
    xdata = curdata['obsdate']
    ydata = curdata['flux']*1e-6
    yerr = np.sqrt(curdata['err']**2 + curdata['rms']**2)*1e-6
    ax.scatter(xdata,ydata,label=f'{freq} GHz')
    ax.errorbar(xdata,ydata,yerr=yerr,fmt=' ')
    ax.set_title(f'{freq} GHz')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel("Flux Density (Jy)")
    ax.set_ylim(1e-5,3e-3)
    for o in np.sort(curdata['obs']):
        subdata = curdata[curdata['obs']==o]
        print(subdata)
        startobs = subdata['startdate'].min()
        endobs = subdata['stopdate'].max()
        ax.axvspan(startobs,endobs, alpha=0.15, color='gray')
# plt.legend()
ax.set_xlabel("Days post-trigger")
plt.tight_layout()
plt.savefig("lightcurve.png")
plt.close()
    
fig = plt.figure()
freq = 9
curdata = plotdata[plotdata['freq']==9]
xdata = curdata['obsdate']
ydata = curdata['flux']*1e-6
yerr = np.sqrt(curdata['err']**2 + curdata['rms']**2)*1e-6
plt.scatter(xdata,ydata,label=f'{freq} GHz')
plt.errorbar(xdata,ydata,yerr=yerr,fmt=' ')
ax = plt.gca()
ax.set_title(f'{freq} GHz')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel("Flux Density (Jy)")
ax.set_ylim(1e-5,3e-3)
for o in np.sort(curdata['obs']):
    subdata = curdata[curdata['obs']==o]
    print(subdata)
    startobs = subdata['startdate'].min()
    endobs = subdata['stopdate'].max()
    plt.axvspan(startobs,endobs, alpha=0.15, color='gray')
ax.set_xlabel("Days post-trigger")
plt.tight_layout()
plt.savefig("9GHzlc.png")
plt.close()
