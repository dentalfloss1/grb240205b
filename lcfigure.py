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

fig = plt.figure()

for freq in np.sort(np.unique(plotdata['freq'])):
    curdata = plotdata[plotdata['freq']==freq]
    xdata = curdata['obsdate']
    print(xdata)
    ydata = curdata['flux']*1e-6
    yerr = curdata['err']*1e-6
    p = plt.plot(xdata,ydata,'.-',label=f'{freq} GHz')
    plt.errorbar(xdata,ydata,yerr=yerr,fmt=' ',color=p[-1].get_color())
ax = plt.gca()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel("Days post trigger")
ax.set_ylabel("Flux Density (Jy)")
ax.set_ylim(1e-5,3e-3)
plt.legend()
plt.savefig("lightcurve.png")
plt.close()
    
