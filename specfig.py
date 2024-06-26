import matplotlib.pyplot as plt
import pandas as pd 
import datetime
import numpy as np

trigger = datetime.datetime(2024, 2, 5, 22, 15, 8, 00)

plotdata = pd.read_csv("spectra.csv")
numplots = plotdata['obs'].max()
nrows = int(np.sqrt(numplots))
ncols = int(np.ceil(np.sqrt(numplots)))
fig,axs = plt.subplots(nrows,ncols,sharex=True,sharey=True,figsize=(10,10))
plotnum = 1 
for rowno,ax in enumerate(axs):
    for colno,a in enumerate(ax):
        if plotnum > numplots:
            fig.delaxes(a)
        else:
            measureddata = plotdata[(plotdata['obs']==plotnum) & (plotdata['flux'] > 0)]
            limitdata = plotdata[(plotdata['obs']==plotnum) & (plotdata['flux'] < 0)]
            print(plotnum,measureddata,limitdata)
            measx = measureddata['freq']
            measy = measureddata['flux']*1e-6
            limx = limitdata['freq']
            limy = limitdata['rms']*1e-6
            measyerr = np.sqrt(measureddata['ferr']**2 + measureddata['rms']**2)*1e-6
            print(measyerr)
            print(measx)
            p = a.plot(measx,measy,label=f"obs {plotnum}")
            a.errorbar(measx,measy,yerr=measyerr,fmt=' ',color=p[-1].get_color())
            a.set_title(f"{measureddata['date'].to_numpy()[0]} days")
            if limitdata.size > 0:
                a.scatter(limx,3*limy,marker='v',color=p[-1].get_color())
            a.set_xscale('log')
            a.set_yscale('log')
            if rowno==(nrows-1):
                a.set_xlabel("Frequency (GHz)")
            if colno==0:
                a.set_ylabel("Flux (Jy)")
            a.set_ylim(1e-5,3e-3)
        plotnum += 1
# plt.legend()
plt.savefig("specta.png")
plt.close()
    
