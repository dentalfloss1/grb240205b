import matplotlib.pyplot as plt
import pandas as pd 
import datetime
import numpy as np

trigger = datetime.datetime(2024, 2, 5, 22, 15, 8, 00)

plotdata = pd.read_csv("spectra.csv")
numplots = plotdata['obs'].max()
nrows = int(np.sqrt(numplots))
ncols = int(np.ceil(np.sqrt(numplots)))
fig,axs = plt.subplots(nrows,ncols)
plotnum = 0 
for ax in axs:
    for a in ax: 
        if plotnum >= numplots:
            fig.delaxes(a)
        else:
            measureddata = plotdata[(plotdata['obs']==plotnum) & (plotdata['flux'] > 0)]
            limitdata = plotdata[(plotdata['obs']==plotnum) & (plotdata['flux'] < 0)]
            measx = measureddata['freq']
            measy = measureddata['flux']*1e-6
            measyerr = measureddata['err']*1e-6
            p = a.plot(measx,measy,'._',label=f"obs {plotnum+1}")
            a.errorbar(measx,measy,yerr=measyerr,fmt=' ',color=p[-1].get_color())
            limx = limitdata['freq']
            limy = limitdata['frms']*1e-6
            a.scatter(limx,limy,marker='v',color=p[-1].get_color())
            a.set_xscale('log')
            a.set_yscale('log')
            a.set_xlabel("Frequency (GHz)")
            a.set_ylim(1e-5,3e-3)
        plotnum += 1
plt.legend()
plt.savefig("specta.png")
plt.close()
    
