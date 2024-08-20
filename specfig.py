import matplotlib.pyplot as plt
import pandas as pd 
import datetime
import numpy as np
from astropy.modeling.powerlaws import SmoothlyBrokenPowerLaw1D as sbpl
from scipy.optimize import curve_fit

trigger = datetime.datetime(2024, 2, 5, 22, 15, 8, 00)
def wrap_sbpl(t,amp, tb, a1, a2, d):
    f = sbpl(amplitude=amp, x_break=tb, alpha_1=a1, alpha_2=a2, delta=d)
    return f(t)

plotdata = pd.read_csv("spectra.csv")
numplots = plotdata['obs'].max()
nrows = int(np.sqrt(numplots))
ncols = int(np.ceil(np.sqrt(numplots)))
fig,axs = plt.subplots(nrows,ncols,sharex=True,sharey=True,figsize=(22,22))
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
            p = a.scatter(measx,measy,label=f"obs {plotnum}",color='black')
            a.errorbar(measx,measy,yerr=measyerr,fmt=' ',color='black')
            xdata = measx
            ydata = measy
            xline = np.linspace(xdata.min(), xdata.max(),num=1000)
            initial_guess = [ydata.max(),xdata[ydata==ydata.max()].to_numpy()[0],-0.8,0.8,0.2]
            bounds = [(1e-4,1e-3),(0.5,100),(-2,-0.2),(0.2,2),(0.1,0.8)]
            bounds0 = tuple([b[0] for b in bounds])
            bounds1 = tuple([b[1] for b in bounds])
            bounds = [bounds0,bounds1]
    
            popt, pcov = curve_fit(wrap_sbpl, xdata, ydata, p0=initial_guess,bounds=bounds)
            a.set_ylim(1e-5,3e-3)
            a.set_xlim(1,25)
            xline = np.linspace(1,25,num=1000)
            yline = wrap_sbpl(xline, *popt)
            a.plot(xline,yline,color='black',alpha=0.5)
            title=f"{measureddata['date'].to_numpy()[0]} days\nF0: {popt[0]:.2E}, nubreak: {popt[1]:.2E}\nalpha1: {popt[2]:.2E}, alpha2: {popt[3]:.2E}\ndelta: {popt[4]:.2E}"
            a.set_title(title)
            if limitdata.size > 0:
                a.scatter(limx,3*limy,marker='v',color=p[-1].get_color())
            a.set_xscale('log')
            a.set_yscale('log')
            if rowno==(nrows-1):
                a.set_xlabel("Frequency (GHz)")
            if colno==0:
                a.set_ylabel("Flux (Jy)")
        plotnum += 1
# plt.legend()
plt.savefig("specta.png")
plt.close()
# fig = plt.figure()
# for plotnum in [8,9]:
#     a = plt.gca()
#     measureddata = plotdata[(plotdata['obs']==plotnum) & (plotdata['flux'] > 0)]
#     limitdata = plotdata[(plotdata['obs']==plotnum) & (plotdata['flux'] < 0)]
#     print(plotnum,measureddata,limitdata)
#     measx = measureddata['freq']
#     measy = measureddata['flux']*1e-6
#     limx = limitdata['freq']
#     limy = limitdata['rms']*1e-6
#     measyerr = np.sqrt(measureddata['ferr']**2 + measureddata['rms']**2)*1e-6
#     print(measyerr)
#     print(measx)
#     p = a.plot(measx,measy,label=f"obs {plotnum}")
#     a.errorbar(measx,measy,yerr=measyerr,fmt=' ',color=p[-1].get_color())
#     a.set_title(f"{measureddata['date'].to_numpy()[0]} days")
#     if limitdata.size > 0:
#         a.scatter(limx,3*limy,marker='v',color=p[-1].get_color())
#     a.set_xscale('log')
#     a.set_yscale('log')
#     a.set_xlabel("Frequency (GHz)")
#     a.set_ylabel("Flux (Jy)")
#     a.set_ylim(1e-5,3e-3)
#     # plt.legend()
#     plt.savefig(f"specta_{plotnum}.png")
#     plt.close()
