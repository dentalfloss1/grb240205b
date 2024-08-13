import matplotlib.pyplot as plt
import pandas as pd 
import datetime
import numpy as np
import itertools
from astropy.modeling.powerlaws import SmoothlyBrokenPowerLaw1D as sbpl
from scipy.optimize import curve_fit
trigger = datetime.datetime(2024, 2, 5, 22, 15, 8, 00)

plotdata = pd.read_csv("grbmeas.csv")
startdate = [(datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S.%f") - trigger).total_seconds()/3600/24 for d in plotdata['start']]
stopdate = [(datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S.%f") - trigger).total_seconds()/3600/24 for d in plotdata['stop']]
obsdur = [(t2-t1) for t1,t2 in zip(startdate,stopdate)]
plotdata['obsdate'] = [t1+(dur/2) for t1,dur in zip(startdate,obsdur)]
plotdata['startdate'] = startdate
plotdata['stopdate'] = stopdate
# plotdata = plotdata[np.isin(plotdata['freq'],[5.5,9.0])]
fig,axs = plt.subplots(6,1,figsize=(7,15),sharex=True,sharey=True)

def wrap_sbpl(t,amp, tb, a1, a2, d):
    f = sbpl(amplitude=amp, x_break=tb, alpha_1=a1, alpha_2=a2, delta=d)
    return f(t)
bands = ["L","S","C","X","Ku","K"]
for band,ax in zip(bands,axs):
    curdata = plotdata[plotdata['band']==band]
    xdata = curdata['obsdate']
    ydata = curdata['flux']*1e-6
    yerr = np.sqrt(curdata['err']**2 + curdata['rms']**2)*1e-6
    marker = itertools.cycle((',', '+', '.', 'o', '*'))
    for freq in np.unique(curdata['freq']):
        subcurdata = curdata[curdata['freq']==freq]
        subxdata = subcurdata['obsdate']
        subydata = subcurdata['flux']*1e-6
        subyerr = np.sqrt(subcurdata['err']**2 + subcurdata['rms']**2)*1e-6
        ax.errorbar(subxdata,subydata,yerr=subyerr,fmt=' ',color='black')
        ax.scatter(subxdata,subydata,label=f'{freq} GHz',marker=next(marker),color='black')
    xline = np.linspace(xdata.min(), xdata.max(),num=1000)
    initial_guess = [ydata.max(),xdata[ydata==ydata.max()].to_numpy()[0],-0.8,0.8,0.2]
    bounds = [(1e-4,1e-3),(1e-1,100),(-2,-0.2),(0.2,2),(0.1,0.8)]
    bounds0 = tuple([b[0] for b in bounds])
    bounds1 = tuple([b[1] for b in bounds])
    bounds = [bounds0,bounds1]

    popt, pcov = curve_fit(wrap_sbpl, xdata, ydata, p0=initial_guess,bounds=bounds)
    yline = wrap_sbpl(xline, *popt)
    ax.plot(xline,yline,color='black',alpha=0.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel("Flux Density (Jy)")
    ax.set_ylim(1e-5,3e-3)
    if len(np.unique(curdata['freq'])) >1:
        freq1 = curdata['freq'].min()
        freq2 = curdata['freq'].max()
        title=f'{freq1} to {freq2} GHz\nF0: {popt[0]:.2E}, tbreak: {popt[1]:.2E}\nalpha1: {popt[2]:.2E}, alpha2: {popt[3]:.2E}, delta: {popt[4]:.2E}'
        ax.set_title(title)
    else:
        title=f'{freq} GHz\nF0: {popt[0]:.2E}, tbreak: {popt[1]:.2E}\nalpha1: {popt[2]:.2E}, alpha2: {popt[3]:.2E}, delta: {popt[4]:.2E}'
        ax.set_title(title)
    ax.legend()
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
