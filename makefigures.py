import matplotlib.pyplot as plt
import pandas as pd 
import datetime
import numpy as np
import itertools
from astropy.modeling.powerlaws import SmoothlyBrokenPowerLaw1D as sbpl
from scipy.optimize import curve_fit
from scipy.stats import linregress
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
tpk = []
nupk = []

def wrap_bigsbpl(ivar, f0, nu0, a1, b1, c1, c2, d):
    t0 = 1
    t, nu = ivar
    res = []
    for tval,nuval in zip(t,nu):
        fpk = f0*(tval/t0)**-a1
        nupk = nu0*(tval/t0)**-b1
        f = sbpl(amplitude=fpk, x_break=nupk, alpha_1=c1, alpha_2=c2, delta=d)
        res.append(f(nuval))
    return np.array(res)
initial_guess = [plotdata['flux'].max()*1e-6,25,1,1,-2,1,0.2]
bounds = [(1e-4,1),(1,100),(0.1,4),(0.1,4),(-4,-0.1),(0.1,4),(0.1,0.5)]
bounds0 = tuple([b[0] for b in bounds])
bounds1 = tuple([b[1] for b in bounds])
bounds = [bounds0,bounds1]
tdata = plotdata['obsdate']
nudata = plotdata['freq']
xdata = (tdata,nudata)
ydata = plotdata['flux']*1e-6
popt, pcov = curve_fit(wrap_bigsbpl, xdata, ydata, p0=initial_guess,bounds=bounds)
print("f0:",popt[0],"nu0:",popt[1],f"F propto t**(-{popt[2]}",f"nu propto t**(-{popt[3]}","alpha1:",-popt[4],"alpha2:",-popt[5],"smoothness:",popt[6])
bigpopt = popt
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
   #  xline = np.linspace(xdata.min(), xdata.max(),num=1000)
   #  initial_guess = [ydata.max(),xdata[ydata==ydata.max()].to_numpy()[0],-0.8,0.8,0.2]
   #  bounds = [(1e-4,1e-3),(1e-1,100),(-2,-0.2),(0.2,2),(0.1,0.8)]
   #  bounds0 = tuple([b[0] for b in bounds])
   #  bounds1 = tuple([b[1] for b in bounds])
   #  bounds = [bounds0,bounds1]
   #      
   #  
   #  popt, pcov = curve_fit(wrap_sbpl, xdata, ydata, p0=initial_guess,bounds=bounds)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel("Flux Density (Jy)")
    ax.set_xlim(1e-1,365)
    ax.set_ylim(1e-5,3e-3)
    xline = np.linspace(1e-1,365,num=1000)
    nu = np.array([freq for f in xline])
    yline = wrap_bigsbpl((xline,nu), *popt)
    ax.plot(xline,yline,color='black',alpha=0.5)
    if len(np.unique(curdata['freq'])) >1:
        freq1 = curdata['freq'].min()
        freq2 = curdata['freq'].max()
        title=f'{freq1} to {freq2} GHz'
        ax.set_title(title)
    else:
        title=f'{freq} GHz'
        ax.set_title(title)
    ax.legend()
    if (freq < 16) and (freq >2):
        tpk.append(popt[1])
        nupk.append(np.average(curdata['freq']))
    for o in np.sort(curdata['obs']):
        subdata = curdata[curdata['obs']==o]
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
    startobs = subdata['startdate'].min()
    endobs = subdata['stopdate'].max()
    plt.axvspan(startobs,endobs, alpha=0.15, color='gray')
ax.set_xlabel("Days post-trigger")
plt.tight_layout()
plt.savefig("9GHzlc.png")
plt.close()

tpk = [1.4,6.3,11.0,17,23,54,75,129,160]
nupk =[16.3,11.1,16.1,4.07,9.47,5.5,3.25,2.51,3.58]

def powerlaw(t,a,k):
    return a*t**k
# fit lcpk
fig = plt.figure()
plt.scatter(tpk,nupk)
initial_guess = [9.0/(9.25**(-3/2))*0.1,-3/2]
bounds = [(10,500),(-3,-0.5)]
bounds0 = tuple([b[0] for b in bounds])
bounds1 = tuple([b[1] for b in bounds])
bounds = [bounds0,bounds1]
popt, pcov = curve_fit(powerlaw, tpk, nupk, p0=initial_guess,bounds=bounds)
xline = np.linspace(1e-1,365,num=1000)
yline = powerlaw(xline, *popt)
plt.plot(xline,yline,color='black',alpha=0.5)
ax = plt.gca()
ax.set_xscale('log')
ax.set_yscale('log')
title=f'nu_m  alpha:{popt[1]}, expectation: -1.5'
ax.set_title(title)
plt.savefig("nu_m_time.png")
plt.close()

popt = bigpopt
chisq = 0
dof = len(plotdata) - len(popt)
for d,nu,f,ferr,rms in plotdata[['obsdate','freq','flux','err','rms']].to_numpy():
    errtot = np.sqrt(ferr**2 + rms**2)*1e-6
    model = yline = wrap_bigsbpl((np.array([d]),np.array([nu])), *popt)
    chisq += ((f*1e-6 - model) / errtot)**2 / dof
print("red. chisq:",chisq,"dof:",dof)

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
            measx = measureddata['freq']
            measy = measureddata['flux']*1e-6
            limx = limitdata['freq']
            limy = limitdata['rms']*1e-6
            measyerr = np.sqrt(measureddata['ferr']**2 + measureddata['rms']**2)*1e-6
            p = a.scatter(measx,measy,label=f"obs {plotnum}",color='black')
            a.errorbar(measx,measy,yerr=measyerr,fmt=' ',color='black')
            xdata = measx
            ydata = measy
            xline = np.linspace(xdata.min(), xdata.max(),num=1000)
            tday = np.array([measureddata['date'].to_numpy()[0] for f in xline])
            popt = bigpopt
            yline = wrap_bigsbpl((tday,xline), *popt)
            a.set_ylim(1e-5,3e-3)
            a.set_xlim(1,25)
            a.plot(xline,yline,color='black',alpha=0.5)
            title=f"{measureddata['date'].to_numpy()[0]} days"
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


