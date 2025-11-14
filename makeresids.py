import matplotlib.pyplot as plt
import traceback
import pandas as pd 
import datetime
import numpy as np
import itertools
from tqdm import tqdm
from astropy.modeling.powerlaws import SmoothlyBrokenPowerLaw1D as sbpl
from scipy.optimize import curve_fit
from scipy.stats import linregress
import argparse
trigger = datetime.datetime(2024, 2, 5, 22, 13, 6, 00)

checkdata = pd.read_csv("checksrc.csv")
plotdata = pd.read_csv("grbmeas.csv")
startdate = [(datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S.%f") - trigger).total_seconds()/3600/24 for d in plotdata['start']]
stopdate = [(datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S.%f") - trigger).total_seconds()/3600/24 for d in plotdata['stop']]
obsdur = [(t2-t1) for t1,t2 in zip(startdate,stopdate)]
plotdata['obsdate'] = [t1+(dur/2) for t1,dur in zip(startdate,obsdur)]
plotdata['startdate'] = startdate
plotdata['stopdate'] = stopdate
startdate = [(datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S.%f") - trigger).total_seconds()/3600/24 for d in checkdata['start']]
stopdate = [(datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S.%f") - trigger).total_seconds()/3600/24 for d in checkdata['stop']]
obsdur = [(t2-t1) for t1,t2 in zip(startdate,stopdate)]
checkdata['obsdate'] = [t1+(dur/2) for t1,dur in zip(startdate,obsdur)]
checkdata['startdate'] = startdate
checkdata['stopdate'] = stopdate
# plotdata = plotdata[np.isin(plotdata['freq'],[5.5,9.0])]
parser = argparse.ArgumentParser()
parser.add_argument("--forwardOnly",action="store_true")
parser.add_argument("--noerrors",action="store_true")
parser.add_argument("--k", type=int, default=2, help="Value for k, (use 0 or 2)")
args = parser.parse_args()
def powerlaw(t,a,k):
    return a*t**k
def wrap_sbpl(t,amp, tb, a1, a2, d):
    f = sbpl(amplitude=amp, x_break=tb, alpha_1=a1, alpha_2=a2, delta=d)
    return f(t)
def double_sbpl(x,amp,x_break1,a1,n1,a2,n2,x_break2):
    return amp*x_break1


def dsbpl(x,amp,xb1,a1,a2,xb2,a3,s):
    if (a3 < a1) & (a1 < a2):
        result =  amp*(((x/xb1)**(a1*s) + (x/xb1)**(a2*s))**(-1) + (xb2/xb1)**(-a2*s)*(x/xb2)**(-a3*s))**(-1/s)
    elif (a3 < a2) & (a2 < a1):
        result = amp*((x/xb1)**(-a1*s) + (x/xb1)**(-a2*s) + ((xb1/xb2)**(a2*s))*((x/xb2)**(-a3*s)) )**(-1/s)
    else:
        raise Exception("Unhandled powerlaw index ordering")
    return result

def verify_powerlaw(x,amp,xb1,a1,a2,xb2,a3):
    result = x
    amp2= amp*(xb2/xb1)**(a2)
    result = np.where( x <= xb1, amp*(x/xb1)**(a1),result)
    result = np.where( (x >  xb1)  & (x <xb2), amp2*(x/xb2)**(a2),result)
    result = np.where( x>=xb2, amp2*(x/xb2)**(a3),result)
    return result


bands = ["L","S","C","X","Ku","K"]
tpk = []
nupk = []
xbanddata = plotdata[(plotdata['obsdate'] <1)]
# print(xbanddata)
tdata = xbanddata['obsdate']
fdata = xbanddata['flux']
ferrdata = xbanddata['err']
# initial_guess = [400,0.031,-2,1,0.1]
# bounds = [(100,1000),(0.024,0.045),(-4,-1),(0.1,1),(0.09,0.11)]
# bounds0 = tuple([b[0] for b in bounds])
# bounds1 = tuple([b[1] for b in bounds])
# bounds = [bounds0,bounds1]
# popt, pcov = curve_fit(wrap_sbpl, tdata, fdata, p0=initial_guess,bounds=bounds,sigma=ferrdata)
fig = plt.figure()
# y = dsbpl(x,700,1, 2,1/3, 100, -0.5,10)
# y2 = dsbpl(7,700,1, 2,1/3, 100, -0.5,10)*dsbpl(x,700,1, 2,5/2, 100, -0.5,10)/dsbpl(7,700,1, 2,5/2, 100, -0.5,10)
print("f0=700,tb1=1,tb2=10")
x = tdata
y = fdata
x2 = np.geomspace(x.min(),x.max(),num=100_000)
y2 = wrap_sbpl(x2, 250, 0.035,-1/2, 1/2, 0.1)
plt.scatter(x,y,marker='x',label="dsbpl",color='tab:orange',alpha=0.5)
plt.plot(x2,y2,marker='x',label="sbpl",color='tab:blue',alpha=0.5)
# plt.errorbar(tdata,fdata,yerr=ferrdata,ls='none',marker='o',color='black')
# plt.plot(x,wrap_sbpl(x,*popt),color="tab:blue", label='sbpl')
# varnames = ["t0","gamma1","gamma2","delta"]
# text = f"f0={popt[0]}+/-{np.absolute(pcov[0][0])**0.5}"
# print(text)
# for ind,var in enumerate(varnames):
#     vnum = ind+1
#     text = f" {var}={popt[vnum]}+/-{np.absolute(pcov[vnum][vnum])**0.5}"
#     print(text)
ax = plt.gca()
ax.legend()
# ax.axvline(1)
# ax.axhline(700)
# ax.axvline(10,color='red',ls=':')
# ax.axhline(700*(10/1)**(5/2),color='red',ls=':')
ax.set_xscale('log')
ax.set_yscale('log')
# plt.show()
plt.close()
# exit()
def get_tbreak(ivar, f0, nu0_1, nu0_2, k):
    t0 = 1
    d=0.4
    s = 10
    a1 = -k/(2*(4-k))
    b1 = -3*k/(5*(4-k))
    b2 = -3/2
    t, nu = ivar
    res = []
    t_break = np.amax(t)
    for tval in np.sort(t):
        nua = nu0_1*(tval/t0)**b1
        num = nu0_2*(tval/t0)**b2
        if num <= nua:
            t_break = tval
            break
    return t_break
def theory_bigsbpl(ivar, f0, nu0_1, nu0_2, k):
    t0 = 1
    d=0.4
    s = 10
    a1 = -k/(2*(4-k))
    b1 = -3*k/(5*(4-k))
    b2 = -3/2
    p = 2.3
    nu0_3 = 1e9
    t, nu = ivar
    y1 = []
    y2 = []
    y3 = []
    t_break = np.amax(t)
    nu_trans = nu0_2
    for tval in np.sort(t):
        nua = nu0_1*(tval/t0)**b1
        num = nu0_2*(tval/t0)**b2
        if num <= nua:
            t_break = tval
            nu_trans = nua
            break
    for tval,nuval in zip(t,nu):
        b1_1 = -3*k/(5*(4-k))
        b2_1 = -3/2
        b3_1 = -(4-3*k)/(2*(4-k))
        nua_1 = nu0_1*(tval/t0)**b1_1
        num_1 = nu0_2*(tval/t0)**b2_1
        nuc_1 = nu0_3*(tval/t0)**b3_1
        if nuval < nua_1:
            c1_1 = 2
            c2_1 = 1/3
            c3_1 = -(p-1)/2
            fnu_m1 = f0*(tval/t0)**a1
            fpk_1 = fnu_m1*(nua_1/num_1)**(1/3)
            nubreak1_1 = nua_1
            nubreak2_1 = num_1
        else:
            fnu_m1 = f0*(tval/t0)**a1
            c1_1 = 1/3
            c2_1 = -(p-1)/2
            c3_1 = -1
            fpk_1 = f0*(tval/t0)**a1
            nubreak1_1 = num_1
            nubreak2_1 = nuc_1
        a1_2 = -k/(2*(4-k))
        b1_2 = -3/2
        b2_2 = -(12*p+8-3*p*k+2*k)/(2*(4-k)*(p+4))
        c1_2 = 2
        c2_2 = 5/2
        c3_2 = -(p-1)/2
        # At t_break num==nua, determines the normalization of these values.
        num_2 = nu_trans*(tval/t0)**b1_2
        nua_2 = nu_trans*(tval/t0)**b2_2
        num_break2 = nu_trans*(t_break/t0)**b1_2
        nua_break2 = nu_trans*(t_break/t0)**b2_2
        fnu_m_2 = f0*(tval/t0)**a1_2
        fpk = fnu_m_2*(num_2/nua_2)**(3)
        fpk_2 = fpk
        res1 = dsbpl(nuval,fpk_1,nubreak1_1,c1_1,c2_1,nubreak2_1,c3_1,s)

        c1_1 = 2
        c2_1 = 1/3
        c3_1 = -(p-1)/2
        num_1 = nu0_1*(t_break/t0)**b1_1
        nua_1 = nu0_2*(t_break/t0)**(b2_1)
        fnu_m1 = f0*(t_break/t0)**a1
        fpk_1 = fnu_m1*(nua_1/num_1)**(1/3)
        F_bk1 = dsbpl(nuval,fpk_1,num_1,c1_1,c2_1,nua_1,c3_1,s)
        fpk2_break = f0*(t_break/t0)**a1_2*(num_break2/nua_break2)**(3)
        F_bk2 = dsbpl(nuval,fpk2_break,num_break2,c1_2,c2_2,nua_break2,c3_2,s)
        res2 = F_bk1*dsbpl(nuval,fpk_2,num_2,c1_2,c2_2,nua_2,c3_2,s)/F_bk2
        # res2 = dsbpl(nuval,fpk_2,num_2,c1_2,c2_2,nua_2,c3_2,s)
        res3 = dsbpl(nuval,fpk_2,num_2,c1_2,c2_2,nua_2,c3_2,s)
        y1.append(res1)
        y2.append(res2)
    result = np.where( t<=t_break,np.array(y1),np.array(y2))
    return result
# Relativistic Rev. Shock
def reverse_shock(ivar, f0, nu0_1,k,givenuvals=False):
    t, nu = ivar
    s=10
    res = []
    p=2.3
    t0=0.05
    nu0_2 = 1e8
    nu0_3 = 1e9
    a1 = -(47-10*k)/(12*(4-k))
    b1 = -(32-7*k)/(15*(4-k))
    b2 = -(73-14*k)/(12*(4-k))
    b3 = -(73-14*k)/(12*(4-k))
    nuvals = []
    try:
        for tval,nuval in zip(t,nu):
            fnu_m = f0*(tval/t0)**a1
            nua = nu0_1*(tval/t0)**b1
            num = nu0_2*(tval/t0)**b2
            nuc = nu0_3*(tval/t0)**b3
            if nuval < nua:
                c1 = 2 
                c2 = 1/3
                c3 = -(p-1)/2
                fpk = fnu_m*(nua/num)**(1/3)
                result = dsbpl(nuval,fpk,nua,c1,c2,num,c3,s)
            else:
                c1 = 1/3
                c2 = -(p-1)/2
                c3 = -(p-1)/2 - 0.1
                fpk = fnu_m
                result = dsbpl(nuval,fpk,num,c1,c2,nuc,c3,s)
            res.append(result)
            if givenuvals:
                nuvals.append((tval,nua,num,nuc))
    except Exception as e:
        print(traceback.format_exc())
        print(t,nu,ivar)
    if givenuvals:
        return np.array(res),np.array(nuvals) 
    else:
        return np.array(res)
def wrap_bigsbpl(ivar, f0,nu01,nu02):
    t0 = 1
    s = 10
    d = 0.2
    k=args.k
    t, nu = ivar
    res = []
    f = theory_bigsbpl(ivar, f0, nu01, nu02, k)
    return f
initial_guess = [1e-3,10,50]
bounds = [(1e-6,1),(1,100),(15,1e5)]
bounds0 = tuple([b[0] for b in bounds])
bounds1 = tuple([b[1] for b in bounds])
bounds = [bounds0,bounds1]
curdata = plotdata[plotdata['obsdate']>10]
tdata = curdata['obsdate']
nudata = curdata['freq']
xdata = (tdata,nudata)
ydata = curdata['flux']*1e-6
yerr = np.sqrt(curdata['err']**2 + curdata['rms']**2)*1e-6
popt, pcov = curve_fit(wrap_bigsbpl, xdata, ydata, p0=initial_guess,bounds=bounds,sigma=yerr)
varnames = ["nua_0","num_0"]
text = f"f0={popt[0]}+/-{np.absolute(pcov[0][0])**0.5}"
print(text)
for ind,var in enumerate(varnames):
    vnum = ind+1
    text = f" {var}={popt[vnum]}+/-{np.absolute(pcov[vnum][vnum])**0.5}"
    print(text)
print("d=0.2")
bigpopt = popt

print(get_tbreak(xdata, bigpopt[0], bigpopt[1], bigpopt[2], args.k))
fig,axs = plt.subplots(3,2,figsize=(15,15),sharex=True,sharey=True)
def getminmax(ivar, bigpopt, bigsigma):
    minarray = []
    maxarray = []
    for i,tval in tqdm(enumerate(ivar[0]),total=len(ivar[0])):
        meas = []
        for f0try in [bigpopt[0]+bigsigma[0],bigpopt[0]-bigsigma[0]]:
            for frevtry in [bigpopt[1]+bigsigma[1],bigpopt[1]-bigsigma[1]]:
                for nu01revtry in [bigpopt[2]+bigsigma[2],bigpopt[2]-bigsigma[2]]:
                    for nu01try in [bigpopt[3]+bigsigma[3],bigpopt[3]-bigsigma[3]]:
                        for nu02try in [bigpopt[4]+bigsigma[4],bigpopt[4]-bigsigma[4]]:
                            fully = wrap_bigsbpl(ivar,f0try,frevtry,nu01revtry,nu01try,nu02try)[i]
                            meas.append(fully)
        minarray.append(np.amin(meas))
        maxarray.append(np.amax(meas))
     
    return np.array(minarray), np.array(maxarray)

for band,ax in zip(bands,axs.flatten()):
    curdata = plotdata[plotdata['band']==band]
    if band in ['X']:
        curdata = plotdata[(plotdata['band']=="X") | (plotdata['band']=="X1")]
    # print(curdata)
    xdata = curdata['obsdate']
    ydata = curdata['flux']
    yerr = np.sqrt(curdata['err']**2 + curdata['rms']**2)
    marker = itertools.cycle((',', '+', '.', 'o', '*'))
    linestyle = itertools.cycle(('-', ':', '-.', '--'))
    xline = np.geomspace(1e-2,365,num=100)
    for freq in np.sort(np.unique(curdata['freq']))[::-1]:
       if band in ['X']:
           subcurdata = curdata[(curdata['freq']==freq) & (curdata['obsdate'] < 1) & (curdata['band']=="X")]
           subxdata = subcurdata['obsdate']
           subydata = subcurdata['flux']
           subyerr = np.sqrt(subcurdata['err']**2 + subcurdata['rms']**2)
           ax.errorbar(subxdata,subydata,yerr=subyerr,fmt=' ',color='black')
           ax.scatter(subxdata,subydata,label=f'{freq} GHz uvfit',facecolors='none',edgecolor='black')
          #  subcurdata = curdata[(curdata['freq']==freq) & (curdata['obsdate'] < 1) & (curdata['band']=="X1")]
          #  subxdata = subcurdata['obsdate']
          #  subydata = subcurdata['flux']
          #  subyerr = np.sqrt(subcurdata['err']**2 + subcurdata['rms']**2)
          #  ax.errorbar(subxdata,subydata,yerr=subyerr,fmt=' ',color='black',alpha=0.5)
          #  ax.scatter(subxdata,subydata,label=f'{freq} GHz uvfit',marker='s',facecolors='none',edgecolor='black',alpha=0.5)
           subcurdata = curdata[(curdata['freq']==freq) & (curdata['obsdate'] >= 1)]
           subxdata = subcurdata['obsdate']
           subydata = subcurdata['flux']
           subyerr = np.sqrt(subcurdata['err']**2 + subcurdata['rms']**2)
           ax.errorbar(subxdata,subydata,yerr=subyerr,fmt=' ',color='black')
           ax.scatter(subxdata,subydata,label=f'{freq} GHz',color='black')

          #  yline = wrap_thinbigsbpl((xline,nu), *thinpopt)*1e6
          #  ax.plot(xline,yline,alpha=0.5,color='black',ls=next(linestyle),label=f"{freq} GHz model, Thin Shell")
          #  subcheck = checkdata[checkdata['band']==band]
          #  checkerr = np.sqrt(subcheck['err']**2 + subcheck['rms']**2)
          #  ax.errorbar(subcheck['obsdate'],subcheck['flux'],yerr=checkerr,fmt=' ',color='black',marker='x', label='check source')
       else:
           subcurdata = curdata[(curdata['freq']==freq) & (curdata['err']!=-1)]
           subxdata = subcurdata['obsdate']
           subydata = subcurdata['flux']
           subyerr = np.sqrt(subcurdata['err']**2 + subcurdata['rms']**2)
           ax.errorbar(subxdata,subydata,yerr=subyerr,fmt=' ',color='black')
           ax.scatter(subxdata,subydata,label=f'{freq} GHz',marker=next(marker),color='black')
           limitdata = curdata[(curdata['freq']==freq) & (curdata['err']==-1)]
           limitxdata = limitdata['obsdate']
           limitydata = limitdata['rms']*3
           if band=='C':
               subcheck = checkdata[checkdata['band']==band]
               checkerr = np.sqrt(subcheck['err']**2 + subcheck['rms']**2)
               ax.errorbar(subcheck['obsdate'],subcheck['flux'],yerr=checkerr,fmt=' ',color='black',marker='x', label='check source')
           nu = np.array([freq for f in xline])
           yline = wrap_bigsbpl((xline,nu), *bigpopt)*1e6
        
    nu = np.array([freq for f in xline])
    yline = wrap_bigsbpl((xline,nu), *bigpopt)*1e6
    ax.plot(xline,yline,alpha=0.5,color='black',ls=next(linestyle),label=f"{freq} GHz Forward")
   #  ax.fill_between(xline,bound_lower,bound_upper,color='black',alpha=0.15)
   #  yline = wrap_bigsbpl((xline,nu), *bigpopt)
   #  ax.plot(xline,yline,alpha=0.5,color='black')
 #    trypopt = [1.8e-3, 355e-6, 9.5, 50, 400]
 #    yline = wrap_bigsbpl((xline,nu), *trypopt)
 #    ax.plot(xline,yline,alpha=0.5,color='black',label=f'tryfit',ls=':')
   #      
   #  
   #  popt, pcov = curve_fit(wrap_sbpl, xdata, ydata, p0=initial_guess,bounds=bounds)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel("Flux Density ($\mu Jy$)")
    ax.set_xlim(1e-2,365)
    ax.set_ylim(10,3000)
    # test_theory = [1e-3, 18.5, 60.3 , 2]
    # print(test_theory)
    # ax.plot(xline,yline,color='black',alpha=0.5,ls=':',label='theory')
    # ax.plot(xline,yline,color='black',alpha=0.5,ls='-')


#     test_theory = [1.e-3, 58 , 95 , 0]
#     print(test_theory)
    
#     ax.axvline(t_break,ls=':')
#     yline = theory_bigsbpl((xline,nu), *test_theory)
#     ax.plot(xline,yline,color='black',alpha=0.5,ls=':',label='ISM')
    # yline = wrap_latebigsbpl((xline,nu), *latebigpopt)
#     ax.plot(xline,yline,color='black',ls='--',alpha=0.5,label='nonrel')
 #    if freq==9.0:
#         yline = wrap_earlybigsbpl(xline, *earlybigpopt)
     #    ax.plot(xline,yline,color='black',ls='--',alpha=0.5,label='Reverse Shock')
# def wrap_latebigsbpl(ivar, f0, nu0, a1, b1, c1, c2):
    if len(np.unique(curdata['freq'])) >1:
        freq1 = curdata['freq'].min()
        freq2 = curdata['freq'].max()
        title=f'{freq1} to {freq2} GHz'
        ax.set_title(title)
    else:
        title=f'{freq} GHz'
        ax.set_title(title)
    ax.legend()
   #  if (freq < 16) and (freq >2):
   #      tpk.append(popt[1])
   #      nupk.append(np.average(curdata['freq']))
    for o in np.sort(np.unique(curdata['obs'])):
   
        subdata = curdata[curdata['obs']==o]
        startobs = subdata['startdate'].min()
        endobs = subdata['stopdate'].max()
        ax.axvspan(startobs,endobs, alpha=0.15, color='gray')
# plt.legend()
ax.set_xlabel("Days post-trigger")
if args.k==2:
    fig.suptitle(f"Stellar Wind Profile")
elif args.k==0:
    fig.suptitle(f"ISM Profile")
plt.tight_layout()
plt.savefig("forwardfit.png")
plt.close()
fig,axs = plt.subplots(3,2,figsize=(15,15),sharex=True,sharey=True)
for band,ax in zip(bands,axs.flatten()):
    curdata = plotdata[plotdata['band']==band]
    if band in ['X']:
        curdata = plotdata[(plotdata['band']=="X") | (plotdata['band']=="X1")]
    # print(curdata)
    xdata = curdata['obsdate']
    ydata = curdata['flux']
    yerr = np.sqrt(curdata['err']**2 + curdata['rms']**2)
    marker = itertools.cycle((',', '+', '.', 'o', '*'))
    linestyle = itertools.cycle(('-', ':', '-.', '--'))
    xline = np.geomspace(1e-2,365,num=100)
    for freq in np.sort(np.unique(curdata['freq']))[::-1]:
        nu = np.array([freq for f in xline])
        yline = wrap_bigsbpl((xline,nu), *bigpopt)*1e6
        subcurdata = curdata[(curdata['freq']==freq) & (curdata['err']!=-1)]
        subxdata = subcurdata['obsdate']
        yforward = []
        for xval in subxdata:
            yforward.append(yline[np.abs(xline-xval)==np.amin(np.abs(xline-xval))][0])
        yforward = np.array(yforward)
        subydata = subcurdata['flux']
        subyerr = np.sqrt(subcurdata['err']**2 + subcurdata['rms']**2)
        ax.scatter(subxdata,(subydata-yforward),label=f'{freq} GHz',marker=next(marker),color='black')
        # axt = ax.twinx()
        # axt.plot(xline,yline)
        limitdata = curdata[(curdata['freq']==freq) & (curdata['err']==-1)]
        limitxdata = limitdata['obsdate']
        limitydata = limitdata['rms']*3
        nu = np.array([freq for f in xline])
        yline = wrap_bigsbpl((xline,nu), *bigpopt)*1e6
        

    ax.set_xscale('log')
    ax.set_yscale('symlog')
    # ax.set_ylim(-1,1)
    ax.set_ylabel("Residual Flux Density ($\mu Jy$)")
    ax.set_xlim(1e-2,365)
    ax.grid()
    ax.axhline(0)
    if len(np.unique(curdata['freq'])) >1:
        freq1 = curdata['freq'].min()
        freq2 = curdata['freq'].max()
        title=f'{freq1} to {freq2} GHz'
        ax.set_title(title)
    else:
        title=f'{freq} GHz'
        ax.set_title(title)
    ax.axvspan(1e-2,10, alpha=0.15, color='gray')
    ax.legend()
   #  if (freq < 16) and (freq >2):
   #      tpk.append(popt[1])
   #      nupk.append(np.average(curdata['freq']))
# plt.legend()
ax.set_xlabel("Days post-trigger")
if args.k==2:
    fig.suptitle(f"Stellar Wind Profile")
elif args.k==0:
    fig.suptitle(f"ISM Profile")
plt.tight_layout()
plt.savefig("showresiduals.png")
plt.close()
