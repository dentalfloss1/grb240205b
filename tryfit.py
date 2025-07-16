import matplotlib.pyplot as plt
import pandas as pd 
import datetime
import numpy as np
import itertools
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
parser.add_argument("--freezeParams",action="store_true")
parser.add_argument("--k", type=int, default=2, help="Value for k, (use 0 or 2)")
args = parser.parse_args()
freezeParams=args.freezeParams
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
if freezeParams:
    def wrap_bigsbpl(ivar, f0, nu0, c1, c2):
        t0 = 1
        a1 = -0.43
        b1 = -0.5
        d = 0.4
        t, nu = ivar
        res = []
        for tval,nuval in zip(t,nu):
            fpk = f0*(tval/t0)**a1
            nupk = nu0*(tval/t0)**b1
            f = sbpl(amplitude=fpk, x_break=nupk, alpha_1=-c1, alpha_2=-c2, delta=d)
            res.append(f(nuval))
        return np.array(res)
    initial_guess = [plotdata['flux'].max()*1e-6,25,2,-1]
    bounds = [(1e-4,1),(1,100),(0.1,4),(-4,-0.1)]
    bounds0 = tuple([b[0] for b in bounds])
    bounds1 = tuple([b[1] for b in bounds])
    bounds = [bounds0,bounds1]
    tdata = plotdata['obsdate']
    nudata = plotdata['freq']
    xdata = (tdata,nudata)
    ydata = plotdata['flux']*1e-6
    popt, pcov = curve_fit(wrap_bigsbpl, xdata, ydata, p0=initial_guess,bounds=bounds)
    varnames = ["nu0","gamma1","gamma2"]
    text = f"f0={popt[0]}+/-{np.absolute(pcov[0][0])**0.5}"
    print(text)
    for ind,var in enumerate(varnames):
        vnum = ind+1
        text = f" {var}={popt[vnum]}+/-{np.absolute(pcov[vnum][vnum])**0.5}"
        print(text)
    print("alpha1=-0.43")
    print("beta1=-0.5")
    print("d=0.4")
else:
    
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
        p = 2
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
    def reverse_shock(ivar, f0, nu0_1,k):
        t, nu = ivar
        s=10
        res = []
        p=2
        t0=0.05
        nu0_2 = 100
        nu0_3 = 1e9
        a1 = -(47-10*k)/(12*(4-k))
        b1 = -(32-7*k)/(15*(4-k))
        b2 = -(73-14*k)/(12*(4-k))
        b3 = -(73-14*k)/(12*(4-k))
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
        return np.array(res)
    def wrap_bigsbpl(ivar, f0, frev, nu0rev,nu01,nu02):
        t0 = 1
        s = 10
        d = 0.2
        k=args.k
        t, nu = ivar
        res = []
        frev = reverse_shock(ivar, frev, nu0rev,k)
        f = theory_bigsbpl(ivar, f0, nu01, nu02, k)
        return frev + f
    initial_guess = [1e-3,5e-5, 10,10,50]
    bounds = [(1e-6,1),(3e-5,2),(1,100),(1,100),(15,1e5)]
    bounds0 = tuple([b[0] for b in bounds])
    bounds1 = tuple([b[1] for b in bounds])
    bounds = [bounds0,bounds1]
    curdata = plotdata[plotdata['band']!="X1"]
    tdata = curdata['obsdate']
    nudata = curdata['freq']
    xdata = (tdata,nudata)
    ydata = curdata['flux']*1e-6
    yerr = np.sqrt(curdata['err']**2 + curdata['rms']**2)*1e-6
    popt, pcov = curve_fit(wrap_bigsbpl, xdata, ydata, p0=initial_guess,bounds=bounds,sigma=yerr)
    varnames = ["f0_rev","nua0_rev","nua_0","num_0"]
    text = f"f0={popt[0]}+/-{np.absolute(pcov[0][0])**0.5}"
    print(text)
    for ind,var in enumerate(varnames):
        vnum = ind+1
        text = f" {var}={popt[vnum]}+/-{np.absolute(pcov[vnum][vnum])**0.5}"
        print(text)
    print("d=0.2")
    bigpopt = popt
    bigpcov = pcov
    bigsigma = np.sqrt(np.diagonal(pcov))
    # Non-relativistic Rev. Shock
    def nonrel_reverse_shock(ivar, f0, nu0_1,k):
        t, nu = ivar
        s=10
        res = []
        p=2
        t0=0.05
        nu0_2 = 100
        nu0_3 = 1e9
        if k==0:
            g = ((7/2) + (3/2) )/2
        else:
            g = ((3/2) + (1/2) )/2
        a1 = -(11*g+12)/(7*(2*g+1))
        b1 = -(3*(11*g+12))/(35*(2*g+1))
        b2 = -(3*(5*g+8))/(7*(2*g+1))
        b3 = -(3*(5*g+8))/(7*(2*g+1))
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
        return np.array(res)
    def forwardshock(ivar, f0,nu01,nu02):
        t0 = 1
        s = 10
        d = 0.2
        k=args.k
        t, nu = ivar
        res = []
        # frev = reverse_shock(ivar, frev, nu0rev,k)
        f = theory_bigsbpl(ivar, f0, nu01, nu02, k)
        return  f
    initial_guess = [1e-3,10,50]
    bounds = [(1e-6,1),(1,100),(15,1e5)]
    bounds0 = tuple([b[0] for b in bounds])
    bounds1 = tuple([b[1] for b in bounds])
    bounds = [bounds0,bounds1]
    curdata = plotdata[plotdata['band']!="X1"]
    tdata = curdata['obsdate']
    nudata = curdata['freq']
    xdata = (tdata,nudata)
    ydata = curdata['flux']*1e-6
    yerr = np.sqrt(curdata['err']**2 + curdata['rms']**2)*1e-6
    popt, pcov = curve_fit(forwardshock, xdata, ydata, p0=initial_guess,bounds=bounds,sigma=yerr)
    varnames = ["nua_0","num_0"]
    text = f"f0={popt[0]}+/-{np.absolute(pcov[0][0])**0.5}"
    print(text)
    for ind,var in enumerate(varnames):
        vnum = ind+1
        text = f" {var}={popt[vnum]}+/-{np.absolute(pcov[vnum][vnum])**0.5}"
        print(text)
    print("d=0.2")
    forwardpopt = popt
    forwardpcov = pcov
    forwardsigma = np.sqrt(np.diagonal(pcov))
   #  def wrap_thinbigsbpl(ivar, f0, frev, nu0rev,nu01,nu02):
   #      t0 = 1
   #      s = 10
   #      d = 0.2
   #      k=args.k
   #      t, nu = ivar
   #      res = []
   #      frev = nonrel_reverse_shock(ivar, frev, nu0rev,k)
   #      f = theory_bigsbpl(ivar, f0, nu01, nu02, k)
   #      return frev + f
   #  initial_guess = [1e-3,5e-5, 10,10,50]
   #  bounds = [(1e-6,1),(3e-5,2),(1,100),(1,100),(15,1e5)]
   #  bounds0 = tuple([b[0] for b in bounds])
   #  bounds1 = tuple([b[1] for b in bounds])
   #  bounds = [bounds0,bounds1]
   #  curdata = plotdata[plotdata['band']!="X1"]
   #  tdata = curdata['obsdate']
   #  nudata = curdata['freq']
   #  xdata = (tdata,nudata)
   #  ydata = curdata['flux']*1e-6
   #  yerr = np.sqrt(curdata['err']**2 + curdata['rms']**2)*1e-6
   #  popt, pcov = curve_fit(wrap_thinbigsbpl, xdata, ydata, p0=initial_guess,bounds=bounds,sigma=yerr)
   #  varnames = ["f0_rev","nua0_rev","nua_0","num_0"]
   #  text = f"f0={popt[0]}+/-{np.absolute(pcov[0][0])**0.5}"
   #  print(text)
   #  for ind,var in enumerate(varnames):
   #      vnum = ind+1
   #      text = f" {var}={popt[vnum]}+/-{np.absolute(pcov[vnum][vnum])**0.5}"
   #      print(text)
   #  print("d=0.2")
   #  thinpopt = popt

fig,axs = plt.subplots(3,2,figsize=(15,15),sharex=True,sharey=True)

for band,ax in zip(bands,axs.flatten()):
    curdata = plotdata[plotdata['band']==band]
    if band in ['X']:
        curdata = plotdata[(plotdata['band']=="X") | (plotdata['band']=="X1")]
    # print(curdata)
    xdata = curdata['obsdate']
    ydata = curdata['flux']*1e-6
    yerr = np.sqrt(curdata['err']**2 + curdata['rms']**2)*1e-6
    marker = itertools.cycle((',', '+', '.', 'o', '*'))
    linestyle = itertools.cycle(('-', ':', '-.', '--'))
    xline = np.geomspace(1e-2,365,num=1000)
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
           subcurdata = curdata[curdata['freq']==freq]
           subxdata = subcurdata['obsdate']
           subydata = subcurdata['flux']
           subyerr = np.sqrt(subcurdata['err']**2 + subcurdata['rms']**2)
           ax.errorbar(subxdata,subydata,yerr=subyerr,fmt=' ',color='black')
           ax.scatter(subxdata,subydata,label=f'{freq} GHz',marker=next(marker),color='black')
           if band=='C':
               subcheck = checkdata[checkdata['band']==band]
               checkerr = np.sqrt(subcheck['err']**2 + subcheck['rms']**2)
               ax.errorbar(subcheck['obsdate'],subcheck['flux'],yerr=checkerr,fmt=' ',color='black',marker='x', label='check source')
           nu = np.array([freq for f in xline])
           yline = wrap_bigsbpl((xline,nu), *bigpopt)*1e6
        
    nu = np.array([freq for f in xline])
    yline = wrap_bigsbpl((xline,nu), *bigpopt)*1e6
    ax.plot(xline,yline,alpha=0.5,color='black',ls=next(linestyle),label=f"{freq} GHz Forward+Reverse")
    upper_param = np.array([bigpopt[0]+bigsigma[0],bigpopt[1]+bigsigma[1],bigpopt[2]-bigsigma[2],bigpopt[3]+bigsigma[3],bigpopt[4]+bigsigma[4]])
    lower_param = np.array([bigpopt[0]-bigsigma[0],bigpopt[1]-bigsigma[1],bigpopt[2]+bigsigma[2],bigpopt[3]-bigsigma[3],bigpopt[4]-bigsigma[4]])
    bound_upper = wrap_bigsbpl((xline,nu),*upper_param)*1e6
    bound_lower = wrap_bigsbpl((xline,nu),*lower_param)*1e6
    ax.fill_between(xline,bound_lower,bound_upper,color='black',alpha=0.15)
    yline = forwardshock((xline,nu), *forwardpopt)*1e6
    ax.plot(xline,yline,alpha=0.5,color='black',ls=next(linestyle),label=f"{freq} GHz Forward")
    bound_upper = forwardshock((xline,nu),*(forwardpopt+forwardsigma))*1e6
    bound_lower = forwardshock((xline,nu),*(forwardpopt-forwardsigma))*1e6
    ax.fill_between(xline,bound_lower,bound_upper,color='black',alpha=0.15)
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
    test_theory = [1e-3, 18.5, 60.3 , 2]
    # print(test_theory)
    # ax.plot(xline,yline,color='black',alpha=0.5,ls=':',label='theory')
    # ax.plot(xline,yline,color='black',alpha=0.5,ls='-')


#     test_theory = [1.e-3, 58 , 95 , 0]
#     print(test_theory)
#     t_break = get_tbreak((xline,nu),*test_theory)
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
plt.savefig("tryfit.png")
plt.close()

fig = plt.figure()
freq = 9
curdata = plotdata[plotdata['band']=='X']
xdata = curdata['obsdate']
ydata = curdata['flux']*1e-6
yerr = np.sqrt(curdata['err']**2 + curdata['rms']**2)*1e-6
plt.scatter(xdata,ydata,label=f'{freq} GHz')
plt.errorbar(xdata,ydata,yerr=yerr,fmt=' ')
ax = plt.gca()
nu = np.array([freq for f in xline])
yline = wrap_bigsbpl((xline,nu), *bigpopt)
ax.plot(xline,yline,alpha=0.5,color='black',ls='-')
ax.set_title(f'{freq} GHz')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel("Flux Density (Jy)")
ax.set_ylim(1e-5,3e-3)
for o in np.sort(np.unique(curdata['obs'])):
    subdata = curdata[curdata['obs']==o]
    startobs = subdata['startdate'].min()
    endobs = subdata['stopdate'].max()
    plt.axvspan(startobs,endobs, alpha=0.15, color='gray')
ax.set_xlabel("Days post-trigger")
plt.tight_layout()
plt.savefig("9GHzlc.png")
plt.close()

fig = plt.figure()
freq=0.81
curdata = plotdata[plotdata['freq']==0.81]
print("UHF data:",curdata)
xdata = curdata['obsdate']
ydata = curdata['rms']*3
plt.scatter(xdata,ydata,label=f'{freq} GHz',marker='v')
ax = plt.gca()
obslimit =ydata.to_numpy()[0] 
ax.axhline(obslimit,ls=':')
ax.set_title(f'{freq} GHz')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel("RMS noise ($\mu Jy/BM$)")
ax.set_ylim(10,3e3)
ax.set_title("UHF Band 3-$\sigma$ Limits")
ax.set_xlim(1,1e4)
# ax.axvline((,ls=':',label="Today")
# for o in np.sort(np.unique(plotdata['obs'])):
#     subdata = plotdata[plotdata['obs']==o]
#     startobs = subdata['startdate'].min()
#     endobs = subdata['stopdate'].max()
#     plt.axvspan(startobs,endobs, alpha=0.15, color='gray')
xline = np.geomspace(1,1e4,num=100000)
nu = np.array([freq for f in xline])
yline = wrap_bigsbpl((xline,nu), *bigpopt)*1e6
limits = xline[np.round(yline,1)==np.round(obslimit,1)]
if limits.size >0:
    ax.axvline(limits[0],ls='--')
    ax.axvline(limits[-1],ls='--')
ax.plot(xline,yline,alpha=0.5,color='black',ls='-',label='Model')
ax.set_xlabel("Days post-trigger")
plt.legend()
plt.tight_layout()
plt.savefig("UHFpredictlc.png")
plt.close()

for freq,obslimit in [(0.8,10.5),(1.3,5),(11.85,3)]:
   #  curdata = plotdata[plotdata['freq']==0.81]
   #  print("UHF data:",curdata)
   #  xdata = curdata['obsdate']
   #  ydata = curdata['rms']*3
   #  plt.scatter(xdata,ydata,label=f'{freq} GHz',marker='v')
    ax = plt.gca()
    # obslimit =ydata.to_numpy()[0] 
    ax.set_title(f'{freq} GHz')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel("RMS noise ($\mu Jy/BM$)")
    # ax.set_ylim(10,3e3)
    xmin = 0.003
    xmax = 1e4
    ax.set_xlim(xmin,xmax)
    # ax.axvline((,ls=':',label="Today")
    # for o in np.sort(np.unique(plotdata['obs'])):
    #     subdata = plotdata[plotdata['obs']==o]
    #     startobs = subdata['startdate'].min()
    #     endobs = subdata['stopdate'].max()
    #     plt.axvspan(startobs,endobs, alpha=0.15, color='gray')
    xline = np.geomspace(xmin,xmax,num=100000)
    nu = np.array([freq for f in xline])
    yline = wrap_bigsbpl((xline,nu), *bigpopt)*1e6
    ax.axhline(obslimit,ls=':')
    print(freq,yline.max())
    limits = xline[np.round(yline,1)==np.round(obslimit,1)]
    ax.set_title(f"{freq} GHz Model")
    if limits.size>1:
        # ax.axvline(limits[0],ls='--')
        # ax.axvline(limits[-1],ls='--')
        ax.set_xlabel("Days post-trigger")
        ax.set_title(f"{freq} GHz Model\n{int(round(limits[-1]-limits[0],0))} days detectable")
    ax.plot(xline,yline,alpha=0.5,color='black',ls='-',label='Model')
    xdata = np.geomspace(plotdata['obsdate'].min(), 365, num=10)
    model = wrap_bigsbpl((xdata,nu),*bigpopt)*1e6
    ydata = 3*np.array([obslimit for x in xdata])
    print(ydata,model)
    plt.scatter(xdata[ydata>model],ydata[ydata>model],label=f'{freq} GHz',marker='^',color='black')
    plt.scatter(xdata[ydata<=model],ydata[ydata<=model],label=f'{freq} GHz',marker='v',color='red')
    ax.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{freq}GHzmodellc_meas.png")
    plt.close()
for freq,obslimit in [(0.074,46.5),(0.2,15),(0.8,7.2),(1.3,3.45)]:
   #  curdata = plotdata[plotdata['freq']==0.81]
   #  print("UHF data:",curdata)
   #  xdata = curdata['obsdate']
   #  ydata = curdata['rms']*3
   #  plt.scatter(xdata,ydata,label=f'{freq} GHz',marker='v')
    ax = plt.gca()
    # obslimit =ydata.to_numpy()[0] 
    ax.set_title(f'{freq} GHz')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel("RMS noise ($\mu Jy/BM$)")
    # ax.set_ylim(10,3e3)
    xmin = 0.003
    xmax = 1e4
    ax.set_xlim(xmin,xmax)
    # ax.axvline((,ls=':',label="Today")
    # for o in np.sort(np.unique(plotdata['obs'])):
    #     subdata = plotdata[plotdata['obs']==o]
    #     startobs = subdata['startdate'].min()
    #     endobs = subdata['stopdate'].max()
    #     plt.axvspan(startobs,endobs, alpha=0.15, color='gray')
    xline = np.geomspace(xmin,xmax,num=100000)
    nu = np.array([freq for f in xline])
    yline = wrap_bigsbpl((xline,nu), *bigpopt)*1e6
    ax.axhline(obslimit,ls=':')
    print(freq,yline.max())
    limits = xline[np.round(yline,1)==np.round(obslimit,1)]
    ax.set_title(f"{freq} GHz Model")
    if limits.size>1:
        # ax.axvline(limits[0],ls='--')
        # ax.axvline(limits[-1],ls='--')
        ax.set_xlabel("Days post-trigger")
        ax.set_title(f"{freq} GHz Model\n{int(round(limits[-1]-limits[0],0))} days detectable")
    ax.plot(xline,yline,alpha=0.5,color='black',ls='-',label='Model')
    ax.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{freq}GHzmodellc.png")
    plt.close()




tpk = [1.4,6.3,11.0,17,23,54,75,129,160]
nupk =[16.3,11.1,16.1,4.07,9.47,5.5,3.25,2.51,3.58]

# fit lcpk
fig = plt.figure()
plt.scatter(tpk,nupk)
initial_guess = [9.0/(9.25**(-3/2))*0.1,-3/2]
bounds = [(10,500),(-3,-0.5)]
bounds0 = tuple([b[0] for b in bounds])
bounds1 = tuple([b[1] for b in bounds])
bounds = [bounds0,bounds1]
popt, pcov = curve_fit(powerlaw, tpk, nupk, p0=initial_guess,bounds=bounds)
xline = np.linspace(1e-2,365,num=1000)
yline = powerlaw(xline, *popt)
plt.plot(xline,yline,color='black',alpha=0.5)
ax = plt.gca()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel("Obs. Date (days post burst)")
ax.set_ylabel("Obs. Freq. (GHz)")
title=f'nu_pk  alpha:{popt[1]}'
ax.set_title(title)
plt.savefig("nu_pk_time.png")
plt.close()
popt = bigpopt
chisq = 0
dof = len(plotdata) - len(popt)
for d,nu,f,ferr,rms in plotdata[['obsdate','freq','flux','err','rms']].to_numpy():
    errtot = np.sqrt(ferr**2 + rms**2)*1e-6
    model = yline = wrap_bigsbpl((np.array([d]),np.array([nu])), *popt)
    chisq += ((f*1e-6 - model) / errtot)**2 / dof
print("red. chisq:",chisq,"dof:",dof,"fitting indices")

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

plotdata = pd.read_csv("spectra_uvfit.csv")
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


