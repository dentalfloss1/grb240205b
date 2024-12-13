import matplotlib.pyplot as plt
import pandas as pd 
import datetime
import numpy as np
import itertools
from astropy.modeling.powerlaws import SmoothlyBrokenPowerLaw1D as sbpl
from scipy.optimize import curve_fit
from scipy.stats import linregress
import argparse
trigger = datetime.datetime(2024, 2, 5, 22, 15, 8, 00)

# plotdata = pd.read_csv("grbmeas_30min.csv")
plotdata = pd.read_csv("grbmeas_45min.csv")
startdate = [(datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S.%f") - trigger).total_seconds()/3600/24 for d in plotdata['start']]
stopdate = [(datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S.%f") - trigger).total_seconds()/3600/24 for d in plotdata['stop']]
obsdur = [(t2-t1) for t1,t2 in zip(startdate,stopdate)]
plotdata['obsdate'] = [t1+(dur/2) for t1,dur in zip(startdate,obsdur)]
plotdata['startdate'] = startdate
plotdata['stopdate'] = stopdate
# plotdata = plotdata[np.isin(plotdata['freq'],[5.5,9.0])]
parser = argparse.ArgumentParser()
parser.add_argument("--freezeParams",action="store_true")
args = parser.parse_args()
freezeParams=args.freezeParams
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
        s = 5
        a1 = -k/(2*(4-k))
        b1 = -3*k/(5*(4-k))
        b2 = -3/2
        p = 2.2
        nu0_3 = 1e9
        t, nu = ivar
        y1 = []
        y2 = []
        t_break = np.amax(t)
        for tval in np.sort(t):
            nua = nu0_1*(tval/t0)**b1
            num = nu0_2*(tval/t0)**b2
            if num <= nua:
                t_break = tval
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
            num_2 = nu0_1*(tval/t0)**b1_2
            nua_2 = nu0_2*(tval/t0)**b2_2
            num_break2 = nu0_1*(t_break/t0)**b1_2
            nua_break2 = nu0_2*(t_break/t0)**b2_2
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
            y1.append(res1)
            y2.append(res2)

        result = np.where( t<=t_break,np.array(y1),np.array(y2))
        return result

    initial_guess = [1e-3,9,20,2]
    bounds = [(1e-6,5e-3),(1,100),(1,100),(0,2.1)]
    bounds0 = tuple([b[0] for b in bounds])
    bounds1 = tuple([b[1] for b in bounds])
    curdata = plotdata[ (plotdata['obsdate'] >10)]
    bounds = [bounds0,bounds1]
    tdata = curdata['obsdate']
    nudata = curdata['freq']
    xdata = (tdata,nudata)
    ydata = curdata['flux']*1e-6
    yerr = np.sqrt(curdata['err']**2 + curdata['rms']**2)*1e-6
  #   theorypopt, pcov = curve_fit(theory_bigsbpl, xdata, ydata, p0=initial_guess,bounds=bounds,sigma=yerr)
  #   popt = theorypopt
  #   varnames = ["nu0_1","nu_02","k"]
  #   text = f"f0={popt[0]}+/-{np.absolute(pcov[0][0])**0.5}"
  #   print(text)
  #   for ind,var in enumerate(varnames):
  #       vnum = ind+1
  #       text = f" {var}={popt[vnum]}+/-{np.absolute(pcov[vnum][vnum])**0.5}"
  #       print(text)
  #   print("s=10")
#     k = 2
#     p = 2
#     a1 = -k/(2*(4-k))
#     b1 = -(12*p+8-3*p*k+2*k)/(2*(4-k)*(p+4))
#     c1 = (20-3*k)/(4*(4-k))
#     c2 = -(12*p-12-3*p*k+5*k)/(4*(4-k))
#     print(a1,b1,c1,c2)
#     theorypopt = [0.000920, 10,20,0]
    def powerlaw(t,a,k):
        return a*t**k

    def reverse_shock(ivar, f0, t0,k):
        t, nu = ivar
        s=10
        res = []
        p=2
        nu0_1 = 1
        nu0_2 = 15
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
    def wrap_bigsbpl(ivar, f0, frev, trev,nu0, a1, b1, c1, c2):
        t0 = 1
        s = 10
        d = 0.2
        k=2
        t_nonrel=22
        t, nu = ivar
        res = []
        frev = reverse_shock(ivar, frev, trev,k)
        for tval,nuval in zip(t,nu):
            fpk = f0*(tval/t0)**a1
            nupk = nu0*(tval/t0)**b1
         #    nupk2 = nu0*(tval/t0)**b2
            f = sbpl(amplitude=fpk, x_break=nupk, alpha_1=-c1, alpha_2=-c2, delta=d)
            # f = sbpl(amplitude=fpk, x_break=nupk, alpha_1=-c1, alpha_2=-c2, delta=d)
            # res1 = dsbpl(nuval,fpk_1,nubreak1_1,c1_1,c2_1,nubreak2_1,c3_1,s)
            res.append(f(nuval))
        return frev + np.array(res)
    initial_guess = [1e-3,5e-5, 3.5,30, -1, -1, 2, 1/3]
    bounds = [(1e-6,1),(3e-5,2),(3,10),(21,100),(-4,-0.1),(-4,-0.1),(0.1,3),(-3,3)]
    bounds0 = tuple([b[0] for b in bounds])
    bounds1 = tuple([b[1] for b in bounds])
    bounds = [bounds0,bounds1]
    curdata = plotdata
    tdata = curdata['obsdate']
    nudata = curdata['freq']
    xdata = (tdata,nudata)
    ydata = curdata['flux']*1e-6
    yerr = np.sqrt(curdata['err']**2 + curdata['rms']**2)*1e-6
    popt, pcov = curve_fit(wrap_bigsbpl, xdata, ydata, p0=initial_guess,bounds=bounds,sigma=yerr)
    varnames = ["frev","trev","nu0","a1","b1","c1","c2"]
    text = f"f0={popt[0]}+/-{np.absolute(pcov[0][0])**0.5}"
    print(text)
    for ind,var in enumerate(varnames):
        vnum = ind+1
        text = f" {var}={popt[vnum]}+/-{np.absolute(pcov[vnum][vnum])**0.5}"
        print(text)
    print("d=0.2")
bigpopt = popt
trypopt = bigpopt.copy()
trypopt[-1] = 2.2
def wrap_latebigsbpl(ivar, f0, nu0, a1, b1, c1, c2):
    d = 0.2
    t0 = 20.0
    a1 = -1/3
    b1 = -7/3
    t, nu = ivar
    res = []
    for tval,nuval in zip(t,nu):
        fpk = f0*(tval/t0)**a1
        nupk = nu0*(tval/t0)**b1
        f = sbpl(amplitude=fpk, x_break=nupk, alpha_1=-c1, alpha_2=-c2, delta=d)
        res.append(f(nuval))
    return np.array(res)
latedata = plotdata[(plotdata['obsdate'] > 20)]
initial_guess = [5e-4,10, -1/3,-7/3,1,-2]
bounds = [(1e-6,1e-3),(1,100),(-0.5,0.5),(-3,3),(0.1,4),(-4,-0.1)]
bounds0 = tuple([b[0] for b in bounds])
bounds1 = tuple([b[1] for b in bounds])
bounds = [bounds0,bounds1]
tdata = latedata['obsdate']
nudata = latedata['freq']
xdata = (tdata,nudata)
ydata = latedata['flux']*1e-6
yerr = latedata['err']*1e-6
# popt, pcov = curve_fit(wrap_latebigsbpl, xdata, ydata, p0=initial_guess,bounds=bounds,sigma=yerr)
# varnames = ["nu0","a1","b1","c1","c2"]
# text = f"f0={popt[0]}+/-{np.absolute(pcov[0][0])**0.5}"
# print(text)
# for ind,var in enumerate(varnames):
#     vnum = ind+1
#     text = f" {var}={popt[vnum]}+/-{np.absolute(pcov[vnum][vnum])**0.5}"
#     print(text)
# print("d=0.2")
# # latebigpopt = [8e-5,10,1,-1,2,-2]
# latebigpopt = popt
def wrap_earlybigsbpl(t, f0, t0):
    c1 = 0.5
    c2 = -0.5
    res = []
    return wrap_sbpl(t, f0, t0,-1/2, 1/2, 0.1)
earlydata = plotdata[(plotdata['obsdate'] < 1) & (plotdata['freq'] > 8)]
print(earlydata)
initial_guess = [250e-6,0.035]
bounds = [(1e-4,5e-4),(0.025,0.05)]
bounds0 = tuple([b[0] for b in bounds])
bounds1 = tuple([b[1] for b in bounds])
bounds = [bounds0,bounds1]
tdata = earlydata['obsdate']
nudata = earlydata['freq']
xdata = tdata
ydata = earlydata['flux']*1e-6
yerr = earlydata['err']*1e-6
wrap_sbpl(x2, 250, 0.035,-1/2, 1/2, 0.1)
popt, pcov = curve_fit(wrap_earlybigsbpl, xdata, ydata, p0=initial_guess,bounds=bounds,sigma=yerr)
varnames = ["t0"]
text = f"f0={popt[0]}+/-{np.absolute(pcov[0][0])**0.5}"
print(text)
for ind,var in enumerate(varnames):
    vnum = ind+1
    text = f" {var}={popt[vnum]}+/-{np.absolute(pcov[vnum][vnum])**0.5}"
    print(text)
print("d=0.2")
# earlybigpopt = [8e-5,10,1,-1,2,-2]
earlybigpopt = popt
def wrap_relbigsbpl(ivar, f0,nu0, a1,b1,c1,c2):
    d = 0.2
    t0 = 1
    b1 = -3/2
    t, nu = ivar
    res = []
    for tval,nuval in zip(t,nu):
        fpk = f0*(tval/t0)**a1
        nupk = nu0*(tval/t0)**b1
        f = sbpl(amplitude=fpk, x_break=nupk, alpha_1=-c1, alpha_2=-c2, delta=d)
        res.append(f(nuval))
    return np.array(res)
curdata = plotdata[(plotdata['obsdate'] >1)]
initial_guess = [5e-4,30,-0.25 ,-3/2, 1,-1]
bounds = [(1e-6,1e-2),(21,100),(-2,0),(-4,-3/2 + 0.2),(0,2),(-3,0)]
bounds0 = tuple([b[0] for b in bounds])
bounds1 = tuple([b[1] for b in bounds])
bounds = [bounds0,bounds1]
tdata = curdata['obsdate']
nudata = curdata['freq']
xdata = (tdata,nudata)
ydata = curdata['flux']*1e-6
yerr = curdata['err']*1e-6
popt, pcov = curve_fit(wrap_relbigsbpl, xdata, ydata, p0=initial_guess,bounds=bounds,sigma=yerr)
varnames = ["nu0","a1","b1","c1","c2"]
text = f"f0={popt[0]}+/-{np.absolute(pcov[0][0])**0.5}"
print(text)
for ind,var in enumerate(varnames):
    vnum = ind+1
    text = f" {var}={popt[vnum]}+/-{np.absolute(pcov[vnum][vnum])**0.5}"
    print(text)
print("d=0.2")
# latebigpopt = [8e-5,10,1,-1,2,-2]
relbigpopt = popt


# def theory_bigsbpl(ivar, f0, nu0_1, nu0_2, k):
#     t0 = 1
#     d=0.4
#     s = 10
#     a1 = -k/(2*(4-k))
#     b1 = -3*k/(5*(4-k))
#     b2 = -3/2
#     t, nu = ivar
#     res = []
#     t_break = np.amax(t)
#     for tval in np.sort(t):
#         nua = nu0_1*(tval/t0)**b1
#         num = nu0_2*(tval/t0)**b2
#         if num <= nua:
#             t_break = tval
#             break
# 
#     for tval,nuval in zip(t,nu):
#         if tval > t_break:
#             b1 = -3*k/(5*(4-k))
#             nua = nu0_1*(tval/t0)**b1
#             num = nu0_2*(tval/t0)**b2
#             c1 = 2
#             c2 = 1/3
#             c3 = -0.6
#             fnu_m = f0*(tval/t0)**a1
#             fpk = fnu_m*(nua/num)**(1/3)
#         else:
#             p=2.2
#             a1 = -k/(2*(4-k))
#             b1 = -(12*p+8-3*p*k+2*k)/(2*(4-k)*(p+4))
#             nua = nu0_1*(tval/t0)**b1
#             num = nu0_2*(tval/t0)**b2
#             c1 = 2
#             c2 = 5/2
#             c3 = -0.6
#             fnu_m = f0*(tval/t0)**a1
#             fpk = fnu_m
#         res.append(dsbpl(nuval,fpk,nua,c1,c2,num,c3,s))
#     return np.array(res)

fig,axs = plt.subplots(6,1,figsize=(10,25),sharex=True,sharey=True)

for band,ax in zip(bands,axs):
    curdata = plotdata[plotdata['band']==band]
    # print(curdata)
    xdata = curdata['obsdate']
    ydata = curdata['flux']*1e-6
    yerr = np.sqrt(curdata['err']**2 + curdata['rms']**2)*1e-6
    marker = itertools.cycle((',', '+', '.', 'o', '*'))
    xline = np.geomspace(1e-2,365,num=1000)
    for freq in np.sort(np.unique(curdata['freq']))[::-1]:
       if (band=="X"):
           subcurdata = curdata[(curdata['freq']==freq) & (curdata['obsdate'] < 2.1)]
           subxdata = subcurdata['obsdate']
           subydata = subcurdata['flux']*1e-6
           subyerr = np.sqrt(subcurdata['err']**2 + subcurdata['rms']**2)*1e-6
           ax.errorbar(subxdata,subydata,yerr=subyerr,fmt=' ',color='black')
           ax.scatter(subxdata,subydata,label=f'{freq} GHz uvfit fixed position',facecolors='none',edgecolor='black')
           subcurdata = curdata[(curdata['freq']==freq) & (curdata['obsdate'] >= 2.1)]
           subxdata = subcurdata['obsdate']
           subydata = subcurdata['flux']*1e-6
           subyerr = np.sqrt(subcurdata['err']**2 + subcurdata['rms']**2)*1e-6
           ax.errorbar(subxdata,subydata,yerr=subyerr,fmt=' ',color='black')
           ax.scatter(subxdata,subydata,label=f'{freq} GHz',color='black')
       else:
           subcurdata = curdata[curdata['freq']==freq]
           subxdata = subcurdata['obsdate']
           subydata = subcurdata['flux']*1e-6
           subyerr = np.sqrt(subcurdata['err']**2 + subcurdata['rms']**2)*1e-6
           ax.errorbar(subxdata,subydata,yerr=subyerr,fmt=' ',color='black')
           ax.scatter(subxdata,subydata,label=f'{freq} GHz',marker=next(marker),color='black')
        
    nu = np.array([freq for f in xline])
    yline = wrap_relbigsbpl((xline,nu), *relbigpopt)
   #  ax.plot(xline,yline,alpha=0.5,color='black',label="rel",ls="-")
    yline = wrap_bigsbpl((xline,nu), *bigpopt)
    ax.plot(xline,yline,alpha=0.5,color='black')
    yline = wrap_bigsbpl((xline,nu), *trypopt)
    # ax.plot(xline,yline,alpha=0.5,color='black',label=f'k={trypopt[-1]}',ls=':')
   #      
   #  
   #  popt, pcov = curve_fit(wrap_sbpl, xdata, ydata, p0=initial_guess,bounds=bounds)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel("Flux Density (Jy)")
    ax.set_xlim(1e-2,365)
    ax.set_ylim(1e-5,3e-3)
    test_theory = [1e-3, 18.5, 60.3 , 2]
    t_break = get_tbreak((xline,nu),*test_theory)
    # print(test_theory)
    ax.axvline(t_break,ls='-')
    yline = theory_bigsbpl((xline,nu), *test_theory)
    # ax.plot(xline,yline,color='black',alpha=0.5,ls=':',label='theory')
    yline = verify_powerlaw(xline,1e-3,2,0.5,0.5,2,-0.75)
    # ax.plot(xline,yline,color='black',alpha=0.5,ls='-')


#     test_theory = [1.e-3, 58 , 95 , 0]
#     print(test_theory)
#     t_break = get_tbreak((xline,nu),*test_theory)
#     ax.axvline(t_break,ls=':')
#     yline = theory_bigsbpl((xline,nu), *test_theory)
#     ax.plot(xline,yline,color='black',alpha=0.5,ls=':',label='ISM')
    # yline = wrap_latebigsbpl((xline,nu), *latebigpopt)
#     ax.plot(xline,yline,color='black',ls='--',alpha=0.5,label='nonrel')
    if freq==9.0:
        yline = wrap_earlybigsbpl(xline, *earlybigpopt)
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
        # ax.axvspan(startobs,endobs, alpha=0.15, color='gray')
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
for o in np.sort(np.unique(curdata['obs'])):
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
# for d,nu,f,ferr,rms in plotdata[['obsdate','freq','flux','err','rms']].to_numpy():
#     errtot = np.sqrt(ferr**2 + rms**2)*1e-6
#     model = yline = wrap_bigsbpl((np.array([d]),np.array([nu])), *popt)
#     chisq += ((f*1e-6 - model) / errtot)**2 / dof
# print("red. chisq:",chisq,"dof:",dof,"fitting indices")
# popt = theorypopt
# chisq = 0
# dof = len(plotdata) - len(popt)
# for d,nu,f,ferr,rms in plotdata[['obsdate','freq','flux','err','rms']].to_numpy():
#     errtot = np.sqrt(ferr**2 + rms**2)*1e-6
#     model = yline = theory_bigsbpl((np.array([d]),np.array([nu])), *popt)
#     chisq += ((f*1e-6 - model) / errtot)**2 / dof
# print("red. chisq:",chisq,"dof:",dof,"fitting theory params")

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


