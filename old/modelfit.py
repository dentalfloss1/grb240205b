import numpy as np
import pandas as pd 
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

def dsbpl(x,amp,xb1,a1,a2,xb2,a3,s):
    if (a3 < a1) & (a1 < a2):
        result =  amp*(((x/xb1)**(a1*s) + (x/xb1)**(a2*s))**(-1) + (xb2/xb1)**(-a2*s)*(x/xb2)**(-a3*s))**(-1/s)
    elif (a3 < a2) & (a2 < a1):
        result = amp*((x/xb1)**(-a1*s) + (x/xb1)**(-a2*s) + ((xb1/xb2)**(a2*s))*((x/xb2)**(-a3*s)) )**(-1/s)
    else:
        raise Exception("Unhandled powerlaw index ordering")
    return result

def theory_bigsbpl(ivar, f0, nu0_1, nu0_2, k, t0=1, d=0.4, s=10, p=2):
    nu0_3 = 1e9
    a1 = -k/(2*(4-k))
    b1 = -3*k/(5*(4-k))
    b2 = -3/2
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
def reverse_shock(ivar, f0, nu0_1, k, s=10, p=2, t0=0.05, nu0_2=100, nu0_3=1e9):
    t, nu = ivar
    res = []
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
def wrap_bigsbpl(ivar, f0, frev, nu0rev, nu01, nu02, t0=1, s=10, d=0.2, k=2):
    t, nu = ivar
    res = []
    frev = reverse_shock(ivar, frev, nu0rev,k)
    f = theory_bigsbpl(ivar, f0, nu01, nu02, k)
    return frev + f

# f0,  f0 rev, nu0_a rev, nu0_a for., nu0_m for.
initial_guess = [1e-3,5e-5, 10,10,50]
bounds = [(1e-6,1),(3e-5,2),(1,100),(1,100),(15,1e5)]
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
