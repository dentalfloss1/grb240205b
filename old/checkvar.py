import matplotlib.pyplot as plt
import pandas as pd 
import datetime
import numpy as np
import itertools
from astropy.modeling.powerlaws import SmoothlyBrokenPowerLaw1D as sbpl
from scipy.optimize import curve_fit
from scipy.stats import linregress, chi2
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
for name,data in [("GRB",plotdata),("Check",checkdata)]:
    print("-------------",name,"--------------")
    for band in ["L","S","C","X","Ku"]:
        subdata = data[(data['obsdate'] > 1) & (data['band']==band)]
        ydata = data['flux']
        yerr = data['err']
         
        avgflux = np.average(ydata)
        chisq = 0
        dof = len(ydata) - 1
        eta = 0 
        avgwtflux = np.average(ydata,weights = 1/yerr)
        for t1,t2,freq,f,fe in zip(subdata["start"],subdata["stop"],subdata['freq'],ydata,yerr):
            chisq += ((f-avgflux)/fe)**2
            eta += (1/(len(ydata)-1)) * ((f-avgwtflux)/fe)**2
            # print(f*1e6,fe*1e6,band,sep=',')
        
        P = 1- chi2.cdf(chisq, dof)
        
        print(band,"Band","$\chi^2$ = ",chisq,"P = ",P,"eta = ",eta)
        
