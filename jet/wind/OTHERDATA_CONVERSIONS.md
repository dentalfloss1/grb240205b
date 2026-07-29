# Otherdata Conversion Notes

This file documents the final GCN-by-GCN decisions for C:\Users\sarah\Modeling\grb240205b\grbfit_modeling\otherdata.txt.

Output format for GRBafterglowfit: obsdate,freq,flux,err,rms
- obsdate is days after the confirmed trigger time: 2024-02-05 22:13:08 UT.
- freq is GHz.
- flux and err are microJy.
- rms is 0 for all optical/NIR photometry here because the GCNs provide magnitude errors, not separate image rms values.
- No Galactic extinction correction is applied to any point.

## Shared formulas

- AB magnitudes: F_microJy = 10^((23.9 - m_AB) / 2.5).
- Vega magnitudes: F_microJy = F0_Jy * 10^(-0.4 * m_Vega) * 1e6.
- Magnitude error propagation: sigma_F = F * ln(10) / 2.5 * sigma_mag.
- Frequency conversion: nu_GHz = c / lambda / 1e9, with c = 299792458 m/s.

## GCN 35686: BOOTES-7

Decision: include. The GCN gives g = 15.5 +/- 0.1 on a 60 s image taken at 2024-02-06 00:33 UT. We assume lowercase g is Sloan-like AB g and use lambda_eff = 477 nm. We interpret the stated time as the exposure start and use midpoint = 2024-02-06 00:33:30 UT.

Conversions:
- obsdate = (2024-02-06 00:33:30 - trigger) / day = 0.097476851852.
- freq = c / 477 nm = 628495.719078 GHz.
- flux = 10^((23.9 - 15.5) / 2.5) = 2290.86765277 microJy.
- err = flux * ln(10) / 2.5 * 0.1 = 211.001449381 microJy.

Row:
0.097476851852,628495.719078,2290.86765277,211.001449381,0

## GCN 35696: BOOTES-6

Decision: include. The GCN gives g = 18.1 +/- 0.1 on a 300 s image taken at 2024-02-06 18:30 UT. We use the same BOOTES convention: Sloan-like AB g with lambda_eff = 477 nm. We interpret the stated time as the exposure start and use midpoint = 2024-02-06 18:32:30 UT.

Conversions:
- obsdate = (2024-02-06 18:32:30 - trigger) / day = 0.846782407407.
- freq = c / 477 nm = 628495.719078 GHz.
- flux = 10^((23.9 - 18.1) / 2.5) = 208.929613085 microJy.
- err = flux * ln(10) / 2.5 * 0.1 = 19.244053094 microJy.

Row:
0.846782407407,628495.719078,208.929613085,19.244053094,0

## GCN 35694: REM

Decision: include both reported points. The r point is explicitly AB and calibrated against SkyMapper. The H point is explicitly Vega and calibrated against 2MASS.

REM r:
- GCN value: r = 15.67 +/- 0.01 AB at t - t0 = 2.63 hr.
- Adopted lambda_eff = 617.6 nm for SkyMapper/Sloan-like r.
- obsdate = 2.63 / 24 = 0.109583333333 days.
- freq = c / 617.6 nm = 485415.249352 GHz.
- flux = 10^((23.9 - 15.67) / 2.5) = 1958.84467351 microJy.
- err = flux * ln(10) / 2.5 * 0.01 = 18.0416261788 microJy.

Row:
0.109583333333,485415.249352,1958.84467351,18.0416261788,0

REM H:
- GCN value: H = 13.92 +/- 0.04 Vega at t - t0 = 2.65 hr.
- Adopted 2MASS H F0 = 1024 Jy and lambda_eff = 1.662 micron.
- obsdate = 2.65 / 24 = 0.110416666667 days.
- freq = c / 1.662 micron = 180380.540313 GHz.
- flux = 1024 Jy * 10^(-0.4 * 13.92) * 1e6 = 2768.85336485 microJy.
- err = flux * ln(10) / 2.5 * 0.04 = 102.008327721 microJy.

Row:
0.110416666667,180380.540313,2768.85336485,102.008327721,0

## GCN 35687: Skynet

Decision: include all 11 rows. The GCN reports uppercase B,V,R,I calibrated with APASS. We treat these as Johnson-Cousins-like Vega magnitudes and use the listed MJD values as exposure midpoints. This is an explicit modeling assumption because APASS itself is Landolt B,V plus Sloan g',r',i', while the GCN reports B,V,R,I.

Adopted effective wavelengths and Vega zero points:
- B: lambda_eff = 445 nm, F0 = 4260 Jy.
- V: lambda_eff = 551 nm, F0 = 3640 Jy.
- R: lambda_eff = 658 nm, F0 = 3080 Jy.
- I: lambda_eff = 806 nm, F0 = 2550 Jy.

Trigger MJD = 60345.925787. For every row, obsdate = MJD - 60345.925787.

Rows:
0.104212962964,673690.916854,1518.4818284,23.7757486299,0
0.105212962968,673690.916854,1518.4818284,23.7757486299,0
0.106212962964,544087.945554,1841.20176969,15.2622854931,0
0.108212962965,455611.638298,1961.3302044,18.0645187643,0
0.109212962969,544087.945554,1648.54719085,13.6653126723,0
0.122212962968,455611.638298,1677.06817075,16.9910055476,0
0.123212962964,673690.916854,1274.7047352,14.0885573819,0
0.123212962964,371950.940447,1881.65578683,32.9283114932,0
0.126212962969,544087.945554,1531.44492732,11.2841032332,0
0.139212962968,455611.638298,1529.50435007,14.0872556645,0
0.139212962968,544087.945554,1309.48757997,9.6486610595,0

## GCN 35684: MASTER

Decision: exclude. The GCN reports mOT ~ 12.8 near maximum and an upper limit up to 15.9 mag, but it does not provide a usable exact measurement time, measurement uncertainty for the approximate detection, or a clear sigma convention for the upper limit.

## GCN 35708: PRIME

Decision: include with approximate timing. The GCN gives J = 19.65 +/- 0.15 AB, total exposure time 800 s, and says the observation occurred approximately 2 days after trigger. We adopt obsdate = 2.0 days exactly and use standard J lambda_eff = 1.25 micron.

Conversions:
- obsdate = 2.0 days.
- freq = c / 1.25 micron = 239833.9664 GHz.
- flux = 10^((23.9 - 19.65) / 2.5) = 50.1187233627 microJy.
- err = flux * ln(10) / 2.5 * 0.15 = 6.9235247366 microJy.

Row:
2,239833.9664,50.1187233627,6.9235247366,0
