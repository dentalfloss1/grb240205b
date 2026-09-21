# GRB 240205B GRBafterglowfit Input Notes

Source table: C:\Users\sarah\Modeling\grb240205b\grbmeas.csv
Generated folder: C:\Users\sarah\Modeling\grb240205b\grbfit_modeling

Trigger time: 2024-02-05 22:13:08 UT
Fermi/GBM trigger: 728863993.169242 / 240205926

Time convention:
- obsdate is the arithmetic midpoint of source start and stop times.
- obsdate is expressed in days after the confirmed trigger time.

Column conversion:
- freq, flux, err, and rms are in the same units as the source table.
- The output table contains the GRBafterglowfit columns obsdate,freq,flux,err,rms.

Zero-rms handling:
- If the source rms was nonzero, source err and rms were preserved.
- If the source rms was 0, the source err value was moved into rms.
- For positive-flux zero-rms rows, err was set to 1e-30 so GRBafterglowfit keeps the row as a detection while the effective uncertainty remains essentially rms.
- For negative-flux zero-rms rows, flux was preserved and err was set to 0; GRBafterglowfit treats these as upper-limit rows and plots them at 3*rms.
- Existing sentinel limits such as flux=-1, err=-1, rms>0 were preserved.
