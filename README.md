# Curve-Fitter

This Curve Fitter GUI is a Python based interface designed to allow users to dynamically adjust a curve to I-V data measurements. 

## About

You can upload your current - voltage data and adjust the following parameters according to Shockley's diode equation.

$$ I = I_{sc} - I_{o}(e^{q(V+IR_s)/nkT} - 1) - \frac{V + IR_s}{R_{sh}} $$

I_sc: Short-circuit curent (mA)  
I_o: Reverse bias saturation current (mA)  
n: Ideality factor  
T: Temperature (K)  
R_s: Series resistance ($\Omega$)  
R_sh: Shunt resistance ($\Omega$)  

After uploading data, click on "Plot Theoretical Curve" to plot the initial curve, and adjusting any parameters thereafter will automatically update the plot. To switch to a log plot, click "Plot Log(Current)", and adjusting any parameters thereafter will automatically update the plot while still in the log view. Click on "Plot Theoretical Curve" to exit out of log view. 

Currently, there is no function to export parameters but this can be easily added in future releases. 
