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

Currently, there is no function to export parameters but this can be easily added in future releases. 
