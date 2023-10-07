# PID-controller-selection
This was work performed during the lab which is the completion of the energy analysis of the district heating rig
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 12:51:40 2023

@author: aayamc
"""

import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
from matplotlib import pyplot

control = pd.read_csv('control.csv',encoding='latin1')

control.info()
control.head()

a = control.iloc[100:500]

x= a['SN']
Sensor_temp = a['T80']
rad_h_temp = a['T40']
Valve_position = a['Y223']


plt.figure().set_figwidth(15)
plt.plot(control['SN'],control['T80'], 'g')
plt.grid()
plt.show()

plt.figure().set_figwidth(15)
plt.plot(control['SN'],control['T40'], 'b--')
plt.grid()
plt.show()



plt.figure(figsize=(15, 10))
plt.subplot(211)

plt.plot(x,Valve_position, 'g')
plt.legend(['Valve position by percentage'])
plt.xlabel('Time (secs)')
plt.ylabel('Valve position %')
plt.grid()
         
plt.subplot(212)

def objective(x1, a1, b1, c1, d1, e1, f1):
 return (a1 * x1) + (b1 * x1**2) + (c1 * x1**3) + (d1 * x1**4) + (e1 * x1**5) + f1

a11 = control.iloc[100:450]
x1,y1 = a11['SN'],a11['T80']

popt, _ = curve_fit(objective, x1, y1)

a1, b1, c1, d1, e1, f1 = popt

plt.scatter(x1, y1,s=20)
x_line = np.arange(min(x1), max(x1), 1)
y_line = objective(x_line, a1, b1, c1, d1, e1, f1)

#plt.plot(x_line, y_line, '--', color='red')
plt.scatter(x= 165, y= 26,color = 'b')
plt.scatter(x= 156, y= 25,color = 'g')
plt.scatter(x= 188, y= 25,color = 'g')
plt.plot(x,Sensor_temp, 'r')
x_=[156,165,188]
y_=[25,26,28.4]
x__ =[188,188]
y__ =[25,28.4]
x___ =[132,132]
y___ =[25,28.4]
plt.plot(x_,y_,'b')
plt.plot(x__,y__,'b')
plt.plot(x___,y___,'g--')
plt.scatter(x= 132, y= 25,color = 'g')
plt.axhline(y = 28.4, color = 'black', linestyle = '-') 

plt.xlabel('Time (secs)')
plt.ylabel('Temperature (°C)')
plt.legend(['temp data for prediction','predicted data','inflection point','actual temp data'])

plt.annotate('%.1f sec'%(156),xy=(156, 25))   
plt.annotate('%.1f sec'%(188),xy=(195, 25)) 
plt.annotate('%.1f sec'%(132),xy=(115, 25))            

plt.grid()
plt.show()


#selecting controller: 

Td = 156 - 132
print(Td)
Tr = 188 - 156
print(Tr)
tempchange =  28.4 - 25
print(tempchange)
valvechange = 60 - 20
print(valvechange)

K_s = tempchange / valvechange 
print(K_s)
D_d = Td / Tr 
print(D_d)

D = K_s * D_d * (1/2) * 100 
print(D)
if (0.0 < D < 1.0):
    print('Proportional controller')

elif (1.0 < D < 2.5):
    print('Proportional integral controller')

elif (2.5 < D < 5):
    print('Proportional integral derivative controller is selection for which')
    Xp = 0.9 * K_s * D_d * 100
    print('Xp- proportional band is {}'.format(Xp))
    Kp = 1/Xp
    print('Kp- proportional gain is {}'.format(Kp))
    Ti = 2.5* Td
    print('Ti- integral time reset time is {}'.format(Ti))
    TD = 0.5 * Td
    print('TD- derivation time is {}'.format(TD))
else:
    print('Cascade')




controller = control['Y223']
serialnumber =control['SN']
Power =control['EO8']

fig, ax = plt.subplots(figsize = (10, 5))
plt.title('Time series plot of temperature and power')
 

ax2 = ax.twinx()
ax.plot(serialnumber, controller, color = 'g')
ax2.plot(serialnumber, Power, color = 'b')
 

ax.set_xlabel('Time (secs)', color = 'r')

ax.set_ylabel('valve change %', color = 'g')
 
ax2.set_ylabel('Power (KW)', color = 'b')
 
 
plt.tight_layout()
 

plt.grid()
plt.show()



