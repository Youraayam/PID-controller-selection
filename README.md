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
from scipy import integrate

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
Temp = control['T80']

fig, ax = plt.subplots(figsize = (10, 5))
plt.title('Time series plot of valve opening,power and temperature')
 
twin1 = ax.twinx()
twin2 = ax.twinx()

twin2.spines.right.set_position(("axes", 1.2))


p1= ax.plot(serialnumber, controller, color = 'g')
p2= twin1.plot(serialnumber, Power, color = 'b')
p3= twin2.plot(serialnumber, Temp, color = 'r') 




ax.set_xlabel('Time (secs)', color = 'r')

ax.set_ylabel('valve change %', color = 'g')
twin1.set_ylabel('Power (KW)', color = 'b')
twin2.set_ylabel('Temperature (°C)', color = 'r')
 




 
plt.tight_layout()
 

plt.grid()
plt.show()




#Controller action 
print('Proportional integral derivative controller is selection for which')
Xp = 0.9 * K_s * D_d * 100
print('Xp- proportional band is {}'.format(Xp))
Kp = 1/Xp
print('Kp- proportional gain is {}'.format(Kp))
Ti = 2.5* Td
print('Ti- integral time reset time is {}'.format(Ti))
TD = 0.5 * Td
print('TD- derivation time is {}'.format(TD))


a = control_df = pd.read_csv('control.csv',encoding='latin1')
time = 1 

integral = 0 

time_prev = 0

e_prev = 0 

N = 5
h = np.zeros(N)
å = np.zeros(N) 
æ = np.zeros(N)




wu = []
MVu = []

P_ = []
I_ = []
D_ = []

for i in range (400, 1050, 1):
        Kp = np.Kp = 0.17429193899782142
        Ti = 60.0
        TD = 12.0
        Ki = Kp/Ti
        Kd = Kp* TD
        e = 28 - a.loc[i,'T80']
        print(e)
        P = Kp*e 
        print(P)
        integral_n = integral + Ki * e 
        print(integral_n)
        D = Kd * (e - e_prev) 
        print(D)
        MV = P + integral + D 
        #print(MV)
        #print('ok')
        e_prev = e 
        integral = integral_n
        MVu.append(MV)
        P_.append(P)
        I_.append(integral)
        D_.append(D)


plt.figure(figsize=(15, 5))
plt.plot(MVu,'r--')
plt.plot(P_,'g--')
plt.plot(I_,'b--')
plt.plot(D_,'y--')


plt.xlabel('Time (secs)')
plt.ylabel('Output signal u(t)')
plt.legend(['Total out put signal u(t)','P','I','D'])

plt.grid()
plt.show() 




