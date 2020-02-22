from __future__ import division
import os
import pandas as pd
import numpy as np
from pandas import HDFStore
from numpy import genfromtxt
import matplotlib.pyplot as plt
from scipy import log as ln
import sys 
from scipy.optimize import curve_fit

### load prob arrays for each true label, find mean and sigma, use means (make sure they approach written down values found before), to redefine conf matrix, use sigmas for cov matrix test, then find the covariances

## remember to transpose when defining conf_matrix

#prob_file = genfromtxt('C:\\Users\\jacgwatkins\\Data\\temp_conf_probs_arrays\\TrueLabel_0_489sets_100eventsEach_classProb.txt')
#
#for i in range(6):
#    
#    probs = prob_file[:,i]
#    print (np.mean(probs), np.std(probs))
#    
#    _ = plt.hist(probs, bins='auto')
#    
  
#ydata = prob_file[:,0]  
##
#_ = plt.hist(ydata, bins='auto')
#    
# 
#
#   
#hist, bin_edges = np.histogram(ydata, density=True)
#bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
#
##plt.plot(bin_centres, hist)
#
#### calculate covariances:
#    
#def gauss(x, mu, sigma, A):
#
#    return A*np.exp(-(x-mu)**2/(2*(sigma**2)))
#
#
#
##xdata = np.linspace(-10, 10, len(ydata))
#
##plt.plot(xdata, ydata, 'o', color='black');
#
##p0 = [0,0,0]
#
##bin_centers = np.linspace(-10, 10, len(ydata))
#
#coeff, var_matrix = curve_fit(gauss, bin_centres, hist)
#
#hist_fit = gauss(bin_centres, coeff[0], coeff[1], coeff[2])
#
#plt.plot(bin_centres, hist, label='Test data')
#plt.plot(bin_centres, hist_fit, label='Fitted data')
##
#print (np.mean(ydata))
#print (coeff[0])
#print (np.std(ydata))
#print (coeff[1])


###plt.plot(bin_centers, gauss(bin_centers, coeff[0], coeff[1], coeff[2]), 'b-', label='data')






#########calculate ovariance, this is just a test, cov matrix calculated after #-separation:

### laoad all 36 arrays for each matrix element, then use double for loop to calcute using np.cov(array1,array2), and take the 2nd element in the first row of the result as the covariance, and stor in an ampty 36 by 36 matrix:

### need to put loaded arrays into same structure to loop through all: 6x6x489 tensor , 36 elements of the matrix, and each element has a 489-element array attached to it (489 for the number of experiments we made from the test set)
#
#array1 = genfromtxt('C:\\Users\\jacgwatkins\\Data\\temp_conf_probs_arrays\\TrueLabel_1_489sets_100eventsEach_classProb.txt')[:,0]
#
#
#array2 = genfromtxt('C:\\Users\\jacgwatkins\\Data\\temp_conf_probs_arrays\\TrueLabel_1_489sets_100eventsEach_classProb.txt')[:,0]
#
#x = np.stack((array1, array2), axis=0)
#
#print (np.cov(x)[0][1])

##############################################################
file1 = genfromtxt('C:\\Users\\jacgwatkins\\Data\\temp_conf_probs_arrays\\TrueLabel_0_489sets_100eventsEach_classProb.txt')
file2 = genfromtxt('C:\\Users\\jacgwatkins\\Data\\temp_conf_probs_arrays\\TrueLabel_1_489sets_100eventsEach_classProb.txt')
file3 = genfromtxt('C:\\Users\\jacgwatkins\\Data\\temp_conf_probs_arrays\\TrueLabel_2_489sets_100eventsEach_classProb.txt')
file4 = genfromtxt('C:\\Users\\jacgwatkins\\Data\\temp_conf_probs_arrays\\TrueLabel_3_489sets_100eventsEach_classProb.txt')
file5 = genfromtxt('C:\\Users\\jacgwatkins\\Data\\temp_conf_probs_arrays\\TrueLabel_4_489sets_100eventsEach_classProb.txt')
file6 = genfromtxt('C:\\Users\\jacgwatkins\\Data\\temp_conf_probs_arrays\\TrueLabel_5_489sets_100eventsEach_classProb.txt')
#
### now store in dictionary to be able to iterate thorugh them
#
file_dictionary = {'file1': file1, 'file2': file2, 'file3': file3, 'file4': file4, 'file5': file5, 'file6': file6} 



placeHolder_matrix = np.zeros((6,6,489))


### remember to account for transpose ordeal
for i in range(6):
    for j in range(6):
        placeHolder_matrix[i][j] = file_dictionary['file{}'.format(i+1)][:,j]
        

placeHolder_matrix = placeHolder_matrix.transpose((1,0,2))  ###transposes first 2 dims


conf_matrix = np.zeros((6,6))

for k in range(6):
    for p in range(6):
        conf_matrix[k][p] = np.mean(placeHolder_matrix[k][p])


#print(conf_matrix)



       
placeHolder_matrix = placeHolder_matrix.reshape(1,36,489)       ### reshapes into a vector, of the other arrays



## now iterate through place holder matrix to find covariances, and store in 36 by 36 matrix:
        
cov_matrix = np.zeros((36,36))
cov_matrix_new = np.zeros((36,36))

for m in range(36):
    for n in range(36):
        cov_matrix[m][n] = np.cov(placeHolder_matrix[0][m], placeHolder_matrix[0][n])[0][1]
        ###sets values less than e-6 to 0:
#        if abs(cov_matrix[m][n]) < 10**(-6) and m != n:
#            cov_matrix[m][n] = 0
        
        ##removes off diagonal entries , to make cov matrix diagonal
        if m != n :
            cov_matrix[m][n] = 0
        ### chekcing code
#        vec1_avg = np.mean(placeHolder_matrix[0][m])
#        vec2_avg = np.mean(placeHolder_matrix[0][n])
#        sum = 0
#        for t in range(489):
#            sum += ( (placeHolder_matrix[0][m][t] - vec1_avg)*(placeHolder_matrix[0][n][t] - vec2_avg) )
#            
#        cov_matrix_new[m][n] = sum/488

#cov_matrix = (10000/1.1) * cov_matrix     ### trying scaling, doiesn't seem to work

#print (cov_matrix)  
     
##for i in range(36):
##    print (np.sqrt(cov_matrix[i][i]))
#
#
#path = 'C:\\Users\\jacgwatkins\\Data\\z_code\\minimization_code\\'

####conf: np.savetxt(path + 'conf_matrix.txt', conf_matrix)

####np.savetxt(path + 'cov_matrix.txt', cov_matrix)
#
#print (cov_matrix, '\n')
#print (cov_matrix_new)              ###for checking cov_matrix values

temp = cov_matrix[:36, :36]
#
#print (temp, '\n')

print ('cov_matrix det ', np.linalg.det(temp), '\n')

#print (np.linalg.inv(temp))
#
print ('inv_cov_matrix det ', np.linalg.det(np.linalg.inv(temp)), '\n')
#
print ('multiplication of determinants: ',np.linalg.det(temp)*np.linalg.det(np.linalg.inv(temp)), '\n')

#print (cov_matrix.min())

print ('multiplication of matrices: ', np.dot(cov_matrix, np.linalg.inv(cov_matrix)))

#diag_cov_matrix = np.zeros(36)
#
#for i in range(36):
#    diag_cov_matrix[i] = cov_matrix[i][i]
#    
#    
#print (diag_cov_matrix)