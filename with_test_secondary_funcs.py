import numpy as np
from numpy import genfromtxt
import datetime
from scipy.optimize import minimize
from scipy.special import gamma
from numpy import genfromtxt
import os
import math
from scipy.optimize import Bounds
import matplotlib.pyplot as plt



 
conf_matrix = genfromtxt('C:\\Users\\jacgwatkins\\Data\\z_code\\minimization_code\\conf_matrix.txt')		### load inverse confusion matrix from saved file, the correct one, transposed such that sum(col) = 1
	
cov_matrix = genfromtxt('C:\\Users\\jacgwatkins\\Data\\z_code\\minimization_code\\cov_matrix.txt')		### load covariance matrix from saved file (obtained from classification algorithm performance)

inv_cov_matrix = np.linalg.inv(cov_matrix)

inv_conf_matrix = np.linalg.inv(conf_matrix) 

avg_array = inv_conf_matrix.flatten()      ### flattened confusion matrix of averages, to be later subtracted from r_array


## matrices used for defining boundaries in minimization for defining boundaries:
diag_cov_matrix = np.zeros(36)

for i in range(36):
    diag_cov_matrix[i] = cov_matrix[i][i]
    
diag_cov_matrix = np.reshape(diag_cov_matrix, (6,6))
diag_cov_matrix = diag_cov_matrix.transpose()
diag_cov_matrix = diag_cov_matrix.flatten()


flat_conf_matrix = conf_matrix.transpose()
flat_conf_matrix = flat_conf_matrix.flatten()


global bnd

bnd = []
            
for i in range(36):
    
    bnd.append((max(flat_conf_matrix[i] - 3 * np.sqrt(diag_cov_matrix[i]), 0.000001), min(flat_conf_matrix[i] + 3 * np.sqrt(diag_cov_matrix[i]), 0.999999)))               ### defining bounds as +- 3*sigma around the average conf_matrix value, but also taking min as 0, and max as 1


global signal

global obs_array            

global scale

#scale = pow((1.1/10000),36)
scale = 1

###obs_array = [13, 19, 24, 19, 16, 9]

obs_array = genfromtxt('C:\\Users\\jacgwatkins\\Data\\predicitonsMC_xc_done\\total100_obs10k\\xc_signal0.1_obs10k.txt')[100]






### obs_array, signal defined globally in the "secondary" functions built from these 'primary' ones
	
def NLL(combined_array):     ### combined vector of the 6 probabilities, and the 36 matrix parameters (representing action of the algorithm on x_true (not its inverse, so we can impose the conditions in minimization) in a signle run) flattened, 42 in total,    		
        					 ### takes in the input this way, because the minimization function calls needs to be a flattened vector of the parameters
    
    
    ### trying different initalization for probs and flat aglorithm matrix mathod so I don't touch the array returnd from the minimization in case it affects how it works:
    
    probs_array = np.zeros(6)
    algorithm_matrix_flat = np.zeros(36)
    
    for i in range(6):
        probs_array[i] = combined_array[i]
    
    for i in np.arange(6,42):
        algorithm_matrix_flat[i-6] = combined_array[i]
        
        
        
    
    global gcounter
    
    gcounter += 1
        
#    temp_array = np.split(combined_array, [6])      ### splits input into 2 vectors, from 0 to 5 the probs vector, and the remaining 36 the flattened algorithm matrix (the r vector)   
    	
#    probs_array = temp_array[0]
	    
#    algorithm_matrix_flat = temp_array[1]
        
    algorithm_matrix = np.reshape(algorithm_matrix_flat, (6,6))
    
    algorithm_matrix = np.transpose(algorithm_matrix)               ### conf_matrix already tansposed before saving, so no need to redo

#    print ('algorithm matrix \n', algorithm_matrix, '\n')    

    r_matrix = np.linalg.inv(algorithm_matrix)				### inverse of matrix representing the action of matrix on particular obs array, to be found by minimization
#    	
    r_array = r_matrix.flatten()
    
   
    
  	## 1st term in NLL:
    first_term = math.log(abs(np.linalg.det(algorithm_matrix)))
	
	
	## 2nd term in NLL 
    outer_sum = 0
    for i in range(6):
        inner_sum = 0
		
        for j in range(6):
			
            inner_sum += (r_matrix[i][j] * obs_array[j])
			 
        outer_sum += inner_sum * math.log(probs_array[i])
						
    second_term = (-1) * outer_sum


	## 3rd term in NLL:
    outer_sum = 0
    for m in range(6):
        inner_sum = 0
		
        for n in range(6):
			
            inner_sum += (r_matrix[m][n] * obs_array[n])
            
            
        bracket_value  =  gamma(inner_sum + 1)
        
        try:        
        
            outer_sum += math.log( bracket_value )    ## +1 because gamma(n) = (n-1)!
		
        except:
            print ('ERROR!!!!!!')
            print ('counter: ', gcounter, '\n')
            print ('obs_array: ', obs_array, '\n')
            print ('prob minima \n', probs_array, '\n')
            print ('algorithm_matrix; \n', algorithm_matrix, '\n')
            print ('conf_matrix \n', conf_matrix, '\n')
            print ('inverse r matrix: \n', r_matrix, '\n')
            print ('multiplication of determinants: ', np.linalg.det(algorithm_matrix) * np.linalg.det(r_matrix), '\n')
            print ('first term is ', first_term, '\n')
            print ('second term is ', second_term, '\n')
            print ('inv cov matrix \n' , inv_cov_matrix, '\n')
            err_matrix = np.zeros((6,6))
            cov_matrix_diag_new = np.zeros((6,6))
            
            for i in range(6):
                for j in range(6):
                    cov_matrix_element = np.sqrt(cov_matrix[i*6+j][i*6+j])
                    cov_matrix_diag_new[i][j] = cov_matrix_element
                    err_matrix[i][j] = (algorithm_matrix[i][j] - conf_matrix[i][j])/cov_matrix_element
            
            print ('cov_matrix_diag \n', cov_matrix_diag_new, '\n')    
                    
            print ('err_matrix \n', err_matrix, '\n')
            
            for i in range(6):
                print ('sum of column ', i, 'is ', sum(algorithm_matrix[:,i]), '\n')
            
            
            for i in range(6):
                print ('number of predicted events of type ', i, 'is ' , np.dot(r_matrix[i], obs_array), '\n')
                
            print ('sum of probs ', sum(combined_array[0:6]), '\n')
           
            
            print ('press to continue')
            
            if abs(sum(combined_array[0:6])-1) > 0.01 or abs(sum(combined_array[6:12])-1) > 0.01 or abs(sum(combined_array[12:18])-1) > 0.01 or abs(sum(combined_array[18:24])-1) > 0.01 or abs(sum(combined_array[24:30])-1) > 0.01 or abs(sum(combined_array[30:36])-1) > 0.01 or abs(sum(combined_array[36:42])-1) > 0.01:
                print ('sum to 1 constraints not respected')
                print ('press to continue \n')
                
                print ('sum of probs ', sum(combined_array[0:6]), '\n')
                print ('sum first 6 Pij ', sum(combined_array[6:12]), '\n')
                print ('sum second 6 Pij ', sum(combined_array[12:18]), '\n')
                print ('sum third 6 Pij ', sum(combined_array[18:24]), '\n')
                print ('sum fourth 6 Pij ', sum(combined_array[24:30]), '\n')
                print ('sum fifth 6 Pij ', sum(combined_array[30:36]), '\n')
                print ('sum sixth 6 Pij ', sum(combined_array[36:42]), '\n')
                
                print ('-------------------------------------------')
                
                x = input()

                return 1000
            
        
    third_term = outer_sum
		
		
	## 4th term in NLL:
    deviation_array = r_array - avg_array			## deviation from the average
	
    temp_fourth_term = np.dot(inv_cov_matrix, deviation_array)
	
    fourth_term = np.dot(deviation_array, temp_fourth_term)
	
    fourth_term = 0.5 * scale * fourth_term
    	
    value = first_term + second_term + third_term + fourth_term
    
    
    
    if gcounter%100 == 0:
    
        print ('obs_array: ', obs_array, '\n')
        print ('prob minima \n', probs_array, '\n')
        print ('algorithm_matrix; \n', algorithm_matrix, '\n')
        print ('conf_matrix \n', conf_matrix, '\n')
        print ('inverse conf: \n', r_matrix, '\n')
        print ('NLL value: \n', value, '\n')
        print ('first term is ', first_term, '\n')
        print ('second term is ', second_term, '\n')
        print ('third term is ', third_term, '\n')
        print ('fourth term is ', fourth_term, '\n')
        print ('temp fourth term ', temp_fourth_term, '\n')
        print ('inv cov matrix \n' , inv_cov_matrix, '\n')
        print ('deviation array \n', deviation_array, '\n')
        err_matrix = np.zeros((6,6))
        cov_matrix_diag_new = np.zeros((6,6))
        
        for i in range(6):
            for j in range(6):
                cov_matrix_element = np.sqrt(cov_matrix[i*6+j][i*6+j])
                cov_matrix_diag_new[i][j] = cov_matrix_element
                err_matrix[i][j] = (algorithm_matrix[i][j] - conf_matrix[i][j])/cov_matrix_element
        
        print ('cov_matrix_diag \n', cov_matrix_diag_new, '\n')    
                
        print ('err_matrix \n', err_matrix, '\n')
        
        for i in range(6):
            print ('sum of column ', i, 'is ', sum(algorithm_matrix[:,i]), '\n')
        
        
        for i in range(6):
            print ('number of predicted events of type ', i, 'is ' , np.dot(r_matrix[i], obs_array), '\n')
            
        
        print ('counter: ', gcounter, '\n')
        
        print ('-------------------------------------------')
        
        print ('space to continue')
        
        x = input()
                
        	
            
        if abs(sum(combined_array[0:6])-1) > 0.01 or abs(sum(combined_array[6:12])-1) > 0.01 or abs(sum(combined_array[12:18])-1) > 0.01 or abs(sum(combined_array[18:24])-1) > 0.01 or abs(sum(combined_array[24:30])-1) > 0.01 or abs(sum(combined_array[30:36])-1) > 0.01 or abs(sum(combined_array[36:42])-1) > 0.01:
            print ('sum to 1 constraints not respected')
            print ('press to continue')
            
            print ('sum of probs ', sum(combined_array[0:6]), '\n')
            print ('sum first 6 Pij ', sum(combined_array[6:12]), '\n')
            print ('sum second 6 Pij ', sum(combined_array[12:18]), '\n')
            print ('sum third 6 Pij ', sum(combined_array[18:24]), '\n')
            print ('sum fourth 6 Pij ', sum(combined_array[24:30]), '\n')
            print ('sum fifth 6 Pij ', sum(combined_array[30:36]), '\n')
            print ('sum sixth 6 Pij ', sum(combined_array[36:42]), '\n')
            
            print ('-------------------------------------------')
          
            x = input()

        
    return value
	
	
	
	
	
	
#def NLL2(combined_array):			### same as NLL function but with fixed signal parameter, 41 parameters to be minimized in total, 
#									### the first five elements of the combined_array are the background probabilities, then 36 algorithm matrix elements flattened
#	
#	
#    temp_array = np.split(combined_array, [5])      ### splits input into 2 vectors, from 0 to 4 the bkgrnd probs vector, and the remaining 36 the flattened algorithm matrix (the r vector)
#	
#    probs_array = temp_array[0]   
#	
#    probs_array = np.insert(probs_array, 0, signal, axis = 0)  		### adds signal to the beginning of the obs_array, so the calculations can be performed as before, also signal is set outside in the secondary fucntions
#	
#    algorithm_matrix_flat = temp_array[1]               ### matrix representing algorithm (flattened) action on true vector to give classified vector
#	    
#    algorithm_matrix = np.reshape(algorithm_matrix_flat, (6,6))
#    
#    algorithm_matrix = np.transpose(algorithm_matrix)	
#    
#    r_matrix = np.linalg.inv(algorithm_matrix)				### matrix representing the action of matrix on particular obs array, to be found by minimization
#    	
#    r_array = r_matrix.flatten()
#    
#  	## 1st term in NLL:
#    first_term = np.linalg.det(algorithm_matrix)
#	
#	
#	## 2nd term in NLL:
#    outer_sum = 0
#    for i in range(6):
#        inner_sum = 0
#		
#        for j in range(6):
#			
#            inner_sum += (r_matrix[i][j] * obs_array[j])
#			 
#        outer_sum += inner_sum * math.log(probs_array[i])
#						
#    second_term = (-1) * outer_sum
#
#
#	## 3rd term in NLL:
#    outer_sum = 0
#    for m in range(6):
#        inner_sum = 0
#		
#        for n in range(6):
#			
#            inner_sum += (r_matrix[m][n] * obs_array[n])
#		
#        outer_sum += math.log( gamma(inner_sum + 1) )    ## +1 because gamma(n) = (n-1)!
#		
#    third_term = outer_sum
#		
#		
#	## 4th term in NLL:
#    deviation_array = r_array - avg_array			## deviation from the average
#	
#    fourth_term = np.dot(inv_cov_matrix, deviation_array)
#	
#    fourth_term = np.dot(deviation_array, fourth_term)
#	
#    fourth_term = 0.5 * fourth_term
#	
#    
#    return first_term + second_term + third_term + fourth_term
#	
	
	
	
	
### easier to keep the top two funcs separated for the purpose of defining constraints in minimization 	
	
	 
	
def NLL_minimize(function):      ### the function passed is the one to be minimized, checks for which function it is and adjusts minimization conditions accordingly
	
    global gcounter
    
    gcounter = 0
    
    checker = True
	 
    if function == NLL:
        
        
        while checker == True:           ### repeates and checker becomes False, as long as its True it will repeat
                
            
#            try:
#                x0 =np.random.uniform(0.01, 1, 6)    		### initializing probablities between 0 and 1, and below initializing matrix elements to the average confmatrix elements
			###################
            x0 = (1/6) * np.ones(6)
            
#            for i in range(6):
#                x0[i] = np.random.uniform(0, 1 - sum(x0))
            
            
            
            ####################
#            elements_init = np.zeros(36)
#            
#            for k in range(36):
#                elements_init[k] = np.random.uniform(bnd[k][0], bnd[k][1])
#            
#            
#            x0 = np.append(x0, elements_init)
#            ###########################


            x0 = np.append(x0, flat_conf_matrix)      
			
            
                        
            
            
#            bnd = [(0.799,0.823),(0.051,0.059),(0.007,0.015),(0.064,0.08),(0.032,0.04),(0.011,0.019),(0.045,0.069),(0.785,0.817),(0.0017,0.0033),(0,0.00063),(0.127,0.151),(0,0.0007),(0.034,0.05),(0.0006,0.0086),(0.709,0.741),(0.092,0.116),(0.108,0.14),(0,0.0007),(0.135,0.167),(0.036,0.052),(0.146,0.178),(0.472,0.512),(0.07,0.086),(0.061,0.085),(0.093,0.117),(0.201,0.233),(0.167,0.191),(0.0352,0.0528),(0.446,0.486),(0,0.025),(0.208,0.24),(0.13,0.162),(0.0125,0.0285),(0.103,0.135),(0.06,0.084),(0.399,0.439)]
            
            
            prob_bnd = (10**(-4), 1 - 10**(-4))
            bounds = (prob_bnd, prob_bnd, prob_bnd, prob_bnd, prob_bnd, prob_bnd, bnd[0], bnd[1], bnd[2], bnd[3], bnd[4], bnd[5], bnd[6], bnd[7], bnd[8], bnd[9], bnd[10], bnd[11], bnd[12], bnd[13], bnd[14], bnd[15], bnd[16], bnd[17], bnd[18], bnd[19], bnd[20], bnd[21], bnd[22], bnd[23], bnd[24], bnd[25], bnd[26], bnd[27], bnd[28], bnd[29], bnd[30], bnd[31], bnd[32], bnd[33], bnd[34], bnd[35])                                              
			
            eq_cons = {'type': 'eq', 'fun' : lambda x: np.array([x[0] + x[1] + x[2] + x[3] + x[4] + x[5] - 1, x[6] + x[7] + x[8] + x[9] + x[10] + x[11] - 1, x[12] + x[13] + x[14] + x[15] + x[16] + x[17] - 1, x[18] + x[19] + x[20] + x[21] + x[22] + x[23] - 1, x[24] + x[25] + x[26] + x[27] + x[28] + x[29] - 1, x[30] + x[31] + x[32] + x[33] + x[34] + x[35] - 1, x[36] + x[37] + x[38] + x[39] + x[40] + x[41] - 1])}   	 ### equality constraint for SLSQP algorithm, the sum of probabilities (first 6) have has to be 1, then every following 6 (which a a matrix row) have to sum to 6
			
					
            res = minimize(function, x0, method='SLSQP', options={'ftol': 1e-10}, constraints = [eq_cons], bounds = bounds)    
			
			
            minimization_res = res.x
			
            
            if abs(sum(minimization_res[0:6])-1) > 0.01 or abs(sum(minimization_res[6:12])-1) > 0.01 or abs(sum(minimization_res[12:18])-1) > 0.01 or abs(sum(minimization_res[18:24])-1) > 0.01 or abs(sum(minimization_res[24:30])-1) > 0.01 or abs(sum(minimization_res[30:36])-1) > 0.01 or abs(sum(minimization_res[36:42])-1) > 0.01:
                checker = True
                print ('sum is incorrect')
            else:
                checker = False
                
                
#            counter += 1
#            if counter > 50:
#                checker = False   ## so doesn't loop for too long
#                    
#            except:
#                checker = True

#            print (minimization_res)
#    elif function == NLL2:
#	
#        while checker == True:           ### repeates and checker becomes False, as long as its True it will repeat
#		
#			
#            x0 =np.random.uniform(0, 1-signal, 41)    		### 41 is no. of parameters to be minimized in NLL2 (with fixed signal)
#			
#			
#            bnd = (0,1) 								### here bound for probabilities is 0 to 1-signal, but matrix elements is still 0 to 1 
#            prob_bnd2 = (0.001, 1 - signal)
#			
#            bounds = (prob_bnd2, prob_bnd2, prob_bnd2, prob_bnd2, prob_bnd2, bnd, bnd, bnd, bnd, bnd, bnd, bnd, bnd, bnd, bnd, bnd, bnd, bnd, bnd, bnd, bnd, bnd, bnd, bnd, bnd, bnd, bnd, bnd, bnd, bnd, bnd, bnd, bnd, bnd, bnd, bnd, bnd, bnd, bnd, bnd, bnd)                                              
#			
#            eq_cons = {'type': 'eq', 'fun' : lambda x: np.array([x[0] + x[1] + x[2] + x[3] + x[4] + x[5] - (1 - signal), x[6] + x[7] + x[8] + x[9] + x[10] + x[11] - 1, x[12] + x[13] + x[14] + x[15] + x[16] + x[17] - 1, x[18] + x[19] + x[20] + x[21] + x[22] + x[23] - 1, x[24] + x[25] + x[26] + x[27] + x[28] + x[29] - 1, x[30] + x[31] + x[32] + x[33] + x[34] + x[35] - 1, x[36] + x[37] + x[38] + x[39] + x[40] + x[41] - 1])}   	 ### double check, is the matrix element corresponding to signal floating or fixed, because if it is fixed, this will change the constraint on matrix element, I'm almost certain it floats, and conditions are the same as above
#			
#					
#            res = minimize(function, x0, method='SLSQP', options={'ftol': 1e-30}, constraints = [eq_cons], bounds = bounds)    ## check if htis works w/o defining the jacobian
#			
#            
#            minimization_res = res.x
#			
#
#
#            if isinstance(minimization_res[0], complex) or isinstance(minimization_res[1], complex) or isinstance(minimization_res[2], complex) or isinstance(minimization_res[3], complex) or isinstance(minimization_res[4], complex) or isinstance(minimization_res[5], complex):
#                checker = True    ###checking to see if there are any complex values, to repeat htis step, till all of them aren't complex
#            elif minimization_res[0] <= 0 or minimization_res[1] <= 0 or minimization_res[2] <= 0 or minimization_res[3] <= 0 or minimization_res[4] <= 0 or minimization_res[5] <= 0 :
#                checker = True    ### all need to be +ve, for checker to become False and the while loop to terminate, and move on ot next input vector, if any are negative, iteration repeats
##            elif abs(np.sum(minimization_res) - 1) > 0.09:    ###checking sum is correct
##                checker = True
#            else:
#                checker = False	
	
	
	
    
    return minimization_res
	
	
		
		
		
### didn't end up using this, but leave it for now:				
		
		
def new_folder(dir_name):													### function to create new folder, checks if folder name already exists first

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
																					
    else:
        print('folder already exists')







#def NLLshort(combined_array):
#    
#    temp_array = np.split(combined_array, [6])      ### splits input into 2 vectors, from 0 to 5 the probs vector, and the remaining 36 the flattened algorithm matrix (the r vector)   
#    		    
#    algorithm_matrix_flat = temp_array[1]
#        
#    algorithm_matrix = np.reshape(algorithm_matrix_flat, (6,6))
#    
#    algorithm_matrix = np.transpose(algorithm_matrix)               ### conf_matrix already tansposed before saving, so no need to redo
#    
#    r_matrix = np.linalg.inv(algorithm_matrix)				### inverse of matrix representing the action of matrix on particular obs array, to be found by minimization
##    	
#    r_array = r_matrix.flatten()
#
#
#	## 4th term in NLL:
#    deviation_array = r_array - avg_array			## deviation from the average
#	
#    fourth_term = np.dot(inv_cov_matrix, deviation_array)
#	
#    fourth_term = np.dot(deviation_array, fourth_term)
#	
#    fourth_term = 0.5 * fourth_term
#	
#	
#	
#    return fourth_term














### SECONDARY functions, the function to get the NLL profile points, and function to get minima is defined using the 4 primary functions


def NLL_profile_points(obs):									### arrays get saved to a new folder inside the output path passed
        
    obs_array = obs
	
    n_inc = 50 
	
    signal_vals = []
    NLL_vals = []
	
    predictions = NLL_minimize(NLL)
	
    signal = predictions[0]
	
    original_NLL = NLL(predictions)
	
    signal_vals.append(signal)
    NLL_vals.append(original_NLL)
	
	
    positive_inc = (1-signal)/n_inc
	
    for i in range(n_inc):
		
        signal += positive_inc
		
		
        predictions2 = NLL_minimize(NLL2)         ### background and matrix elements minima (excluding singal probability)
		
		
        signal_vals.append(signal)
        NLL_vals.append(NLL2(predictions2))
		
		
        if i > 5:
            break        
        
        if NLL2(predictions2) > (original_NLL + 1):
            if NLL2(predictions2) > (original_NLL + 4):
                n_inc = 500
    
            else:
                break
				
	



    signal = predictions[0]
	
    negative_inc = signal/n_inc
	
    for k in range(n_inc):
		
        signal -= negative_inc
		
        predictions2 = NLL_minimize(NLL2)
		
		
        signal_vals.append(signal)
        NLL_vals.append(NLL2(predictions2))
		
        if i > 5:
            break        
        
        if NLL2(predictions2) > (original_NLL + 1):
            if NLL2(predictions2) > (original_NLL + 4):
                n_inc = 500
    
            else:
                break
	

    arrays = np.vstack((signal_vals, NLL_vals))
    return arrays
	
	
	
	
	
	
	
	
	
def get_profiles(obs_path, output_path, n_events, true_signal):   					### input path is the obs file organized in 6 columns for each event type, the output path is the folder to save the outputs, nevents is the no. of events in each observation vector
	
    print ('start time is: ', datetime.datetime.now())
	
    obs_file = genfromtxt(obs_path)
	
    output_path = output_path + '\\signal' + str(true_signal) + '_events' + str(n_events) + '_signal_NLL_arrays'
	
    new_folder(output_path)
	
    for row_index in range(len(obs_file)):
		
        global obs_array
        
        obs_array = obs_file[row_index, :]
		
        current_output_path = output_path + '\\signal_NLL_arrays' + str(row_index + 1) + '_TrueSig' + str(true_signal) + '_TotalEvents' + str(n_events) + '.txt'
		
        current_arrays_file = NLL_profile_points(obs_array)
		
        np.savetxt(current_output_path, current_arrays_file)
		
		
        print ('end time is: ', datetime.datetime.now())
	
	
	
	
def get_obs_minima(obs_path, output_path, n_events, true_signal):				### input paths is the obs file same as above, output path is where the matrix len(obs_file)x42 (saving all 6 probabs, and the 36 parameters) is going to be saved, becasue easier to save this to the same file, where for profiels, easier to save NLL/singal arrays of each obs to a separate profile, due to varying lengths between each
	
    print ('start time is: ', datetime.datetime.now())
	
    output_path = output_path + '\\signal' + str(true_signal) + '_events' + str(n_events) + '_NLL_minima'
	
    new_folder(output_path)
	
    output_path = output_path + '\\Test5_TrueSig_' + str(true_signal) + '_totalEvents_' + str(n_events) + '_minima.txt'
	
    obs_file = genfromtxt(obs_path)
	
    minima_file = np.zeros(42)  	## just a place holder to enable use of vstack fucntion below
	
    
    for row_index in range(len(obs_file)):
        
        global obs_array
        
        obs_array = obs_file[row_index, :]
		
#        try:
            
        current_minima = NLL_minimize(NLL)
        
#            current_minima = np.append(current_minima , NLL(current_minima))
		
        minima_file = np.vstack((minima_file, current_minima))
		
        print (row_index, ' is done')
    	
        ## for testing:
        
#        except:
#            print('error')
            
#            continue
		
        if row_index > 100:
            break
		
    
    minima_file = minima_file[1:,:]			### getting rid of the placeholder zeros row
	
    np.savetxt(output_path, minima_file)
	
    print ('end time is: ', datetime.datetime.now())
    
    
    ###################################################################
   
NLLs = []
    
for h in range(100):
#    try:
    NLLs.append(NLL(NLL_minimize(NLL)))
#    except:
#        print('error')
#        continue
 
print (len(NLLs))
print (NLLs)    
    
hist, bin_edges = np.histogram(NLLs, density=True)

bin_centres = (bin_edges[:-1] + bin_edges[1:])/2

plt.plot(bin_centres, hist)