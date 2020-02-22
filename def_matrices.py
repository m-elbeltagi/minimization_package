import numpy as np


### file now obselets, conf, and cov matrices defined in load_conf_matrices file

### define and save confuson, covariance matrices, to be loaded in other files

path = 'C:\\Users\\jacgwatkins\\Data\\z_code\\minimization_code\\'

conf_matrix = np.array([[0.811, 0.055, 0.011, 0.072, 0.036, 0.015], [0.057, 0.801, 0.0025, 0.00023, 0.139, 0.0003], [0.042, 0.0046, 0.725, 0.104, 0.124, 0.0003], [0.151, 0.044, 0.162, 0.492, 0.078, 0.073], [0.105, 0.217, 0.179, 0.044, 0.466, 0.009], [0.224, 0.146, 0.0205, 0.119, 0.072, 0.419]])      

conf_matrix = np.transpose(conf_matrix)           ### transposed so it correctly represnts matrix action on x_true

inv_conf_matrix = np.linalg.inv(conf_matrix)

## for now diagonal covariance matrix, till we get covariances (using std devs matrix used before, so not to rewrite):

std_devs = np.array([[0.003, 0.001, 0.001, 0.002, 0.001, 0.001], [0.003, 0.004, 0.0002, 0.0001, 0.003, 0.0001], [0.002, 0.001, 0.004, 0.003, 0.004, 0.0001], [0.004, 0.002, 0.004, 0.005, 0.002, 0.003], [0.003, 0.004, 0.003, 0.0022, 0.005, 0.004], [0.004, 0.004, 0.002, 0.004, 0.003, 0.005]])    ##I use the larger error, some are asymmetric, so they have skewed distributions, but I'm just assuming they're gaussian with larger error as std dev

std_devs = np.transpose(std_devs)                 ### to account for transposing conf_matrix

std_devs = std_devs.flatten() 


cov_matrix = np.zeros((36, 36))

counter = 0

for row in cov_matrix:
	
	row[counter] = std_devs[counter] ** 2
	
	counter += 1
	
	
#np.savetxt(path + 'inv_conf_matrix.txt', inv_conf_matrix)

#np.savetxt(path + 'cov_matrix.txt', cov_matrix)
np.set_printoptions(precision=2)

print(conf_matrix)
    
