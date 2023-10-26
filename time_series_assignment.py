import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import statsmodels.graphics.tsaplots as tsa
from statsmodels.tsa.ar_model import AutoReg
from scipy import optimize
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import statistics as st
from statistics import NormalDist


# You can plot all figures from the report by uncommenting the
# plotting functions


############################################ Functions ###################################################################

def perform_boxcar(data, n): # Performs boxcar smoothing
    smoothed_data = data.copy()
    smoothed_data['smoothed_wind_speed'] = data['wind_speed'].rolling(window=(2*n+1), center=True, min_periods=1).mean()
    smoothed_data.dropna(subset=['smoothed_wind_speed'], inplace=True)
    smoothed_data.reset_index(drop=True, inplace=True)
    return smoothed_data[['date', 'smoothed_wind_speed']]


def sin_func(t, A, B, C, D): # Sin function to minimize
    y = A + B*np.sin(C*t + D)
    return y


def get_w_array(phi_0, phi_1, phi_2, l, s, a0, a1): # Constructs the noise array
    X = np.random.normal(loc=0, scale=s, size=400)
    initial_series=np.zeros(l)
    initial_series[0] = a0
    initial_series[1] = a1
    
    for i in range(2, l):
        initial_series[i] = phi_0 + phi_1*initial_series[i-1] + phi_2*initial_series[i-2] + X[i]

    return initial_series

#########################################################################################################################



######################################### Data Pre-processing ##############################################################

# Read in data
data = pd.read_csv('/home/lmeredith/data_analysis_and_machine_learning/Data/DailyDelhiClimateTrain.csv', header=0)
n = len(data)
print(n)


# Check if data is clean
print(data.columns[data.isna().any()].tolist())

# Remove columns I don't want
data.drop(data.columns[[1,2,4]], axis=1, inplace=True)

# Remove outliers from the data
mean_wind_speed = np.mean(data['wind_speed'])
variance = (n/(n-1))*np.var(data['wind_speed'])
std = np.sqrt(variance)
cutoff = mean_wind_speed + 2*std
data = data[data['wind_speed'] < cutoff].reset_index(drop=True)


# Split data into 2013-2015 and 2015-2017
start_date = '2013-01-01'
end_date = '2015-12-31'
training_data, test_data  = data[data['date'] <= end_date], data[data['date'] > end_date]


# Smooth the data using boxcar smoothing and can plot to see the difference
window_size = 14
smoothed_training_data = perform_boxcar(training_data, window_size)
smoothed_test_data = perform_boxcar(test_data, window_size)


# Reproduces figure 1

# plt.plot(training_data['date'], 
#          training_data['wind_speed'], 
#          color='black', label='Raw data')
# plt.plot(smoothed_training_data['date'], 
#          smoothed_training_data['smoothed_wind_speed'], 
#          color='red', label='Smoothed data n=14')
# plt.xticks(['2013-01-01', '2014-01-01', '2015-01-01', '2015-12-31'], size=15)
# plt.yticks(size=15)
# plt.xlabel('Time', size=20)
# plt.ylabel('Wind Speed km/h', size=20)
# plt.legend()
# plt.show()


#########################################################################################################################




######################## Sine Regression ################################################################################

t = np.linspace(0, len(smoothed_training_data), len(smoothed_training_data))

# Initial guesses
amplitude = 4
frequency = 0.03
phase = 0
offset = 6
popt, pcov = optimize.curve_fit(sin_func, t, 
                                smoothed_training_data['smoothed_wind_speed'], 
                                p0=[amplitude, frequency, phase, offset])

new_amp, new_freq, new_phase, new_offset = popt[0], popt[1], popt[2], popt[3]
deterministic_array = [sin_func(i, new_amp, new_freq, new_phase, new_offset) 
                       for i in t] # Deterministic part

full_time_series = [i for i in smoothed_training_data['smoothed_wind_speed']]


# Plots figure 2

# plt.plot(t, full_time_series, color='black', label='Series data')
# plt.plot(t, deterministic_array, color='red', label='Sin fit')
# plt.xlabel('Time (days)', size=20)
# plt.ylabel('Wind speed km/h', size=20)
# plt.xticks(size=15)
# plt.yticks(size=15)
# plt.legend()
# plt.show()

###############################################################################################




############################# Isolate the noise data ##########################################


# Convert to np.arrays for easier manipulation 
D_array = np.zeros(len(deterministic_array))
Y_array = np.zeros(len(full_time_series))
for i in range(len(deterministic_array)):
    D_array[i] = deterministic_array[i]
    Y_array[i] = full_time_series[i]

W_array = Y_array - D_array # Isolated noise array

###############################################################################################




############################ Dicky-Fuller Test ################################################
# Code from online webpage: https://machinelearningmastery.com/time-series-data-stationary-python/

result = adfuller(W_array)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
if result[0] < result[4]["5%"]:
    print ("Reject Ho - Time Series is Stationary")
else:
    print ("Failed to Reject Ho - Time Series is Non-Stationary")

############################## Stationary #####################################################





############################# ACF and PACF ####################################################

# This part of the code plots the ACF and PACF
# PACF plot is figure 3

# acf = tsa.plot_acf(W_array, lags=20) # No abrupt drop => No MA(q) process
# pacf = tsa.plot_pacf(W_array, lags=20) # Abrupt drop at lag 2 => AR(2) process
# plt.xlabel('Lag', size=20)
# plt.xticks(size=15)
# plt.yticks(size=15)
# plt.title('')
# plt.ylabel('Partial Autocorrelations', size=20)
# plt.show()

###############################################################################################





############################# Perform AR(2) ###################################################


ar_model_fit = AutoReg(W_array, lags=2).fit()
phi0, phi1, phi2 = ar_model_fit.params[0], ar_model_fit.params[1], ar_model_fit.params[2]

# Use the last 2 data points from training noise set to get first 2 points of the
# generated noise array
a_2, a_1 = W_array[1052], W_array[1053]
ar1 = phi0 + phi1*a_1 + phi2*a_2 + np.random.normal(loc=0, scale=0.1)
ar2 = phi0 + phi1*ar1 + phi2*a_1 + np.random.normal(loc=0, scale=0.1)


test_times = [i for i in range(len(smoothed_test_data['date']))]
noise_prediction = get_w_array(phi0, phi1, phi2, len(test_times), s=0.1, a0=ar1, a1=ar2)
deterministic_test_array = [sin_func(i, new_amp, new_freq, new_phase, new_offset) 
                            for i in test_times]
D_test_arr = np.zeros(len(deterministic_test_array))
for i in range(len(deterministic_test_array)):
    D_test_arr[i] = deterministic_test_array[i]

deterministic_test_array = [sin_func(i, new_amp, new_freq, new_phase, new_offset) 
                            for i in test_times]
test_speeds = [i for i in smoothed_test_data['smoothed_wind_speed']]
time_series_prediction = deterministic_test_array + noise_prediction

rmse = np.sqrt(mean_squared_error(test_speeds, time_series_prediction))
print("RMSE:", rmse)


# Plots figure 4

# plt.plot(test_times, test_speeds, color='black', label='Smoothed test data')
# plt.plot(test_times, time_series_prediction, color='red', label='Prediction')
# plt.xlabel('Time (days)', size=20)
# plt.ylabel('Wind speed km/h', size=20)
# plt.xticks(size=15)
# plt.yticks(size=15)
# plt.legend()
# plt.show()




#################################################################################################






############################### Repeat 100000 times #############################################

sims = 1000
all_sims = []


for s in range(sims):

    test_times = [i for i in range(len(smoothed_test_data['date']))]
    noise_prediction = get_w_array(phi0, phi1, phi2, len(test_times), 0.1, a0=ar1, a1=ar2)

    deterministic_test_array = [sin_func(i, new_amp, new_freq, new_phase, new_offset) 
                                for i in test_times]
    
    D_test_arr = np.zeros(len(deterministic_test_array))
    for i in range(len(deterministic_test_array)):
        D_test_arr[i] = deterministic_test_array[i]

    deterministic_test_array = [sin_func(i, new_amp, new_freq, new_phase, new_offset) 
                                for i in test_times]
    
    test_speeds = [i for i in smoothed_test_data['smoothed_wind_speed']]
    time_series_prediction = noise_prediction
    all_sims.append(time_series_prediction)

maximum = np.max(all_sims, axis=0)


# Plots figure 5

# plt.plot(test_times, test_speeds, color='black', label='Smoothed test data')
# plt.errorbar(test_times, deterministic_test_array + time_series_prediction, 
#              maximum, label='Prediction', alpha=0.3, color='red')
# plt.xlabel('Time (days)', size=20)
# plt.ylabel('Wind speed km/h', size=20)
# plt.yticks(size=15)
# plt.xticks(size=15)
# plt.legend()
# plt.show()


#############################################################################################