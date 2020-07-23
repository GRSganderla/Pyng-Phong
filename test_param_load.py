from parameters import *

# load
parametersDict = load('parameters.json');
print(parametersDict);

# initialize vars
initialize_vars(parametersDict);

# print to see values
print(position);
print(orientation);
print(coef_surface);
print(coef_band);