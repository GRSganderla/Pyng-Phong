import parameters

# load
parametersDict = parameters.load_param_file('parameters.json');
#print(parametersDict);

# initialize vars
parameters.initialize_param_vars(parametersDict);

# print to see values
print('position at test =', parameters.position);
print(parameters.orientation);
print(parameters.coef_surface.a);
print(parameters.coef_band.difuse);
