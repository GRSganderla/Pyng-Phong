import parameters as pr

# load
parametersDict = pr.load_file('parameters.json');
#print(parametersDict);

# initialize vars
pr.initialize_vars(parametersDict);

# print to see values
print('position at test =', pr.position);
print(pr.orientation);
print(pr.coef_surface.a);
print(pr.coef_band.difuse);
