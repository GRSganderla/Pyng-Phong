import json

class coefficients():
	pass

position = None;
orientation = None;
focal_distance = None;
screen_width = None;
screen_height = None;
pixel_size_x = None;
pixel_size_y = None;

coef_surface = coefficients();
coef_surface.a = None;
coef_surface.b = None;
coef_surface.c = None;
coef_surface.d = None;
coef_surface.e = None;
coef_surface.f = None;
coef_surface.g = None;
coef_surface.h = None;
coef_surface.j = None;
coef_surface.k = None;

light = None;

coef_band = coefficients();
coef_band.difuse = None;
coef_band.specular = None;
coef_band.ambient = None;
coef_band.n = None;

# load file and return json dictionary
def load(file_path):
	parameters = None;

	# with json file open readonly
	with open(file_path, 'r') as param_file:
		# load json
		parameters = json.load(param_file);

	# if json wasn't loaded correctly
	if(parameters == None):
		# error
		print('Error on');
		return None;

	return parameters;

# attribute values to global variables from given parameter dictionary
def initialize_vars(parameters):
	position = parameters['position'];
	orientation = parameters['orientation'];
	focal_distance = parameters['focal_distance'];
	screen_width = parameters['screen_width'];
	screen_height = parameters['screen_height'];
	pixel_size_x = parameters['pixel_size_x'];
	pixel_size_y = parameters['pixel_size_y'];

	coef_surface.a = parameters['coef_surface']['a'];
	coef_surface.b = parameters['coef_surface']['b'];
	coef_surface.c = parameters['coef_surface']['c'];
	coef_surface.d = parameters['coef_surface']['d'];
	coef_surface.e = parameters['coef_surface']['e'];
	coef_surface.f = parameters['coef_surface']['f'];
	coef_surface.g = parameters['coef_surface']['g'];
	coef_surface.h = parameters['coef_surface']['h'];
	coef_surface.j = parameters['coef_surface']['j'];
	coef_surface.k = parameters['coef_surface']['k'];

	light = parameters['light'];

	coef_band.difuse = parameters['coef_band']['difuse'];
	coef_band.specular = parameters['coef_band']['specular'];
	coef_band.ambient = parameters['coef_band']['ambient'];
	coef_band.n = parameters['coef_band']['n'];
