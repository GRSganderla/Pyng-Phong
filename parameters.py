import json

class coefficients():
	pass

position = 1;
orientation = None;
focal_distance = None;
screen_width = None;
screen_height = None;
pixel_size_x = None;
pixel_size_y = None;

cfs = coefficients();
cfs.a = None;
cfs.b = None;
cfs.c = None;
cfs.d = None;
cfs.e = None;
cfs.f = None;
cfs.g = None;
cfs.h = None;
cfs.j = None;
cfs.k = None;

light = None;

cfb = coefficients();
cfb.difuse = None;
cfb.specular = None;
cfb.ambient = None;
cfb.n = None;

# load file and return json dictionary
def load_file(file_path):
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
	global position, orientation, focal_distance, screen_width;
	global screen_height, pixel_size_x, pixel_size_y;

	position = parameters['position'];
	orientation = parameters['orientation'];
	focal_distance = parameters['focal_distance'];
	screen_width = parameters['screen_width'];
	screen_height = parameters['screen_height'];
	pixel_size_x = parameters['pixel_size_x'];
	pixel_size_y = parameters['pixel_size_y'];

	global cfs;

	cfs.a = parameters['cfs']['a'];
	cfs.b = parameters['cfs']['b'];
	cfs.c = parameters['cfs']['c'];
	cfs.d = parameters['cfs']['d'];
	cfs.e = parameters['cfs']['e'];
	cfs.f = parameters['cfs']['f'];
	cfs.g = parameters['cfs']['g'];
	cfs.h = parameters['cfs']['h'];
	cfs.j = parameters['cfs']['j'];
	cfs.k = parameters['cfs']['k'];

	global light;

	light = parameters['light'];

	global cfb;

	cfb.difuse = parameters['cfb']['difuse'];
	cfb.specular = parameters['cfb']['specular'];
	cfb.ambient = parameters['cfb']['ambient'];
	cfb.n = parameters['cfb']['n'];
