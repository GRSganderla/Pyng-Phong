import parameters as pr
import numpy as np
import math

image_matrix = None;
image_space_coords = None;
projection_matrix = None;
projection_center = None;

def generate_image(paramFile):
	# load
	parametersDict = pr.load_file(paramFile);
	#print(parametersDict);

	# initialize vars
	pr.initialize_vars(parametersDict);

	global projection_center;
	projection_center = np.array(pr.position) - pr.focal_distance * np.array(pr.orientation);

	global image_matrix;
	image_matrix = np.zeros([pr.screen_width, pr.screen_height]);
	#print(image_matrix);

	global image_space_coords;
	image_space_coords = calculate_image_space_coord(image_matrix);
	#print(image_space_coords);

	global projection_matrix;
	projection_matrix = calculate_projection_matrix(image_space_coords);
	#print(projection_matrix);


def calculate_image_space_coord(image_matrix):
	# generate an array same size as image matrix
	global image_space_coords;
	image_space_coords = np.full_like(image_matrix, point2D(), dtype=point2D);

	# loop through all elements and calculate it's space value
	for idxs, _ in np.ndenumerate(image_space_coords):
		# get indexes
		i = idxs[0]; j = idxs[1];
		# calculate x and y
		image_space_coords[i, j].x = (1.0/pr.pixel_size_x)*(i - (pr.screen_width/2.0));
		image_space_coords[i, j].y = (1.0/pr.pixel_size_y)*(i - (pr.screen_height/2.0));

		#print(image_space_coords[i, j]);

	return image_space_coords;

def calculate_projection_matrix(image_space_coords):
	# generate matrix same size as space coords
	global projection_matrix
	projection_matrix = np.full_like(image_space_coords, projection_point(), dtype=projection_point);
	global projection_center
	xpc = projection_center[0]; ypc = projection_center[1]; zpc = projection_center[2];

	# loop through all elements and calculate it's values
	for idxs, _ in np.ndenumerate(projection_matrix):
		# get indexes
		i = idxs[0]; j = idxs[1];
		projection_matrix[i, j].z1 = calculate_z_from_params(xpc, ypc, zpc);

def calculate_z_from_params(xpc, ypc, zpc):
	part_A = 0;
	part_B = 0;

	sqrt_part = math.sqrt(-2 * pr.cfs.c * pr.cfs.g * xpc + pr.cfs.d**2 * xpc**2 * part_B**2 + 2 * pr.cfs.j * pr.cfs.h * part_B - 2 * pr.cfs.e * part_B * pr.cfs.k + pr.cfs.j**2 - pr.cfs.c * pr.cfs.k + pr.cfs.f**2 * part_A**2 * zpc**2 + pr.cfs.e**2 * part_B**2 * zpc**2 + pr.cfs.d**2 * part_A**2 * ypc**2 - 2 * pr.cfs.j * pr.cfs.e * ypc + 2 * pr.cfs.j * pr.cfs.g * part_A + 2 * pr.cfs.j * pr.cfs.f * xpc - 2 * pr.cfs.f * part_A * pr.cfs.k - pr.cfs.b * part_B**2 * pr.cfs.k - 2 * pr.cfs.c * pr.cfs.h * ypc - pr.cfs.c * pr.cfs.a * xpc**2 - pr.cfs.c * pr.cfs.b * ypc**2 - pr.cfs.a * part_A**2 * pr.cfs.k + 2 * pr.cfs.d * xpc * part_B**2 * pr.cfs.h - 2 * pr.cfs.d * xpc * part_B * pr.cfs.f * part_A * zpc - 2 * pr.cfs.d * xpc * part_B * pr.cfs.e * ypc - 2 * pr.cfs.d * xpc * part_B * pr.cfs.g * part_A + 2 * pr.cfs.d * xpc * part_B**2 * pr.cfs.e * zpc + 2 * pr.cfs.d * xpc**2 * part_B * pr.cfs.f - 2 * pr.cfs.d**2 * xpc * part_B * part_A * ypc + 2 * pr.cfs.b * ypc * part_B * pr.cfs.a * xpc * part_A + 2 * pr.cfs.b * ypc * part_B * pr.cfs.f * part_A * zpc + 2 * pr.cfs.b * ypc * part_B * pr.cfs.g * part_A + 2 * pr.cfs.b * ypc * part_B * pr.cfs.f * xpc + pr.cfs.b * pr.cfs.h**2 * part_B**2 + pr.cfs.e**2 * ypc**2 + pr.cfs.g**2 * part_A**2 + pr.cfs.f**2 * xpc**2 - 2 * pr.cfs.a * part_A**2 * zpc * pr.cfs.e * ypc + 2 * pr.cfs.a * xpc * part_A * pr.cfs.h * part_B + 2 * pr.cfs.a * xpc * part_A * pr.cfs.e * ypc + 2 * pr.cfs.a * xpc * part_A * pr.cfs.e * part_B * zpc + 2 * pr.cfs.h * part_B * pr.cfs.f * part_A * zpc - 2 * pr.cfs.h * part_B * pr.cfs.e * ypc + 2 * pr.cfs.h * part_B * pr.cfs.g * part_A + 2 * pr.cfs.h * part_B**2 * pr.cfs.e * zpc + 2 * pr.cfs.h * part_B * pr.cfs.f * xpc - 2 * pr.cfs.h * part_B * pr.cfs.d * part_A * ypc - 2 * pr.cfs.f * part_A * zpc * pr.cfs.e * ypc + 2 * pr.cfs.f * part_A**2 * zpc * pr.cfs.g + 2 * pr.cfs.f * part_A * zpc**2 * pr.cfs.e * part_B - 2 * pr.cfs.f**2 * part_A * zpc * xpc + 2 * pr.cfs.f * part_A**2 * zpc * pr.cfs.d * ypc + 2 * pr.cfs.e * ypc * pr.cfs.g * part_A - 2 * pr.cfs.e**2 * ypc * part_B * zpc + 2 * pr.cfs.e * ypc * pr.cfs.f * xpc + 2 * pr.cfs.e * ypc**2 * pr.cfs.d * part_A - 2 * pr.cfs.e * ypc * pr.cfs.d * part_A* part_B *zpc + 2 * pr.cfs.g * part_A * pr.cfs.e * part_B * zpc - 2 * pr.cfs.g * part_A * pr.cfs.f * xpc + 2 * pr.cfs.g * part_A**2 * pr.cfs.d * ypc - 2 * pr.cfs.e * part_B * zpc * pr.cfs.f * xpc - 2 * pr.cfs.b * part_B**2 * zpc * pr.cfs.f * xpc - 2 * pr.cfs.f * xpc * pr.cfs.d * part_A * ypc + 2 * pr.cfs.d * xpc * part_B * pr.cfs.j + 2 * pr.cfs.b * ypc * part_B * pr.cfs.j - 2 * pr.cfs.j * pr.cfs.a * part_A**2 * zpc + 2 * pr.cfs.j * pr.cfs.a * xpc * part_A - 2 * pr.cfs.j * pr.cfs.f * part_A * zpc - 2 * pr.cfs.j * pr.cfs.e * part_B * zpc - 2 * pr.cfs.j * pr.cfs.b * part_B**2 * zpc + 2 * pr.cfs.j * pr.cfs.d * part_A * ypc - 4 * pr.cfs.j * pr.cfs.d * part_A * part_B * zpc - 4 * pr.cfs.f * part_A * pr.cfs.h * ypc - 2 * pr.cfs.f * part_A * pr.cfs.b * ypc**2 - 2 * pr.cfs.b * part_B**2 * pr.cfs.g * xpc - pr.cfs.b * part_B**2 * pr.cfs.a * xpc**2 - 4 * pr.cfs.e * part_B * pr.cfs.g * xpc - 2 * pr.cfs.e * part_B * pr.cfs.a * xpc**2 - 2 * pr.cfs.a * part_A**2 * pr.cfs.h * ypc - pr.cfs.a * part_A**2 * pr.cfs.b * ypc**2 - 2 * pr.cfs.c * pr.cfs.d * xpc * ypc - pr.cfs.c * pr.cfs.b * part_B**2 * zpc**2 - pr.cfs.c * pr.cfs.a * part_A**2 * zpc**2 + 2 * pr.cfs.c * pr.cfs.a * xpc * part_A * zpc + 2 * pr.cfs.c * pr.cfs.h * part_B* zpc - 2 * pr.cfs.c * pr.cfs.d * part_A * zpc**2 * part_B + 2 * pr.cfs.c * pr.cfs.b * ypc * part_B * zpc + 2 * pr.cfs.c * pr.cfs.d * xpc * part_B * zpc + 2 * pr.cfs.c * pr.cfs.g * part_A * zpc + 2 * pr.cfs.c * pr.cfs.d * part_A * zpc * ypc - 2 * pr.cfs.d * part_A * part_B * pr.cfs.k);

	z1 = 1/2 * (-2 * pr.cfs.d * xpc * part_B - 2 * pr.cfs.b * ypc * part_B - 2 * pr.cfs.j + 2 * pr.cfs.a * part_A**2 * zpc - 2 * pr.cfs.a * xpc * part_A - 2 * pr.cfs.h * part_B + 2 * pr.cfs.f * part_A * zpc - 2 * pr.cfs.e * ypc - 2 * pr.cfs.g * part_A + 2 * pr.cfs.e * part_B * zpc + 2 * pr.cfs.b * part_B**2 * zpc - 2 * pr.cfs.f * xpc - 2 * pr.cfs.d * part_A * ypc + 4 * pr.cfs.d * part_A * part_B * zpc + 2 * sqrt_part) / (2 * pr.cfs.f * part_A + pr.cfs.b * part_B**2 + pr.cfs.c + 2 * pr.cfs.e * part_B + 2 * pr.cfs.d * part_A * part_B + pr.cfs.a * part_A**2);

	z2 = 1/2 * (-2 * pr.cfs.d * xpc * part_B - 2 * pr.cfs.b * ypc * part_B - 2 * pr.cfs.j + 2 * pr.cfs.a * part_A**2 * zpc - 2 * pr.cfs.a * xpc * part_A - 2 * pr.cfs.h * part_B + 2 * pr.cfs.f * part_A * zpc - 2 * pr.cfs.e * ypc - 2 * pr.cfs.g * part_A + 2 * pr.cfs.e * part_B * zpc + 2 * pr.cfs.b * part_B**2 * zpc - 2 * pr.cfs.f * xpc - 2 * pr.cfs.d * part_A * ypc + 4 * pr.cfs.d * part_A * part_B * zpc - 2 * sqrt_part) / (2 * pr.cfs.f * part_A + pr.cfs.b * part_B**2 + pr.cfs.c + 2 * pr.cfs.e * part_B + 2 * pr.cfs.d * part_A * part_B + pr.cfs.a * part_A**2);

	#print(projection_center);
	return (0, 0);

class point2D:
	def __init__(self):
		self.x = 0.0;
		self.y = 0.0;

	def __str__(self):
		return '(' + str(self.x) + ', ' + str(self.y) + ')';

class point3D:
	def __init__(self):
		self.x = 0.0;
		self.y = 0.0;
		self.z = 0.0;

	def __str__(self):
		return '(' + str(self.x) + ', ' + str(self.y) + ', ' + str(self.z) + ')';

class projection_point:
	def __init__(self):
		self.x1 = 0.0;
		self.y1 = 0.0;
		self.z1 = 0.0;
		self.x2 = 0.0;
		self.y2 = 0.0;
		self.z2 = 0.0;

	def __str__(self):
		return '(' + str(self.x1) + ', ' + str(self.y1) + ', ' + str(self.z1) + ')/'
		+ '(' + str(self.x2) + ', ' + str(self.y2) + ', ' + str(self.z) + ')2';

if __name__ == '__main__':
	generate_image('parameters.json');