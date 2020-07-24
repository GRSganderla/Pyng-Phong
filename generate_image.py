import parameters as pr
import numpy as np
import math

image_matrix = None;
image_space_coords = None;
projection_matrix = None;

def generate_image(paramFile):
	# load
	parametersDict = pr.load_file(paramFile);
	#print(parametersDict);

	# initialize vars
	pr.initialize_vars(parametersDict);

	global image_matrix;
	image_matrix = np.zeros([pr.screen_width, pr.screen_height]);
	#print(image_matrix);

	global image_space_coords;
	image_space_coords = calculate_image_space_coord(image_matrix);
	#print(image_space_coords);

	global projection_matrix;
	projection_matrix = calculate_projection_matrix(image_space_coords);
	#print(projection_matrix);

	global image_result
	image_result = np.full_like(projection_matrix, point3D(), dtype=point3D)
	calculate_pixel_color(projection_matrix, image_result, pr.cfs)
	

def calculate_vet_normal(coef, image):

	n1 = 2 * coef.a * image.x + 2 * coef.d * image.y + 2 * coef.f * image.z + 2 * coef.g
	n2 = 2 * coef.b * image.y + 2 * coef.d * image.x + 2 * coef.e * image.z + 2 * coef.h
	n3 = 2 * coef.c * image.z + 2 * coef.e * image.y + 2 * coef.f * image.x + 2 * coef.j

	return np.array([n1, n2, n3])

def calculate_light_dir(image):

	l1 = pr.light[0] - image.x
	l2 = pr.light[1] - image.y
	l3 = pr.light[2] - image.z

	return np.array([l1, l2, l3])

def calculate_obser_dir(image):

	od1 = pr.position[0] - image.x
	od2 = pr.position[1] - image.y
	od3 = pr.position[2] - image.z

	return np.array([od1, od2, od3])

def calculate_vectors(coef, image):

	vet_n = calculate_vet_normal(coef, image)

	vet_lig = calculate_light_dir(image)

	vet_orien = calculate_obser_dir(image)

	vet_reflex = calculate_reflex_dir(vet_n, vet_lig)

	return (vet_n, vet_lig, vet_orien, vet_reflex)

def calculate_reflex_dir(N, L):
	w = 2 * (N[0]*L[0] + N[1]*L[1] + N[2]*L[2])
	R = w * N - L

	return R

def calculate_norm(vect):

	sum_xyz = vect[0] ** 2 + vect[1] ** 2 + vect[2] ** 2

	return math.sqrt(sum_xyz)

def calculate_cos(vect_1, vect_2):

	up_num = vect_1 * vect_2
	dw_num = calculate_norm(vect_n) * calculate_norm(vect_l)

	return (up_num/dw_num)

def calculate_pixel_color(projection, image, coef):

	xpc = pr.position[0]; ypc = pr.position[1]; zpc = pr.position[2];

	for idxs, _ in np.ndenumerate(projection_matrix):
		# get indexes
		i = idxs[0]
		j = idxs[1]

		dist1, dist2 = calculate_dist_pc(xpc, ypc, zpc, projection[i, j])

		if dist1 > dist2:
			image[i, j].x = projection[i, j].x2
			image[i, j].y = projection[i, j].y2
			image[i, j].z = projection[i, j].z2
		else:
			image[i, j].x = projection[i, j].x1
			image[i, j].y = projection[i, j].y1
			image[i, j].z = projection[i, j].z1

		vet_normal, vet_light_dir, vet_observ_dir, vet_reflex_dir = calculate_vectors(coef, image[i, j])

		cos_theta = calculate_cos(vet_normal, vet_light_dir)
		cos_alpha = calculate_cos(vet_reflex_dir, vet_observ_dir)
			


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
		image_space_coords[i, j].y = (1.0/pr.pixel_size_y)*(j - (pr.screen_height/2.0));

		#print(image_space_coords[i, j]);

	return image_space_coords;

def calculate_projection_matrix(image_space_coords):
	# generate matrix same size as space coords
	global projection_matrix
	projection_matrix = np.full_like(image_space_coords, projection_point(), dtype=projection_point);

	xpc = pr.position[0]; ypc = pr.position[1]; zpc = pr.position[2];

	# loop through all elements and calculate it's values
	for idxs, _ in np.ndenumerate(projection_matrix):
		# get indexes
		i = idxs[0]; j = idxs[1];

		r_xyz = calculate_matrix_r(pr.orientation[0], pr.orientation[1], pr.orientation[2])

		# calculate A and B values
		part_A = calculate_A(r_xyz, image_space_coords[i, j], pr.focal_distance);
		part_B = calculate_B(r_xyz, image_space_coords[i, j], pr.focal_distance);

		# calculate z values
		z_result = calculate_z_from_params(xpc, ypc, zpc, part_A, part_B);
		projection_matrix[i, j].z1 = z_result[0];
		projection_matrix[i, j].z2 = z_result[1];

		# calculate x value
		x_result = calculate_x_from_params(xpc, zpc, part_A, z_result);
		projection_matrix[i, j].x1 = x_result[0];
		projection_matrix[i, j].x2 = x_result[1];

		# calculate y value
		y_result = calculate_y_from_params(ypc, zpc, part_B, z_result);
		projection_matrix[i, j].y1 = y_result[0];
		projection_matrix[i, j].y2 = y_result[1];


def calculate_dist_pc(xpc, ypc, zpc, new_coord):
	
	pit1 = ((xpc - new_coord.x1)**2) + ((ypc - new_coord.y1)**2) + ((zpc - new_coord.z1)**2)
	pit1 = ((xpc - new_coord.x2)**2) + ((ypc - new_coord.y2)**2) + ((zpc - new_coord.z2)**2)

	return (math.sqrt(pit1), math.sqrt(pit2))

def calculate_z_from_params(xpc, ypc, zpc, part_A, part_B):

	sqrt_part = math.sqrt(-2 * pr.cfs.c * pr.cfs.g * xpc + pr.cfs.d**2 * xpc**2 * part_B**2
		+ 2 * pr.cfs.j * pr.cfs.h * part_B - 2 * pr.cfs.e * part_B * pr.cfs.k + pr.cfs.j**2
		- pr.cfs.c * pr.cfs.k + pr.cfs.f**2 * part_A**2 * zpc**2 + pr.cfs.e**2 * part_B**2
		* zpc**2 + pr.cfs.d**2 * part_A**2 * ypc**2 - 2 * pr.cfs.j * pr.cfs.e * ypc + 2
		* pr.cfs.j * pr.cfs.g * part_A + 2 * pr.cfs.j * pr.cfs.f * xpc - 2 * pr.cfs.f * part_A
		* pr.cfs.k - pr.cfs.b * part_B**2 * pr.cfs.k - 2 * pr.cfs.c * pr.cfs.h * ypc - pr.cfs.c
		* pr.cfs.a * xpc**2 - pr.cfs.c * pr.cfs.b * ypc**2 - pr.cfs.a * part_A**2 * pr.cfs.k + 2
		* pr.cfs.d * xpc * part_B**2 * pr.cfs.h - 2 * pr.cfs.d * xpc * part_B * pr.cfs.f * part_A
		* zpc - 2 * pr.cfs.d * xpc * part_B * pr.cfs.e * ypc - 2 * pr.cfs.d * xpc * part_B
		* pr.cfs.g * part_A + 2 * pr.cfs.d * xpc * part_B**2 * pr.cfs.e * zpc + 2 * pr.cfs.d
		* xpc**2 * part_B * pr.cfs.f - 2 * pr.cfs.d**2 * xpc * part_B * part_A * ypc + 2
		* pr.cfs.b * ypc * part_B * pr.cfs.a * xpc * part_A + 2 * pr.cfs.b * ypc * part_B
		* pr.cfs.f * part_A * zpc + 2 * pr.cfs.b * ypc * part_B * pr.cfs.g * part_A + 2
		* pr.cfs.b * ypc * part_B * pr.cfs.f * xpc + pr.cfs.b * pr.cfs.h**2 * part_B**2
		+ pr.cfs.e**2 * ypc**2 + pr.cfs.g**2 * part_A**2 + pr.cfs.f**2 * xpc**2 - 2
		* pr.cfs.a * part_A**2 * zpc * pr.cfs.e * ypc + 2 * pr.cfs.a * xpc * part_A
		* pr.cfs.h * part_B + 2 * pr.cfs.a * xpc * part_A * pr.cfs.e * ypc + 2 * pr.cfs.a
		* xpc * part_A * pr.cfs.e * part_B * zpc + 2 * pr.cfs.h * part_B * pr.cfs.f
		* part_A * zpc - 2 * pr.cfs.h * part_B * pr.cfs.e * ypc + 2 * pr.cfs.h * part_B
		* pr.cfs.g * part_A + 2 * pr.cfs.h * part_B**2 * pr.cfs.e * zpc + 2 * pr.cfs.h
		* part_B * pr.cfs.f * xpc - 2 * pr.cfs.h * part_B * pr.cfs.d * part_A * ypc - 2
		* pr.cfs.f * part_A * zpc * pr.cfs.e * ypc + 2 * pr.cfs.f * part_A**2 * zpc
		* pr.cfs.g + 2 * pr.cfs.f * part_A * zpc**2 * pr.cfs.e * part_B - 2 * pr.cfs.f**2
		* part_A * zpc * xpc + 2 * pr.cfs.f * part_A**2 * zpc * pr.cfs.d * ypc + 2
		* pr.cfs.e * ypc * pr.cfs.g * part_A - 2 * pr.cfs.e**2 * ypc * part_B * zpc
		+ 2 * pr.cfs.e * ypc * pr.cfs.f * xpc + 2 * pr.cfs.e * ypc**2 * pr.cfs.d
		* part_A - 2 * pr.cfs.e * ypc * pr.cfs.d * part_A* part_B *zpc + 2 * pr.cfs.g
		* part_A * pr.cfs.e * part_B * zpc - 2 * pr.cfs.g * part_A * pr.cfs.f * xpc
		+ 2 * pr.cfs.g * part_A**2 * pr.cfs.d * ypc - 2 * pr.cfs.e * part_B * zpc
		* pr.cfs.f * xpc - 2 * pr.cfs.b * part_B**2 * zpc * pr.cfs.f * xpc - 2 * pr.cfs.f
		* xpc * pr.cfs.d * part_A * ypc + 2 * pr.cfs.d * xpc * part_B * pr.cfs.j + 2
		* pr.cfs.b * ypc * part_B * pr.cfs.j - 2 * pr.cfs.j * pr.cfs.a * part_A**2
		* zpc + 2 * pr.cfs.j * pr.cfs.a * xpc * part_A - 2 * pr.cfs.j * pr.cfs.f
		* part_A * zpc - 2 * pr.cfs.j * pr.cfs.e * part_B * zpc - 2 * pr.cfs.j
		* pr.cfs.b * part_B**2 * zpc + 2 * pr.cfs.j * pr.cfs.d * part_A * ypc - 4
		* pr.cfs.j * pr.cfs.d * part_A * part_B * zpc - 4 * pr.cfs.f * part_A * pr.cfs.h
		* ypc - 2 * pr.cfs.f * part_A * pr.cfs.b * ypc**2 - 2 * pr.cfs.b * part_B**2
		* pr.cfs.g * xpc - pr.cfs.b * part_B**2 * pr.cfs.a * xpc**2 - 4 * pr.cfs.e
		* part_B * pr.cfs.g * xpc - 2 * pr.cfs.e * part_B * pr.cfs.a * xpc**2 - 2
		* pr.cfs.a * part_A**2 * pr.cfs.h * ypc - pr.cfs.a * part_A**2 * pr.cfs.b
		* ypc**2 - 2 * pr.cfs.c * pr.cfs.d * xpc * ypc - pr.cfs.c * pr.cfs.b * part_B**2
		* zpc**2 - pr.cfs.c * pr.cfs.a * part_A**2 * zpc**2 + 2 * pr.cfs.c * pr.cfs.a
		* xpc * part_A * zpc + 2 * pr.cfs.c * pr.cfs.h * part_B* zpc - 2 * pr.cfs.c
		* pr.cfs.d * part_A * zpc**2 * part_B + 2 * pr.cfs.c * pr.cfs.b * ypc * part_B
		* zpc + 2 * pr.cfs.c * pr.cfs.d * xpc * part_B * zpc + 2 * pr.cfs.c * pr.cfs.g
		* part_A * zpc + 2 * pr.cfs.c * pr.cfs.d * part_A * zpc * ypc - 2 * pr.cfs.d
		* part_A * part_B * pr.cfs.k);

	z1 = 1/2 * (-2 * pr.cfs.d * xpc * part_B - 2 * pr.cfs.b * ypc * part_B - 2 * pr.cfs.j
		+ 2 * pr.cfs.a * part_A**2 * zpc - 2 * pr.cfs.a * xpc * part_A - 2 * pr.cfs.h * part_B
		+ 2 * pr.cfs.f * part_A * zpc - 2 * pr.cfs.e * ypc - 2 * pr.cfs.g * part_A + 2
		* pr.cfs.e * part_B * zpc + 2 * pr.cfs.b * part_B**2 * zpc - 2 * pr.cfs.f * xpc
		- 2 * pr.cfs.d * part_A * ypc + 4 * pr.cfs.d * part_A * part_B * zpc + 2
		* sqrt_part) / (2 * pr.cfs.f * part_A + pr.cfs.b * part_B**2 + pr.cfs.c + 2
		* pr.cfs.e * part_B + 2 * pr.cfs.d * part_A * part_B + pr.cfs.a * part_A**2);

	z2 = 1/2 * (-2 * pr.cfs.d * xpc * part_B - 2 * pr.cfs.b * ypc * part_B - 2 * pr.cfs.j
		+ 2 * pr.cfs.a * part_A**2 * zpc - 2 * pr.cfs.a * xpc * part_A - 2 * pr.cfs.h * part_B
		+ 2 * pr.cfs.f * part_A * zpc - 2 * pr.cfs.e * ypc - 2 * pr.cfs.g * part_A + 2
		* pr.cfs.e * part_B * zpc + 2 * pr.cfs.b * part_B**2 * zpc - 2 * pr.cfs.f * xpc
		- 2 * pr.cfs.d * part_A * ypc + 4 * pr.cfs.d * part_A * part_B * zpc - 2
		* sqrt_part) / (2 * pr.cfs.f * part_A + pr.cfs.b * part_B**2 + pr.cfs.c + 2
		* pr.cfs.e * part_B + 2 * pr.cfs.d * part_A * part_B + pr.cfs.a * part_A**2);

	#print((z1, z2));
	return (z1, z2);

def calculate_x_from_params(xpc, zpc, part_A, z_coord):
	x1 = xpc + (z_coord[0] - zpc) * part_A;
	x2 = xpc + (z_coord[1] - zpc) * part_A;

	#print((x1, x2));
	return (x1, x2);

def calculate_y_from_params(ypc, zpc, part_B, z_coord):
	y1 = ypc + (z_coord[0] - zpc) * part_B;
	y2 = ypc + (z_coord[1] - zpc) * part_B;

	#print((y1, y2));
	return (y1, y2);

# create a 3x3 matriz of X,Y,Z orientation angles
def calculate_matrix_r(omega, theta, kappa):

	r11 = math.cos(theta) * math.cos(kappa)
	r12 = math.sen(omega) * math.sen(theta) * math.cos(kappa) - math.cos(omega) * math.sen(kappa)
	r13 = math.sen(omega) * math.sen(kappa) + math.cos(omega) * math.sen(theta) * math.cos(kappa)
	r21 = math.cos(theta) * math.sen(kappa)
	r22 = math.cos(omega) * math.cos(kappa) + math.sen(omega) * math.sen(theta) * math.sen(kappa)
	r23 = math.cos(omega) * math.sen(theta) * math.sen(kappa) - math.sen(omega) * math.cos(kappa)
	r31 = -math.sen(theta)
	r32 = math.sen(omega) * math.cos(theta)
	r33 = math.cos(omega) * math.cos(theta)

	return np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])

def calculate_A(R, coord, f):
	side_up = R[1, 1] * coord.x + R[2, 1] * coord.y + R[3, 1] * f
	side_dw = R[1, 3] * coord.x + R[2, 3] * coord.y + R[3, 3] * f

	return (side_up / side_dw)

def calculate_B(R, coord, f):
	side_up = R[1, 2] * coord.x + R[2, 2] * coord.y + R[3, 2] * f
	side_dw = R[1, 3] * coord.x + R[2, 3] * coord.y + R[3, 3] * f

	return (side_up / side_dw)

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
		+ '(' + str(self.x2) + ', ' + str(self.y2) + ', ' + str(self.z2) + ')2';

if __name__ == '__main__':
	generate_image('sphere.json');