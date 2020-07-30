import parameters as pr
import numpy as np
import math
import matplotlib.pyplot as plt

image_matrix = None;
image_space_coords = None;
projection_matrix = None;
raw_image = None;
result_image = None;

def generate_image(paramFile):
	# load
	parametersDict = pr.load_file(paramFile);
	#print(parametersDict);

	# initialize vars
	print('Initialize Vars');
	pr.initialize_vars(parametersDict);

	global image_matrix;
	print('Image matrix');
	image_matrix = np.zeros([pr.screen_height, pr.screen_width]);
	#print(image_matrix);

	global image_space_coords;
	print('Image space coords')
	image_space_coords = calculate_image_space_coord(image_matrix);
	#print(image_space_coords);
	#print_matrix(image_space_coords);

	global projection_matrix;
	print('Projection matrix')
	projection_matrix = calculate_projection_matrix(image_space_coords);
	#print(projection_matrix);
	#print_matrix(projection_matrix);

	global raw_image
	print('Raw image');
	raw_image = calculate_pixel_color(projection_matrix, raw_image, pr.cfs);

	print('Result image');
	global result_image
	result_image = calculate_image_intensity(raw_image, result_image);
	#print_matrix(result_image);

	#print(result_image);

	plt.imshow(result_image, cmap='Greys'); # anatomia
	plt.show();

	coords = np.array([point.as_array() for point in image_space_coords.flatten()]);

	fig = plt.figure();
	ax = fig.add_subplot(111, projection='3d');
	points = np.array([point for point in result_image.flatten()]);
	#print(points.shape);
	ax.scatter(coords[:, 0], coords[:, 1], points[:]);
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	plt.show();

	

def calculate_vet_normal(coef, pixel_coord):

	n1 = 2 * coef.a * pixel_coord.x + 2 * coef.d * pixel_coord.y + 2 * coef.f * pixel_coord.z + 2 * coef.g
	n2 = 2 * coef.b * pixel_coord.y + 2 * coef.d * pixel_coord.x + 2 * coef.e * pixel_coord.z + 2 * coef.h
	n3 = 2 * coef.c * pixel_coord.z + 2 * coef.e * pixel_coord.y + 2 * coef.f * pixel_coord.z + 2 * coef.j

	return np.array([n1, n2, n3])

def calculate_light_dir(pixel_coord):

	l1 = pr.light[0] - pixel_coord.x
	l2 = pr.light[1] - pixel_coord.y
	l3 = pr.light[2] - pixel_coord.z

	#return np.array([l1, l2, l3])
	return pr.light - pixel_coord.as_array();

def calculate_obser_dir(pixel_coord):

	od1 = pr.position[0] - pixel_coord.x
	od2 = pr.position[1] - pixel_coord.y
	od3 = pr.position[2] - pixel_coord.z

	#return np.array([od1, od2, od3])
	return pr.position - pixel_coord.as_array();

def calculate_vectors(coef, image):

	vet_n = calculate_vet_normal(coef, image)

	vet_lig = calculate_light_dir(image)

	vet_orien = calculate_obser_dir(image)

	vet_reflex = calculate_reflex_dir(vet_n, vet_lig)

	return (vet_n, vet_lig, vet_orien, vet_reflex)

def calculate_reflex_dir(N, L):
	w = 2 * (N.dot(L))
	R = w * N - L

	return R

def calculate_norm(vect):
	return np.linalg.norm(vect);

def calculate_cos(vect_1, vect_2):

	dw_num = calculate_norm(vect_1) * calculate_norm(vect_2)

	return 255 if dw_num == 0 else (vect_1.dot(vect_2) / dw_num)
	

def calculate_pixel_color(projection, image, coef):
	image = np.full_like(projection_matrix, None, dtype=point3D)

	print(projection_matrix.shape)

	xpc = pr.position[0]; ypc = pr.position[1]; zpc = pr.position[2];

	for idxs, _ in np.ndenumerate(image):
		# get indexes
		i = idxs[0]
		j = idxs[1]

		image[i, j] = point3D();

		dist1, dist2 = calculate_dist_pc(xpc, ypc, zpc, projection[i, j])

		if dist2 < dist1:
			image[i, j].x = projection[i, j].x2
			image[i, j].y = projection[i, j].y2
			image[i, j].z = projection[i, j].z2
		else:
			image[i, j].x = projection[i, j].x1
			image[i, j].y = projection[i, j].y1
			image[i, j].z = projection[i, j].z1

		vet_normal, vet_light_dir, vet_observ_dir, vet_reflex_dir = calculate_vectors(coef, image[i, j])

		image[i, j].cos_theta = calculate_cos(vet_normal, vet_light_dir)
		image[i, j].cos_alpha = calculate_cos(vet_reflex_dir, vet_observ_dir)

		#print(image[i, j].cos_theta);
		#print(vet_normal);

	return image;



def calculate_image_space_coord(image_matrix):
	# generate an array same size as image matrix
	global image_space_coords;
	image_space_coords = np.full_like(image_matrix, None, dtype=point2D);

	# loop through all elements and calculate it's space value
	for idxs, _ in np.ndenumerate(image_space_coords):
		#print(idxs);
		# get indexes
		i = idxs[0]; j = idxs[1];
		# calculate x and y
		image_space_coords[i, j] = point2D();
		image_space_coords[i, j].x = (1.0/pr.pixel_size_x)*(i - (pr.screen_height/2.0));
		image_space_coords[i, j].y = (1.0/pr.pixel_size_y)*(j - (pr.screen_width/2.0));

		#print(image_space_coords[i, j]);

	return image_space_coords;

def calculate_projection_matrix(image_space_coords):
	# generate matrix same size as space coords
	global projection_matrix
	projection_matrix = np.full_like(image_space_coords, None, dtype=projection_point);

	xpc = pr.position[0]; ypc = pr.position[1]; zpc = pr.position[2];

	# loop through all elements and calculate it's values
	for idxs, _ in np.ndenumerate(projection_matrix):
		# get indexes
		i = idxs[0]; j = idxs[1];

		projection_matrix[i, j] = projection_point();

		r_xyz = calculate_matrix_r(pr.orientation)

		#print(r_xyz);

		# calculate A and B values
		part_A = calculate_A(r_xyz, image_space_coords[i, j], pr.focal_distance);
		part_B = calculate_B(r_xyz, image_space_coords[i, j], pr.focal_distance);

		#print(part_A);

		# calculate z values
		z_result = calculate_z_from_params(xpc, ypc, zpc, part_A, part_B);
		projection_matrix[i, j].z1 = z_result[0];
		projection_matrix[i, j].z2 = z_result[1];

		#print(z_result[1]);

		# calculate x value
		x_result = calculate_x_from_params(xpc, zpc, part_A, z_result);
		projection_matrix[i, j].x1 = x_result[0];
		projection_matrix[i, j].x2 = x_result[1];

		# calculate y value
		y_result = calculate_y_from_params(ypc, zpc, part_B, z_result);
		projection_matrix[i, j].y1 = y_result[0];
		projection_matrix[i, j].y2 = y_result[1];
	
	return projection_matrix;


def calculate_dist_pc(xpc, ypc, zpc, new_coord):
	
	pit1 = ((xpc - new_coord.x1)**2) + ((ypc - new_coord.y1)**2) + ((zpc - new_coord.z1)**2)
	pit2 = ((xpc - new_coord.x2)**2) + ((ypc - new_coord.y2)**2) + ((zpc - new_coord.z2)**2)

	return (math.sqrt(pit1), math.sqrt(pit2))

def calculate_z_from_params(xpc, ypc, zpc, part_A, part_B):

	part_1 = -2 * pr.cfs.d * xpc * part_B - 2 * pr.cfs.b * ypc * part_B - 2 * pr.cfs.j + 2 * pr.cfs.a * part_A**2 * zpc - 2 * pr.cfs.a * xpc * part_A - 2 * pr.cfs.h * part_B + 2 * pr.cfs.f * part_A * zpc - 2 * pr.cfs.e * ypc - 2 * pr.cfs.g * part_A + 2 * pr.cfs.e * part_B * zpc + 2 * pr.cfs.b * part_B**2 * zpc - 2 * pr.cfs.f * xpc - 2 * pr.cfs.d * part_A * ypc + 4 * pr.cfs.d * part_A * part_B * zpc;

	sqrt_part = math.sqrt(-2 * pr.cfs.c * pr.cfs.g * xpc + pr.cfs.d**2 * xpc**2 * part_B**2 + 2 * pr.cfs.j * pr.cfs.h * part_B - 2 * pr.cfs.e * part_B * pr.cfs.k + pr.cfs.j**2 - pr.cfs.c * pr.cfs.k + pr.cfs.f**2 * part_A**2 * zpc**2 + pr.cfs.e**2 * part_B**2 * zpc**2 + pr.cfs.d**2 * part_A**2 * ypc**2 + 2 * pr.cfs.j * pr.cfs.e * ypc + 2 * pr.cfs.j * pr.cfs.g * part_A + 2 * pr.cfs.j * pr.cfs.f * xpc - 2 * pr.cfs.f * part_A * pr.cfs.k - pr.cfs.b * part_B**2 * pr.cfs.k - 2 * pr.cfs.c * pr.cfs.h * ypc - pr.cfs.c * pr.cfs.a * xpc**2 - pr.cfs.c * pr.cfs.b * ypc**2 - pr.cfs.a * part_A**2 * pr.cfs.k + 2 * pr.cfs.d * xpc * part_B**2 * pr.cfs.h - 2 * pr.cfs.d * xpc * part_B * pr.cfs.f * part_A * zpc - 2 * pr.cfs.d * xpc * part_B * pr.cfs.e * ypc - 2 * pr.cfs.d * xpc * part_B * pr.cfs.g * part_A + 2 * pr.cfs.d * xpc * part_B**2 * pr.cfs.e * zpc + 2 * pr.cfs.d * xpc**2 * part_B * pr.cfs.f - 2 * pr.cfs.d**2 * xpc * part_B * part_A * ypc + 2 * pr.cfs.b * ypc * part_B * pr.cfs.a * xpc * part_A + 2 * pr.cfs.b * ypc * part_B * pr.cfs.f * part_A * zpc + 2 * pr.cfs.b * ypc * part_B * pr.cfs.g * part_A + 2 * pr.cfs.b * ypc * part_B * pr.cfs.f * xpc + pr.cfs.h**2 * part_B**2 + pr.cfs.e**2 * ypc**2 + pr.cfs.g**2 * part_A**2 + pr.cfs.f**2 * xpc**2 - 2 * pr.cfs.a * part_A**2 * zpc * pr.cfs.e * ypc + 2 * pr.cfs.a * xpc * part_A * pr.cfs.h * part_B + 2 * pr.cfs.a * xpc * part_A * pr.cfs.e * ypc + 2 * pr.cfs.a * xpc * part_A * pr.cfs.e * part_B * zpc + 2 * pr.cfs.h * part_B * pr.cfs.f * part_A * zpc - 2 * pr.cfs.h * part_B * pr.cfs.e * ypc + 2 * pr.cfs.h * part_B * pr.cfs.g * part_A + 2 * pr.cfs.h * part_B**2 * pr.cfs.e * zpc + 2 * pr.cfs.h * part_B * pr.cfs.f * xpc - 2 * pr.cfs.h * part_B * pr.cfs.d * part_A * ypc - 2 * pr.cfs.f * part_A * zpc * pr.cfs.e * ypc + 2 * pr.cfs.f * part_A**2 * zpc * pr.cfs.g + 2 * pr.cfs.f * part_A * zpc**2 * pr.cfs.e * part_B - 2 * pr.cfs.f**2 * part_A * zpc * xpc + 2 * pr.cfs.f * part_A**2 * zpc * pr.cfs.d * ypc + 2 * pr.cfs.e * ypc * pr.cfs.g * part_A - 2 * pr.cfs.e**2 * ypc * part_B * zpc + 2 * pr.cfs.e * ypc * pr.cfs.f * xpc + 2 * pr.cfs.e * ypc**2 * pr.cfs.d * part_A - 2 * pr.cfs.e * ypc * pr.cfs.d * part_A * part_B * zpc + 2 * pr.cfs.g * part_A * pr.cfs.e * part_B * zpc - 2 * pr.cfs.g * part_A * pr.cfs.f * xpc + 2 * pr.cfs.g * part_A**2 * pr.cfs.d * ypc - 2 * pr.cfs.e * part_B * zpc * pr.cfs.f * xpc - 2 * pr.cfs.b * part_B**2 * zpc * pr.cfs.f * xpc - 2 * pr.cfs.f * xpc * pr.cfs.d * part_A * ypc + 2 * pr.cfs.d * xpc * part_B * pr.cfs.j + 2 * pr.cfs.b * ypc * part_B * pr.cfs.j - 2 * pr.cfs.j * pr.cfs.a * part_A**2 * zpc + 2 * pr.cfs.j * pr.cfs.a * xpc * part_A - 2 * pr.cfs.j * pr.cfs.f * part_A * zpc - 2 * pr.cfs.j * pr.cfs.e * part_B * zpc - 2 * pr.cfs.j * pr.cfs.b * part_B**2 * zpc + 2 * pr.cfs.j * pr.cfs.d * part_A * ypc - 4 * pr.cfs.j * pr.cfs.d * part_A * part_B * zpc - 4 * pr.cfs.f * part_A * pr.cfs.h * ypc - 2 * pr.cfs.f * part_A * pr.cfs.b * ypc**2 - 2 * pr.cfs.b * part_B**2 * pr.cfs.g * xpc - pr.cfs.b * part_B**2 * pr.cfs.a * xpc**2 - 4 * pr.cfs.e * part_B * pr.cfs.g * xpc - 2 * pr.cfs.e * part_B * pr.cfs.a * xpc**2 - 2 * pr.cfs.a * part_A**2 * pr.cfs.h * ypc - pr.cfs.a * part_A**2 * pr.cfs.b * ypc**2 - 2 * pr.cfs.c * pr.cfs.d * xpc * ypc - pr.cfs.c * pr.cfs.b * part_B**2 * zpc**2 - pr.cfs.c * pr.cfs.a * part_A**2 * zpc**2 + 2 * pr.cfs.c * pr.cfs.a * xpc * part_A * zpc + 2 * pr.cfs.c * pr.cfs.h * part_B * zpc - 2 * pr.cfs.c * pr.cfs.d * part_A * zpc**2 * part_B + 2 * pr.cfs.c * pr.cfs.b * ypc * part_B * zpc + 2 * pr.cfs.c * pr.cfs.d * xpc * part_B * zpc + 2 * pr.cfs.c * pr.cfs.g * part_A * zpc + 2 * pr.cfs.c * pr.cfs.d * part_A * zpc * ypc - 2 * pr.cfs.d * part_A * part_B * pr.cfs.k);

	div_part = (2 * pr.cfs.f * part_A + pr.cfs.b * part_B**2 + pr.cfs.c + 2 * pr.cfs.e * part_B + 2 * pr.cfs.d * part_A * part_B + pr.cfs.a * part_A**2)

	z1 = 1/2 * (part_1 + 2 * sqrt_part) / div_part;

	z2 = 1/2 * (part_1 - 2 * sqrt_part) / div_part;

	#print(sqrt_part);
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
def calculate_matrix_r(orientations):
	omega, theta, kappa = orientations;

	r11 = math.cos(theta) * math.cos(kappa)
	r12 = math.sin(omega) * math.sin(theta) * math.cos(kappa) - math.cos(omega) * math.sin(kappa)
	r13 = math.sin(omega) * math.sin(kappa) + math.cos(omega) * math.sin(theta) * math.cos(kappa)
	r21 = math.cos(theta) * math.sin(kappa)
	r22 = math.cos(omega) * math.cos(kappa) + math.sin(omega) * math.sin(theta) * math.sin(kappa)
	r23 = math.cos(omega) * math.sin(theta) * math.sin(kappa) - math.sin(omega) * math.cos(kappa)
	r31 = -math.sin(theta)
	r32 = math.sin(omega) * math.cos(theta)
	r33 = math.cos(omega) * math.cos(theta)

	return np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])

def calculate_A(R, coord, f):
	side_up = R[0, 0] * coord.x + R[1, 0] * coord.y + R[2, 0] * f
	side_dw = R[0, 2] * coord.x + R[1, 2] * coord.y + R[2, 2] * f

	return (side_up / side_dw)

def calculate_B(R, coord, f):
	side_up = R[0, 1] * coord.x + R[1, 1] * coord.y + R[2, 1] * f
	side_dw = R[0, 2] * coord.x + R[1, 2] * coord.y + R[2, 2] * f

	return (side_up / side_dw)

def calculate_image_intensity(raw_image, result_image):
	result_image = np.zeros_like(raw_image, dtype=float);

	for idxs, _ in np.ndenumerate(raw_image):
		i = idxs[0]; j = idxs[1];

		result_image[i, j] = calculate_pixel_intensity(raw_image[i, j]);
	
	return result_image;


def calculate_pixel_intensity(raw_pixel):
	return 0*pr.cfb.ambient + 1*(pr.cfb.difuse * raw_pixel.cos_theta + pr.cfb.specular * raw_pixel.cos_alpha**pr.cfb.n);

def print_matrix(matrix):
	for i in range(matrix.shape[0]):
		for j in range(matrix.shape[1]):
			print(matrix[i, j], end='');
	print('\n');

class point2D:
	def __init__(self):
		self.x = 0.0;
		self.y = 0.0;

	def __str__(self):
		return '(' + str(self.x) + ', ' + str(self.y) + ')';

	def as_array(self):
		return np.array([self.x, self.y]);

class point3D:
	def __init__(self):
		self.x = 0.0;
		self.y = 0.0;
		self.z = 0.0;

	def __str__(self):
		return '(' + str(self.x) + ', ' + str(self.y) + ', ' + str(self.z) + ')';

	def as_array(self):
		return np.array([self.x, self.y, self.z]);

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

	def as_array1(self):
		return np.array([self.x1, self.y1, self.z1]);

	def as_array2(self):
		return np.array([self.x2, self.y2, self.z2]);

if __name__ == '__main__':
	generate_image('sphere.json');