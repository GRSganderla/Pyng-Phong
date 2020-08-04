import parameters as pr
import numpy as np
import math
import matplotlib.pyplot as plt

image_matrix = None
image_space_coords = None
projection_matrix = None
raw_image = None
result_image = None
omega = theta = kappa = None

def generate_image(paramFile):
	# load
	parametersDict = pr.load_file(paramFile)
	#print(parametersDict)

	# initialize vars
	print('Initialize Vars')
	pr.initialize_vars(parametersDict)

	global omega, theta, kappa
	omega, theta, kappa = get_angles_from_orientations(pr.orientation)

	global image_matrix
	print('Image matrix')
	image_matrix = np.zeros((pr.screen_height, pr.screen_width))
	#print(image_matrix)

	xpc, ypc, zpc = pr.position
	for idxs, _ in np.ndenumerate(image_matrix):
		# get indexes
		i = idxs[0] # y axis
		j = idxs[1] # x axis

		# get image to space x and y transformation
		space_coord = np.array([0.0, 0.0])
		space_coord[0] = (1.0/pr.pixel_size_x)*(i - (pr.screen_width/2.0))
		space_coord[1] = (1.0/pr.pixel_size_y)*(j - (pr.screen_height/2.0))
		#print(space_coord)

		# P1 and P2
		projection_point = np.zeros((2, 3))

		r_xyz = calculate_matrix_r(omega, theta, kappa)

		#print(r_xyz)

		# calculate A and B values
		value_A = calculate_A(r_xyz, space_coord, pr.focal_distance)
		value_B = calculate_B(r_xyz, space_coord, pr.focal_distance)
		#print((value_A, value_B))

		# calculate z values
		z_result = calculate_z_from_params(xpc, ypc, zpc, value_A, value_B)
		projection_point[0, 2] = z_result[0]
		projection_point[1, 2] = z_result[1]

		#print((z_result[0], z_result[1]))

		# calculate x value
		x_result = calculate_x_from_params(xpc, zpc, value_A, z_result)
		projection_point[0, 0] = x_result[0]
		projection_point[1, 0] = x_result[1]

		# calculate y value
		y_result = calculate_y_from_params(ypc, zpc, value_B, z_result)
		projection_point[0, 1] = y_result[0]
		projection_point[1, 1] = y_result[1]

		#print((projection_point[0], projection_point[1]))

		pixel_coord = np.zeros((3))

		dist1, dist2 = calculate_dist_pc(projection_point)
		#print((dist1, dist2))

		# pixel_coord[0] = space_coord[0]
		# pixel_coord[1] = space_coord[1]
		if dist1 < dist2:
			pixel_coord = projection_point[0]
		else:
			pixel_coord = projection_point[1]

		#print(pixel_coord)

		# vet_normal = calculate_vet_normal(pixel_coord)
		#print(vet_normal)

		# vet_light_dir = calculate_light_dir(pixel_coord)
		#print(vet_light_dir)

		#vet_observ_dir = calculate_obser_dir(pixel_coord)

		#vet_reflex_dir = calculate_reflex_dir(vet_normal, vet_light_dir)

		#cos_theta = calculate_cos(vet_normal, vet_light_dir)
		#cos_alpha = calculate_cos(vet_reflex_dir, vet_observ_dir)

		#print((cos_theta, cos_alpha));

		# image_matrix[i, j] = calculate_pixel_intensity(cos_theta, cos_alpha=0)
		image_matrix[i, j] = pixel_coord[2]


	#coords = np.array([point.as_array() for point in raw_image.flatten()])

	#fig = plt.figure()
	#ax = fig.add_subplot(111, projection='3d')
	#points = np.array([point.z for point in raw_image.flatten()])
	# ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2])
	# ax.set_xlabel('X Label')
	# ax.set_ylabel('Y Label')
	# ax.set_zlabel('Z Label')
	# plt.show()
	
	plt.imshow(image_matrix, cmap='Greys') # anatomia
	plt.show()

	# coords = np.array([point.as_array() for point in image_space_coords.flatten()])

	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')
	# ax.scatter(coords[:, 0], coords[:, 1], image_matrix[:])
	# ax.set_xlabel('X Label')
	# ax.set_ylabel('Y Label')
	# ax.set_zlabel('Z Label')
	# plt.show()


def calculate_dist_pc(new_coord):
	return (np.linalg.norm(pr.position - new_coord[0]), np.linalg.norm(pr.position - new_coord[1]))

def calculate_z_from_params(xpc, ypc, zpc, value_A, value_B):
	part_1 = -2 * pr.cfs.d * xpc * value_B - 2 * pr.cfs.b * ypc * value_B - 2 * pr.cfs.j + 2 * pr.cfs.a * value_A**2 * zpc - 2 * pr.cfs.a * xpc * value_A - 2 * pr.cfs.h * value_B + 2 * pr.cfs.f * value_A * zpc - 2 * pr.cfs.e * ypc - 2 * pr.cfs.g * value_A + 2 * pr.cfs.e * value_B * zpc + 2 * pr.cfs.b * value_B**2 * zpc - 2 * pr.cfs.f * xpc - 2 * pr.cfs.d * value_A * ypc + 4 * pr.cfs.d * value_A * value_B * zpc

	sqrt_part = math.sqrt(-2 * pr.cfs.c * pr.cfs.g * xpc + pr.cfs.d**2 * xpc**2 * value_B**2 + 2 * pr.cfs.j * pr.cfs.h * value_B - 2 * pr.cfs.e * value_B * pr.cfs.k + pr.cfs.j**2 - pr.cfs.c * pr.cfs.k + pr.cfs.f**2 * value_A**2 * zpc**2 + pr.cfs.e**2 * value_B**2 * zpc**2 + pr.cfs.d**2 * value_A**2 * ypc**2 + 2 * pr.cfs.j * pr.cfs.e * ypc + 2 * pr.cfs.j * pr.cfs.g * value_A + 2 * pr.cfs.j * pr.cfs.f * xpc - 2 * pr.cfs.f * value_A * pr.cfs.k - pr.cfs.b * value_B**2 * pr.cfs.k - 2 * pr.cfs.c * pr.cfs.h * ypc - pr.cfs.c * pr.cfs.a * xpc**2 - pr.cfs.c * pr.cfs.b * ypc**2 - pr.cfs.a * value_A**2 * pr.cfs.k + 2 * pr.cfs.d * xpc * value_B**2 * pr.cfs.h - 2 * pr.cfs.d * xpc * value_B * pr.cfs.f * value_A * zpc - 2 * pr.cfs.d * xpc * value_B * pr.cfs.e * ypc - 2 * pr.cfs.d * xpc * value_B * pr.cfs.g * value_A + 2 * pr.cfs.d * xpc * value_B**2 * pr.cfs.e * zpc + 2 * pr.cfs.d * xpc**2 * value_B * pr.cfs.f - 2 * pr.cfs.d**2 * xpc * value_B * value_A * ypc + 2 * pr.cfs.b * ypc * value_B * pr.cfs.a * xpc * value_A + 2 * pr.cfs.b * ypc * value_B * pr.cfs.f * value_A * zpc + 2 * pr.cfs.b * ypc * value_B * pr.cfs.g * value_A + 2 * pr.cfs.b * ypc * value_B * pr.cfs.f * xpc + pr.cfs.h**2 * value_B**2 + pr.cfs.e**2 * ypc**2 + pr.cfs.g**2 * value_A**2 + pr.cfs.f**2 * xpc**2 - 2 * pr.cfs.a * value_A**2 * zpc * pr.cfs.e * ypc + 2 * pr.cfs.a * xpc * value_A * pr.cfs.h * value_B + 2 * pr.cfs.a * xpc * value_A * pr.cfs.e * ypc + 2 * pr.cfs.a * xpc * value_A * pr.cfs.e * value_B * zpc + 2 * pr.cfs.h * value_B * pr.cfs.f * value_A * zpc - 2 * pr.cfs.h * value_B * pr.cfs.e * ypc + 2 * pr.cfs.h * value_B * pr.cfs.g * value_A + 2 * pr.cfs.h * value_B**2 * pr.cfs.e * zpc + 2 * pr.cfs.h * value_B * pr.cfs.f * xpc - 2 * pr.cfs.h * value_B * pr.cfs.d * value_A * ypc - 2 * pr.cfs.f * value_A * zpc * pr.cfs.e * ypc + 2 * pr.cfs.f * value_A**2 * zpc * pr.cfs.g + 2 * pr.cfs.f * value_A * zpc**2 * pr.cfs.e * value_B - 2 * pr.cfs.f**2 * value_A * zpc * xpc + 2 * pr.cfs.f * value_A**2 * zpc * pr.cfs.d * ypc + 2 * pr.cfs.e * ypc * pr.cfs.g * value_A - 2 * pr.cfs.e**2 * ypc * value_B * zpc + 2 * pr.cfs.e * ypc * pr.cfs.f * xpc + 2 * pr.cfs.e * ypc**2 * pr.cfs.d * value_A - 2 * pr.cfs.e * ypc * pr.cfs.d * value_A * value_B * zpc + 2 * pr.cfs.g * value_A * pr.cfs.e * value_B * zpc - 2 * pr.cfs.g * value_A * pr.cfs.f * xpc + 2 * pr.cfs.g * value_A**2 * pr.cfs.d * ypc - 2 * pr.cfs.e * value_B * zpc * pr.cfs.f * xpc - 2 * pr.cfs.b * value_B**2 * zpc * pr.cfs.f * xpc - 2 * pr.cfs.f * xpc * pr.cfs.d * value_A * ypc + 2 * pr.cfs.d * xpc * value_B * pr.cfs.j + 2 * pr.cfs.b * ypc * value_B * pr.cfs.j - 2 * pr.cfs.j * pr.cfs.a * value_A**2 * zpc + 2 * pr.cfs.j * pr.cfs.a * xpc * value_A - 2 * pr.cfs.j * pr.cfs.f * value_A * zpc - 2 * pr.cfs.j * pr.cfs.e * value_B * zpc - 2 * pr.cfs.j * pr.cfs.b * value_B**2 * zpc + 2 * pr.cfs.j * pr.cfs.d * value_A * ypc - 4 * pr.cfs.j * pr.cfs.d * value_A * value_B * zpc - 4 * pr.cfs.f * value_A * pr.cfs.h * ypc - 2 * pr.cfs.f * value_A * pr.cfs.b * ypc**2 - 2 * pr.cfs.b * value_B**2 * pr.cfs.g * xpc - pr.cfs.b * value_B**2 * pr.cfs.a * xpc**2 - 4 * pr.cfs.e * value_B * pr.cfs.g * xpc - 2 * pr.cfs.e * value_B * pr.cfs.a * xpc**2 - 2 * pr.cfs.a * value_A**2 * pr.cfs.h * ypc - pr.cfs.a * value_A**2 * pr.cfs.b * ypc**2 - 2 * pr.cfs.c * pr.cfs.d * xpc * ypc - pr.cfs.c * pr.cfs.b * value_B**2 * zpc**2 - pr.cfs.c * pr.cfs.a * value_A**2 * zpc**2 + 2 * pr.cfs.c * pr.cfs.a * xpc * value_A * zpc + 2 * pr.cfs.c * pr.cfs.h * value_B * zpc - 2 * pr.cfs.c * pr.cfs.d * value_A * zpc**2 * value_B + 2 * pr.cfs.c * pr.cfs.b * ypc * value_B * zpc + 2 * pr.cfs.c * pr.cfs.d * xpc * value_B * zpc + 2 * pr.cfs.c * pr.cfs.g * value_A * zpc + 2 * pr.cfs.c * pr.cfs.d * value_A * zpc * ypc - 2 * pr.cfs.d * value_A * value_B * pr.cfs.k)

	div_part = (2 * pr.cfs.f * value_A + pr.cfs.b * value_B**2 + pr.cfs.c + 2 * pr.cfs.e * value_B + 2 * pr.cfs.d * value_A * value_B + pr.cfs.a * value_A**2)

	z1 = 1/2 * (part_1 + 2 * sqrt_part) / div_part

	z2 = 1/2 * (part_1 - 2 * sqrt_part) / div_part

	#print(sqrt_part)
	#print((z1, z2))
	return (z1, z2)

def calculate_x_from_params(xpc, zpc, value_A, z_coord):
	x1 = xpc + (z_coord[0] - zpc) * value_A
	x2 = xpc + (z_coord[1] - zpc) * value_A

	#print((x1, x2))
	return (x1, x2)

def calculate_y_from_params(ypc, zpc, value_B, z_coord):
	y1 = ypc + (z_coord[0] - zpc) * value_B
	y2 = ypc + (z_coord[1] - zpc) * value_B

	#print((y1, y2))
	return (y1, y2)

# create a 3x3 matriz of X,Y,Z orientation angles
def calculate_matrix_r(omega, theta, kappa):
	#print((omega, theta, kappa))

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

def get_angles_from_orientations(orientation):
	omega = np.array([1.0, 0.0, 0.0])
	theta = np.array([0.0, 1.0, 0.0])
	kappa = np.array([0.0, 0.0, 1.0])

	omega = get_angle(omega, orientation)
	theta = get_angle(theta, orientation)
	kappa = get_angle(kappa, orientation)

	return (omega, theta, kappa)

def get_angle(vect_1, vect_2):
	return np.arccos(np.clip(np.dot(vect_1, vect_2), -1.0, 1.0))

def calculate_A(R, coord, f):
	side_up = R[0, 0] * coord[0] + R[1, 0] * coord[1] + R[2, 0] * f
	side_dw = R[0, 2] * coord[0] + R[1, 2] * coord[1] + R[2, 2] * f

	return (side_up / side_dw) if side_dw != 0 else 10

def calculate_B(R, coord, f):
	side_up = R[0, 1] * coord[0] + R[1, 1] * coord[1] + R[2, 1] * f
	side_dw = R[0, 2] * coord[0] + R[1, 2] * coord[1] + R[2, 2] * f

	return (side_up / side_dw) if side_dw != 0 else 10

def calculate_pixel_intensity(cos_theta, cos_alpha):
	return 1*pr.cfb.ambient + 1*(pr.cfb.difuse * cos_theta + pr.cfb.specular * cos_alpha**pr.cfb.n)

def print_matrix(matrix):
	for i in range(matrix.shape[0]):
		for j in range(matrix.shape[1]):
			print(matrix[i, j], end='')
	print('\n')


def calculate_vet_normal(pixel_coord):
	n1 = 2 * pr.cfs.a * pixel_coord[0] + 2 * pr.cfs.d * pixel_coord[1] + 2 * pr.cfs.f * pixel_coord[2] + 2 * pr.cfs.g
	n2 = 2 * pr.cfs.b * pixel_coord[1] + 2 * pr.cfs.d * pixel_coord[0] + 2 * pr.cfs.e * pixel_coord[2] + 2 * pr.cfs.h
	n3 = 2 * pr.cfs.c * pixel_coord[2] + 2 * pr.cfs.e * pixel_coord[1] + 2 * pr.cfs.f * pixel_coord[2] + 2 * pr.cfs.j

	return np.array([n1, n2, n3])

def calculate_light_dir(pixel_coord):
	return pr.light - pixel_coord

def calculate_obser_dir(pixel_coord):
	return pr.position - pixel_coord

def calculate_reflex_dir(N, L):
	w = 2 * (N.dot(L))
	R = w * N - L

	return R

def calculate_cos(vect_1, vect_2):
	dw_num = np.linalg.norm(vect_1) * np.linalg.norm(vect_2)

	return (vect_1.dot(vect_2) / dw_num) if dw_num != 0 else 0

class point2D:
	def __init__(self):
		self.x = 0.0
		self.y = 0.0

	def __str__(self):
		return '(' + str(self.x) + ', ' + str(self.y) + ')'

	def as_array(self):
		return np.array([self.x, self.y])

class point3D:
	def __init__(self):
		self.x = 0.0
		self.y = 0.0
		self.z = 0.0

	def __str__(self):
		return '(' + str(self.x) + ', ' + str(self.y) + ', ' + str(self.z) + ')'

	def as_array(self):
		return np.array([self.x, self.y, self.z])

class projection_point:
	def __init__(self):
		self.x1 = 0.0
		self.y1 = 0.0
		self.z1 = 0.0
		self.x2 = 0.0
		self.y2 = 0.0
		self.z2 = 0.0

	def __str__(self):
		return '(' + str(self.x1) + ', ' + str(self.y1) + ', ' + str(self.z1) + ')/' + '(' + str(self.x2) + ', ' + str(self.y2) + ', ' + str(self.z2) + ')2'

	def as_array1(self):
		return np.array([self.x1, self.y1, self.z1])

	def as_array2(self):
		return np.array([self.x2, self.y2, self.z2])

if __name__ == '__main__':
	generate_image('sphere.json')