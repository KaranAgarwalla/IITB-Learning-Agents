import argparse

# argparse utilities
def check_positive(value):
	try:
		value = int(value)
	except ValueError:
		raise argparse.ArgumentTypeError("{} is an invalid non-negative integer value".format(value))
	if value < 0:
		raise argparse.ArgumentTypeError("{} is an invalid non-negative integer value".format(value))
	return value

def restricted_float(x):
	try:
		x = float(x)
	except ValueError:
		raise argparse.ArgumentTypeError("{} is not a floating-point literal".format(x))
	if x < 0.0 or x > 1.0:
		raise argparse.ArgumentTypeError("{} is not in range [0, 1]".format(x))
	return x