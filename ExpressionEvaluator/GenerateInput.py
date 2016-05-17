import random

def main():
	exp = raw_input("Please enter the expression: ")
	varNames = []
	for c in exp:
		if c.isalpha() and not c in varNames:
			varNames.append(c)
	print varNames
	size = raw_input("Enter the desired length of vector: ")
	size = int(size)
	file = open("testfile", "w")
	file.write(str(exp)+"\n")
	for name in varNames:
		file.write(name+",")
	file.write("\n")
	for row in xrange(size):
		for col in xrange(len(varNames)):
			file.write(str(random.uniform(-10000, 10000)))
			file.write(",")
		file.write("\n")
	file.close()


if __name__ == "__main__":
    main()