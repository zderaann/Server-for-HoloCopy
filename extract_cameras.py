import sys

foldername = sys.argv[1]
f = open(foldername + "/parsed_cameras.txt", "w+")


with open(foldername + "/images.txt", 'r') as file:
    while True:
        line = file.readline()
        # print(data)
        if line.startswith('# Number of images:'):
            numOfCams = int(line.split(' ')[4].split(',')[0])
            print("Number of cameras: " + str(numOfCams))
            break
    for i in range(0, numOfCams):
        line = file.readline()
        f.write(line)
        file.readline()

f.close()