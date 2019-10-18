from PIL import Image
import sys
import os


folder = sys.argv[1]

for filename in os.listdir(folder):
    name = filename.split(".")[0]
    Image.open(folder + "/" + filename).save(folder + "/" + name + ".jpg")
#print(folder)

#Image.open("sample1.jpg").save("sample1.png")