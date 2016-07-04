import sys
import os
import subprocess

# input folder
fi = sys.argv[1]

# init var
classes = os.listdir(fi)
list_classes = []

##################-MAIN-#########################

counter = 0
# Create sets
for cls in classes:
    list_classes.append(cls)
    imgs = os.listdir(fi + cls)
    for img in imgs:
        command_name = 'python label_image.py ./evaluation_files/eval_col_35x35/%s' % img
        print command_name
        print counter
        counter+=1
        subprocess.call(command_name, shell=True)
