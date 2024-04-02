from ultralytics import YOLO
import sys
import os
import torch
model = YOLO(sys.argv[1]) 
model.cuda()
cwd = os.getcwd()

# List all files in the current working directory
files = os.listdir(cwd)

# Filter for files with a .png extension
png_files = [file for file in files if file.endswith(".png") or file.endswith(".jpeg") or file.endswith(".jpg")]
#print(png_files)
for file in png_files:
    results = model(file,conf = float(sys.argv[2]))
    nested_list = results[0].boxes.xywhn.cpu().tolist()
    clss = results[0].boxes.cls.cpu().tolist()
    with open(os.path.splitext(os.path.basename(file))[0] +".txt", 'w') as file:
        for i in range(len(nested_list)):
            file.write(str(int(clss[i])) + " ")# + str(nested_list[i]) + '\n')
            for j in range(len(nested_list[i])):
                if(j != len(nested_list[i]) -1):
                    file.write(f"{nested_list[i][j]} ")
                else:
                    file.write(f"{nested_list[i][j]}\n")
#image = results[0].plot()
#r, g, b = cv2.split(image)

# Swap the Red and Blue channels
#swapped_image = cv2.merge([b, g, r])

# Display or save the swapped image
