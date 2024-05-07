try:
    from ultralytics import YOLO
    import os
    import sys
    model = YOLO(sys.argv[1]) 
    model.cuda()
    cwd = os.getcwd()
    imgpath = sys.argv[2]
    files = os.listdir(cwd)
    results = model(imgpath,conf = float(sys.argv[3]))
    nested_list = results[0].boxes.xywhn.tolist()
    clss = results[0].boxes.cls.tolist()
    for i in range(len(nested_list)):
        print(str(int(clss[i])),end = ' ')
        for j in range(len(nested_list[i])):
            if(j != len(nested_list[i]) -1):
                print(nested_list[i][j],end = ' ')
            else:
                print(f"{nested_list[i][j]}")
except Exception as e:
    sys.stderr.write("Python Error\n" + e)
    exit(1)
exit(0)