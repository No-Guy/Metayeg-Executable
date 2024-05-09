try:
    from ultralytics import YOLO
    import os
    import sys
    import torch
    from Preprocess import Proc
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = YOLO(sys.argv[1]) 
    model.to(device)
    cwd = os.getcwd()
    imgpath = sys.argv[2]

    results = 0
    if(sys.argv[1][-4] == 'p'):
        results = model(Proc(imgpath),conf = float(sys.argv[3]))
    else:
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