from __main__ import *
import math
count=0
gc=0
oc=0
yc=0
switch=2

def buildData(frame):

    cv2.imshow("Original Frame", frame)
    cv2.setMouseCallback("Original Frame", getClickInfoGreen,frame)
    cv2.waitKey(0)



def getClickInfoGreen(event, x, y, flag, frame):
    global cx, cy, count, gc,oc,yc, switch


    if event == cv2.EVENT_LBUTTONDOWN and switch==2:
        cx = x
        cy = y
        switch=1
    elif event == cv2.EVENT_LBUTTONDOWN and switch==1:
        switch=2
        r = int(np.sqrt((x - cx) ^ 2 + (y - cy) ^ 2))*3
        circle,xs,ys= rotate(x, y, r, 10)


        crop_img = frame[cy-r:cy+r, cx-r:cx+r]


        if count % 3 == 0:
            print("green")
            cv2.imwrite("green" + str(gc) + ".png", crop_img)
            gc += 1
        elif count % 3 == 1:
            print("orange")
            cv2.imwrite("orange" + str(oc) + ".png", crop_img)
            oc += 1
        elif count % 3 == 2:
            print("yllw")
            cv2.imwrite("yllw" + str(yc) + ".png", crop_img)
            yc += 1
        count += 1

        # cv2.imshow("cropped", crop_img)

        frame = cv2.circle(frame, (cx, cy), r, (255, 0, 0), 1)
        cv2.imshow("Original Frame", frame)


def rotate(x, y, r, points):
    fullarc = []
    xarcT=[]
    yarcT=[]
    gate = 360 / points
    for i in range(points):
        xarc = r * math.cos(gate * i) + x
        yarc = r * math.cos(gate * i) + y

        fullarc.append([xarc, yarc])
        xarcT.append(xarc)
        yarcT.append(yarc)

    return fullarc, xarcT, yarcT


