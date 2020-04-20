from __main__ import *

def dewarp(DWF,frame,H,box):


    # DWF = np.zeros([frame.shape[0], frame.shape[1], 3])
    # DFW = np.float64(DWF)
    # print('############')
    x_max, y_max = np.amax(box, axis=0)
    x_min, y_min = np.amin(box, axis=0)

    # print(H)


    # Hinv=np.linalg.pinv(H)

    # Hinv=Hinv/Hinv[2,2]
    # Hinv = H/ H[2, 2]

    Hinv=H
    # print('\n',H)

    for ix in range(x_min,x_max):
        for iy in range(y_min,y_max):
            p1=np.transpose(np.array([ix,iy,1]))

            # print('p1',p1)


            p2=-np.matmul(Hinv,p1)

            p2=(p2/p2[2]).astype(int)



            # print([p2[0],p2[1]])
            # print('p2',p2)
            try:
                # if p2[0]<0 or p2[1]<0 or p2[0] >= frame.shape[1] or p2[1] >= frame.shape[0]:
                #     None
                # else:

                if (p2[0] < frame.shape[1]) and (p2[1] < frame.shape[0]) and (p2[0] > -1) and (p2[1] > -1):
                    # print('############')
                    # print(frame[p2[0],p2[1]])
                    DWF[iy,ix]=frame[p2[0],p2[1]]
            except:
                None

    # print(DWF.shape)
    return DWF