import Project1Main as fa
import numpy as np
# p1 = np.array([[3, 0], [5, 0], [3, 0]])
# p2 = np.array([[0, 4], [0, 5], [0, 4]])
# d, theta = fa.distanceAndAngle(p1, p2)
# print(np.pi/2)
# print(theta)
# print(d)
#
# corners = [[2,3], [5,2],[4,1],[3.5,1],[1,2],[2,1],[3,1],[3,3],[4,3]]
#
# print("Before sort: " + str(corners))
# """sort the corner from top-right to bottom-right to bottom-left to top-left"""
# # corners, center_original = shiftToASpot(corners, np.array([0, 0]))
# corners = sorted(corners, key=fa.clockwiseangle_and_distance)
# # corners, shouldBeZeros = shiftToASpot(corners, center_original)
# print("After sort: " + str(corners))


img_dst = np.zeros([3,3,0])  # np.array([[[0,0,0], [2,3,2], [5,2,1]],[[0,0,0], [2,3,2], [5,2,1]]])

print(img_dst[:,:])

(rows, cols, channels) = img_dst.shape

print(str(rows) + ", " + str(cols))
for row in range(0, rows):
    for col in range(0, cols):
        print(str(row) + ", " + str(col))
        if img_dst[row, col].all() == 0:
            img_dst[row, col] = np.array([6,6,6])

print(img_dst)
# img_dst = img_dst.transpose()
print(img_dst)
