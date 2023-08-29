import math
import numpy as np


def kiem_dinh():
    # Kích thước nhỏ, khó sai thì nhập vào chương trình
    k = int(input("Nhập số biến k: "))
    n = int(input("Nhập số lượng mẫu n: "))

    print("Nhập ma trận X:")
    X = np.array(([[0] * (k)] * n), "f")
    for i in range(0, n):
        for j in range(0, k):
            t = "Nhập phần tử X{}{}: "
            X[i, j] = float(input(t.format(i + 1, j + 1)))
            j = j + 1
        i = i + 1

    print("Nhập vector Y:")
    Y = np.array([[0] * 1] * n, "f")
    for i in range(0, n):
        t = "Nhập phần tử Y{}: "
        Y[i, 0] = float(input(t.format(i + 1)))
        i = i + 1

    # Kích thước lớn, dễ sai thì nhập thẳng vào code
    """
    X = np.array([
    [6,1,2,1],
    [10,1,2,2],
    [10,1,3,2],
    [11,1,3,2],
    [13,1,3,1.7],
    [13,2,3,2.5],
    [13,1,3,2],
    [17,2,3,2.5],
    [19,2,3,2],
    [18,1,3,2],
    [13,1,4,2],
    [18,1,4,2],
    [17,2,4,3],
    [20,2,4,3],
    [21,2,4,3]
    ])
    
    Y = np.array([[169],[218.5],[216.5],[225],[229.9],[235],[239.9],
    [247.9],[260],[269.9],[234.9],[255],[269.9],[294.5],[309.9]])
    
    # Số hàng và cột
    n = len(X)
    k = X.size/n - 1
    print('n = ',n)
    print('k = ',k)
    """

    temp = np.array([[1] * n])
    X = np.concatenate((temp.transpose(), X), axis=1)  # Thêm 1 vào cột đầu tiên ma trận X
    print(X)
    print(Y)
    # X = delete(X,[3],1) # Loại bỏ B3

    # B mũ
    B = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(Y)
    print('Uoc luong B = ')
    print(B)

    # H0: B1 = B2 = ... = Bk = 0
    # H1: Tồn tại Bi != 0
    # f0 = MSr/MSe phân phối Fisher
    # Miền bác bỏ H0: (f(a;k;n−k-1); +vo cung)
    # SSE
    SSe = Y.transpose().dot(Y) - B.transpose().dot(X.transpose()).dot(Y)
    print('SSe = ', SSe)
    # SST
    SSt = Y.transpose().dot(Y) - pow(np.sum(Y), 2) / n
    print('SSt = ', SSt)
    # SSr
    SSr = SSt - SSe
    print('SSr = ', SSr)

    # MSr, MSe
    MSr = SSr / k
    MSe = SSe / (n - k - 1)  # = ước lượng o^2
    print('f0 = ', MSr / MSe)

    R2 = 1 - SSe / SSt
    print('R^2 = ', R2)
    print('R = ', math.sqrt(R2))
    print('R^2 hieu chinh = ', 1 - SSe * (n - 1) / SSt / (n - k - 1))

    # Kiểm định từng hệ số hồi quy
    # H0: Bi = 0
    # H1: Bi != 0
    # t0 = Bi mũ/se(Bi mũ) phân phối Student
    # Miền bác bỏ H0: (−inf; −t(a/2;n−k−1)) and (t(a/2;n−k−1); +inf)

    # Chon hệ số cần kiểm định
    i = 1
    SE2 = MSe * (np.linalg.inv(X.transpose().dot(X)))
    print('SE', [i], '=', math.sqrt(SE2[i][i]))


    # test thử commit change
    print('B', i, '= ', B[i])
    t0 = B[i] / math.sqrt(SE2[i][i])
    print('t', i, '= ', t0)
