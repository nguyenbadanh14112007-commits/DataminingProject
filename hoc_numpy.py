import numpy as np 
#Random numbers
rng = np.random.default_rng()
array = np.array([1 , 2, 3 , 4, 5])
# print(rng.integers( low = 1 , high = 101 , size = (3 ,2 ) )) #số nguyên
# print( rng.uniform( low = -1 , high = 1 )) #số thực
# rng.shuffle(array) #trộn mảng
x = rng.choice(array) #chọn 1 phần tử ngẫu nhiên trong array
print(x)







#filtering: lớp lọc
# array = np.array([[23 , 19 , 4 , 9 , 14 , 45 , 27] , 
#                   [5 , 24 , 55 , 35 , 67 , 18 , 26]])
# array1 = array[array > 18] 
# array2 = array[( array > 5 ) & (array < 18 )]
# array2 = array[( array > 5 ) | (array < 18 )]
# array3 = np.where(array > 18 , array , 0 ) # thay đổi giá trị nếu thấy giá trị đó < 18 còn  > 18 thì giữ nguyên
# print(array3) 
# hàm tổng hợp: các hàm tóm tắt dữ liệu và thường trả về 1 giá trị duy nhất 
# array = np.array([[1, 2 , 3 , 4 , 5] , 
#                   [6 , 7 , 8 , 9 , 10]])
# print(np.sum(array , exis = 0 )) # áp dụng cho từng cột
# print(np.sum(array , exis = 1 )) # áp dụng cho tất cả các cột
# print(np.sum(array)) #tổng ma trận
# print(np.mean(array)) # số trung bình ma trận
# print(np.std(array))  # độ phân tán của ma trận
# print(np.var(array)) #phương sai của ma trận
# print(np.min(array)) # giá trị nhỏ nhất max là lớn nhất
# print(np.argmin(array)) # vị trí của giá trị nhỏ nhất hoặc lớn nhất ( argmax)

#Broandcasting: cho phép tính toán trên các mảng khác nhau với điều kiện kích thước 2 mảng trùng khớp hoặc có ít nhất 1 số 1 ở 1 trong 2 mảng 
# array = np.array([[1 , 2 , 3 , 4]])
# array1 = np.array([[1] , [2] , [3] , [4]]) 
# print(array*array1)
# array = np.array([[1 , 2 , 3 , 4] , 
#                  [ 5 , 6 , 7 , 8 ] , 
#                  [ 9 , 10 , 11 , 12] , 
#                  [ 13 , 14 , 15 , 16]]) mảng bậc 2
# array = np.array([1 , 2, 3])
# array2 = np.array([4 , 5 , 6])
# print(array[1])
# array[start][end][step] chọn hàng 
# array[: , cột , cột end ] vd array[: , 1::2 ] từ cột 1 đến 2 
# print(array[ 0:2 , 2:4 ])
# scalar irithmetic
# print( array + array2 ) #cộng 2 ma trận 
# print(array + 1 ) # cộng trừ nhân chia tấT cả phần tử 
# print( np.sqrt(array)) #căn bậc
# print( np.pow(array , 3)) # round , floor , ceil, pi,..
