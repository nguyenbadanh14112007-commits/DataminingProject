def is_nt(x):
    if x < 2 : 
        return False ;
    if x == 2 :
        return True ;
    for i in range( 2 , (x//2) + 1 ):
        if x%i == 0:
            return False ;
    return True ;
def is_cp(x):
    if x < 1:
        return False 
    if x == 1:
        return True 
    for i in range ( 2 , x//2 + 1 ):
        if i*i == x:
            return True
    return False 

# x = input("Nhap x: " )
# if ( is_nt(int(x)) ):
#     print( x , " la so nguyen to!")
# elif is_cp(int(x)):
#     print( x , " la so chinh phuong!")
# else:
#     print( x , " khong phai la so nguyen to va khong phai so chinh phuong!")
for i in range ( 10 , 1 , -2 ):
    print(i) 