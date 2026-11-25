import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
df = pd.read_csv("data.csv")
type_count = df["Type1"].value_counts()
plt.bar( type_count.index , type_count.values , color="cyan" ,edgecolor="black" )
plt.title("# of Pokemon by Primary Type")
plt.ylabel("Count")
plt.xlabel("Type")
plt.tight_layout()
plt.show()









# Figure and Ax
# print(plt.subplots(2,2)) # tạo ma trận 2x2
# x = np.array([1 , 2, 3 , 4 , 5])
# Figure, axes = plt.subplots(2,2) 
# axes[0 , 0].plot( x , x*2 , color="red")
# axes[0 , 0].set_title("x*2")
# axes[0 , 1].plot( x , x**2 , color="blue")
# axes[0 , 1].set_title("x**2")
# axes[1 , 0].plot( x , x**3 , color="green")
# axes[1 , 0].set_title("x**3")
# axes[1 , 1].plot( x , x**4 , color="purple")
# axes[1 , 1].set_title("x**4")

# plt.tight_layout() # chỉnh bố cục
# plt.show()




# biểu đồ histogram biểu đồ cột
# scores = np.random.normal(loc=80 , scale=10 , size = 100 ) # điểm số phân bố xung quanh 80, độ phân tán 10 số lượng 100
# scores = np.clip( scores , 0 , 100 )
# plt.hist(scores , bins=10 , color="lightgreen" , edgecolor="Black")   #bins là số khoảng
# plt.title("Thong ke diem so cua lop")
# plt.xlabel("Diem so")
# plt.ylabel("So luong")
# plt.show()


# biểu đồ phân tán
# x = np.array([ 0 , 1, 1 , 2, 3 , 3 , 4 , 5 , 6 , 7 , 7 , 8]) # số giờ học 
# y = np.array([ 55 , 60 , 62 , 57 , 50 , 66 , 90 , 82 , 77 , 85 , 95 , 99]) #điểm số 
# x1 = np.array([ 0 , 4 , 1 , 2, 1 , 3 , 9 , 3 , 6 , 5 , 7 , 8]) # số giờ học 
# y1 = np.array([ 40 , 67 , 55 , 57 , 50 , 83 , 90 , 78 , 65 , 85 , 95 , 99]) #điểm số 
# plt.scatter(x , y , color="cyan" , alpha=0.5 , s=200 , label="Class A") # alpha độ trong suốt , s là kích thước
# plt.scatter( x1 , y1 , color="red" , s=100 , label="Class B") 
# plt.title("Diem kiem tra")
# plt.xlabel("So gio hoc")
# plt.ylabel("So diem")
# plt.legend() #chú giải cho label trên scatter 
# plt.show()






# Biểu đồ cầu 
# categories = np.array(["Freshmen" , "Sophomores" , "Juniors" , "Seniors" ])
# values = np.array([ 300 , 250 , 275 , 225])
# colors = ["red" , "yellow" , "blue" , "green"]
# plt.pie( values , labels= categories , autopct="%1.1f%%" , colors= colors , explode=[0 , 0 , 0 , 0.1] , shadow=True )
# plt.title("Sai gon college")
# plt.show()


# tạo biểu đồ cột 
# categories = np.array(["Grains" , "Fruits" , "Vegetables" , "Protein" , "Dairy" , "Sweets"])
# values = np.array([4 , 3 , 2 , 5 , 3 , 1])
# plt.bar( categories , values , color="red")
# plt.title("Daily consumption")
# plt.xlabel("Food")
# plt.ylabel("Quantity")
# plt.show() 


# x = np.array([ 2023, 2024 , 2025 , 2026])
# y = np.array([ 15 , 25 , 30 , 20 ])
# y1 = np.array([5 , 28 , 17 , 23])
# y2 = np.array([13 , 15 , 20 , 30])
# grid: tạo ra các đường lướt 
# plt.grid(axis="both" , linewidth= 2, color="lightgray" , linestyle="dashed")
# plt.plot(x , y)
# plt.plot(x , y1)
# plt.plot(x , y2)
# plt.xticks(x)
# plt.show() 

#tuỳ chỉnh thuộC tính
# plt.title("Class size" , fontsize=30 , family="Arial" , fontweight="bold" , color="cyan") #tiêu đề
# plt.xlabel("Year" , fontsize=15 , family="Arial" , fontweight="bold" , color="blue" )
# plt.ylabel("Students" , fontsize=15 , family="Arial" , fontweight="bold" , color="blue" )
# plt.tick_params(axis="both", colors ="red")
# plt.plot(x , y)
# plt.plot(x , y1)
# plt.plot(x , y2)

# plt.xticks(x) # chỉ giữ lại năm 
# plt.show()


# line_style = dict(  marker="." , 
#                     markersize= 10 , 
#                     markerfacecolor="cyan" ,
#                     markeredgecolor="red" ,
#                     linestyle="dashdot" ,
#                     linewidth=2)
# plt.plot(x,y, color="#1c5bfc" ,**line_style) # marker là điểm đánh dấu trên đồ thị
# plt.plot(x , y1 , color="red" , **line_style)
# plt.show()
