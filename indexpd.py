import pandas as pd 
df = pd.read_csv("data.csv" , index_col="Name")
# Data cleaning: làm sạch dữ liệu là quá trình sửa chữa và loại bỏ các dữ liệu thừa hoặc không liên quan 
# xoá cột thừa
df = df.drop(columns=["Legendary" , "No"])  
# xoá hoặc thay đổi các cột bị thiếu thuộc tính
df = df.dropna(subset=["Type2"]) #xoá các cột nếu thuộc tính Type2 để trống
df = df.fillna({"Type2": "None"})# thay thế các cột có thuộc tính Type 2 để trống bằng none
# sửa các dữ liệu thiếu nhất quán
df["Type1"] = df["Type1"].replace({"Glass":"GLASS" ,
                                   "Fire": "FIRE"}) # đổi ngữ nghĩa
# chuẩn hoá văn bản
df["Name"] = df["Name"].str.lower() # viết hoa thành thường
# sửa kiểu dữ liệu
df["Legendary"] = df["Legendary"].astype(bool) #đổi kiểu dữ liệu
# xoá dữ liệu trùng lặp
df = df.drop_duplicates() 



# hàm tổng hợp sử dụng thư viện pandas
# mọi cột
# print(df.mean(numeric_only=True)) # giá trị trung bình các thuộc tính của đối tượng
# print(df.sum(numeric_only=True)) #tổng
# print(df.min(numeric_only=True)) # tìm min max
# print(df.count) # đếm số lượng giá trị trong mỗi cột
#Một cột duy nhất
# print(df["Height"].mean()) # giá trị trung bình các thuộc tính Height
# print(df["Weight"].sum()) #tổng
# print(df["Height"].min()) # tìm min max
# print(df["Weight"].count) # đếm số lượng giá trị trong mỗi cột
# df.describe() liệt kê toàn bộ các giá trị cần tra cứu như  count  độ lệch chuẩn sum mean tứ phân vị,...

#nhóm theo group by
# group = df.groupby("Type1") 
# print(group["Height"].mean()) # với mọi Type1 in chiều cao trung bình của nó như loài rồng có chiều cao trung bình bao nhiêu chẳng hạn
# print(group["Height"].sum())
# print(group["Height"].min())
# print(group["Height"].max())
# print(group["Height"].count()) # đếm số đối tượng thuộc mỗi loại Type1
#filtering: lọc dữ liệU
# tall_pokemon = df[df["Height"] >= 1 ]
# heavy_pokemon = df[df["Weight"] > 50 ]
# ff_pokemon = df[(df["Type1"] == "Fire") & (df["Type2"] == "Flying")]
# print(ff_pokemon)




#Selection 
#Selection By Column
# print(df["Name"].to_string())
# print(df[["Name" , "Height" , "Weight"]])
# Select By Row/s
# print(df.loc["Kakuna"])
# print(df.iloc[0:11:2 , 0:3 ])


#importing như đọc từ file ấy 
# df = pd.read_csv("data.csv")
# df = pd.read_json("data.json")
# print(df)



#DataFrame: là kiểu dữ liệu gồm cả hàng lẫn cột như ma trận bậc 2 và dữ liệu như SQl và mỗi dataFrame là mỗi dữ liệu của đối tượng 
# data = { "Name": ["Spongebob" , "Patrick" , "Squidward"],
#          "age": [ 30 , 33 , 35 ]
# }
# df = pd.DataFrame(data , index=["employee 1" , "employee 2" , "employee 3"]) 
# #add a new column 
# df["Job"] = ["Cook" , "N/A" , "cashier" ]
# #add a new row
# new_row = pd.DataFrame([{"Name": "sandy" , "age": 32 , "Job": "Engineer"}] , index=["employee 4"])
# df = pd.concat([df , new_row]) 
# print(df)


# Series: dữ liệu kiểu cột đơn như sql 
# calories = {"day 1": 1750 , "day 2": 2100 , "day 3": 1700}
# series = pd.Series(calories) 
# print(series)
# data = [1 , 2 , 3 , 4] 
# serie = pd.Series(data , index = [ "a", "b" , "c" , "d"]) #index là chỉ mục của từng giá trị trong data
# print(serie.loc["a"]) 