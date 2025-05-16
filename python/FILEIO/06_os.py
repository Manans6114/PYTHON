import os


a = os.listdir("dir")
print(a)
print(os.getcwd())
print(os.path.exists("dir"))
# os.remove("sample.txt")
os.rmdir("dir")