

s = "hello world" # Strings are immutable


a = len(s)

# print(a)

# print(s.upper(), s)
# print(s.lower())
# print(s.capitalize())
# print(s.title())




# text = " hello world "
# print(text.strip()) # Output: "hello world"   // removes leading and trailing whitespaces from a string
# print(text.lstrip()) # Output: "hello world " // removes leading and trailing whitespaces from a string towards the left
# print(text.rstrip()) # Output: " hello world" // removes leading and trailing whitespaces from a string towards the right




# text = "Python is fun and fun and fun"
# print(text.find("is")) # Output: 7 Index of first occurence
# print(text.replace("fun", "awesome"))




 
# text = "Apples,Bananas,Pineapples"
# print(text.split(","))
# print(",".join(['Apples', 'Bananas', 'Pineapples']))





text = "Python123"
print(text.isalpha()) # Output: False
print(text.isdigit()) # Output: False
print(text.isalnum()) # Output: True
print(text.isspace()) # Output: False