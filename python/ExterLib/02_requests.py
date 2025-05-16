import requests

r = requests.get('https://api.github.com/users/Manans6114')


# print(r.text)
with open("Venom.txt", "w") as f:
    f.write(r.text)