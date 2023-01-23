import csv
import requests
csv_reader = csv.reader(open("hugu-challenge-filenames.csv"))
num = 0
for line in csv_reader:
    png = line[0]
    url = "https://cs.stanford.edu/people/rak248/VG_100K_2/1.jpg"
    # url = "http://cs.stanford.edu/api/v0/images/all?page=1"
    # print(url)
    r = requests.get(url)
    img = r.content
    path = 'pictures/' + png
    with open(path, 'wb') as f:
        f.write(img)
    # print(r)
    num += 1
    if num % 100 == 0:
        print(num)














