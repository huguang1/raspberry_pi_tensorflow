with open('b.csv', 'a') as f2:
    with open('a.csv') as f:
        for line in f.readlines():
            line = line.lstrip()
            b = ''.join(line.split(' '))
            f2.write(b)
