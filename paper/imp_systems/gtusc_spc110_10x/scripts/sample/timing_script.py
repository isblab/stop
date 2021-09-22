import re
import time

regex = re.compile('--- frame [0-9]+ score')
file_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
while True:
    time.sleep(60)
    for n in file_list:
        path = f'timing/stdout_{n}.txt'
        with open(path, 'r') as f:
            rd = f.read()
            m = regex.findall(rd)
        with open('timing/timing_data.txt', 'a') as f:
            f.write(f'{n}:{len(m)}:{time.time()}\n')

