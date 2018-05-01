from matplotlib import pyplot as plt

#random_file = 'SATModelCount/result/speed20.txt'
#regular_file = 'SATModelCount/result/rspeed20.txt'
random_file = 'result/speed20.txt'
regular_file = 'result/rspeed20.txt'

def read_file(filename):
    reader = open(filename, 'r')

    values = {}
    while True:
        line = reader.readline().split()
        if len(line) < 6:
            break

        if line[1] not in values:
            values[line[1]] = float(line[3])
        else:
            values[line[1]] += float(line[3])

    f_list = []
    time_list = []
    for key, value in values.iteritems():
        f_list.append(float(key))
        time_list.append(value / 20.0)

    f_list, time_list = zip(*sorted(zip(f_list, time_list)))
    return f_list, time_list

f_list, time_list = read_file(random_file)
rf_list, rtime_list = read_file(regular_file)
rf_list = [item - 1.0 / 20.0 for item in rf_list]

plt.plot(f_list, time_list, label='original')
plt.plot(rf_list, rtime_list, label='regular', hold=True)
plt.xlabel('f')
plt.ylabel('runtime(s)')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.show()