from matplotlib import pyplot as plt

#random_file = 'SATModelCount/result/speed20.txt'
#regular_file = 'SATModelCount/result/rspeed20.txt'
random_file = 'result/1speed20.txt'
regular_file = 'result/1rspeed20.txt'

random_file = 'result/speed_m=10.txt'
regular_file = 'result/rspeed_m=10.txt'

def read_file_times(filename):
    reader = open(filename, 'r')

    values = {}
    while True:
        line = reader.readline().split()
        if len(line) < 6:
            break

        if line[1] not in values:
            values[line[1]] = float(line[6][0:-1])
            print line[6][0:-1], line[3]
        else:
            values[line[1]] += float(line[6][0:-1])

    f_list = []
    time_list = []
    for key, value in values.iteritems():
        f_list.append(float(key))
        time_list.append(value / 20.0)

    f_list, time_list = zip(*sorted(zip(f_list, time_list)))
    return f_list, time_list

#read the number of satisfied solutions for each f value
def read_file_solutions(filename): 
    reader = open(filename, 'r')

    values = {}
    while True:
        line = reader.readline().split()
        if len(line) < 6:
            break

        if line[1] not in values:
            values[line[1]] = 0

        if line[5] == '(True,':
            values[line[1]] += 1
        else:
            assert(line[5] == '(False,')

    f_list = []
    satisfied_count_list = []
    for key, value in values.iteritems():
        f_list.append(float(key))
        satisfied_count_list.append(value / 20.0)

    f_list, satisfied_count_list = zip(*sorted(zip(f_list, satisfied_count_list)))
    return f_list, satisfied_count_list

f_list, time_list = read_file_times(random_file)
rf_list, rtime_list = read_file_times(regular_file)
#rf_list = [item - 1.0 / 20.0 for item in rf_list]

plt.plot(f_list, time_list, label='original')
plt.plot(rf_list, rtime_list, label='regular', hold=True)
plt.xlabel('f')
plt.ylabel('runtime(s)')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.show()

#f_list, time_list = read_file_solutions(random_file)
#rf_list, rtime_list = read_file_solutions(regular_file)
##rf_list = [item - 1.0 / 20.0 for item in rf_list]
#
#plt.plot(f_list, time_list, label='original')
#plt.plot(rf_list, rtime_list, label='regular', hold=True)
#plt.xlabel('f')
#plt.ylabel('fraction of satisfied solutions found')
##plt.yscale('log')
##plt.xscale('log')
#plt.legend()
#plt.show()