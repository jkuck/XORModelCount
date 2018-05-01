from __future__ import division
from matplotlib import pyplot as plt
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()
from collections import defaultdict


#random_file = 'SATModelCount/result/speed20.txt'
#regular_file = 'SATModelCount/result/rspeed20.txt'
#random_file = 'result/speed20.txt'
#regular_file = 'result/rspeed20.txt'

REPEATS = 100
random_file = 'heatmap_result_constantF/speed_smallF_REPEATS=%d_lang12.txt' % REPEATS
#regular_file = 'heatmap_result/rspeed_smallF_REPEATS=%d_lang12.txt' % REPEATS
regular_file = 'heatmap_result_constantF/rspeed_smallF_REPEATS=%d_lang12.txt' % REPEATS
permutation_file = 'heatmap_result_constantF/pspeed_smallF_REPEATS=%d_lang12.txt' % REPEATS

def read_file(filename, repeats):
    '''

    Inputs:
    - repeats: (int) the number of experiments run for each f and m value
    '''
    print "reading file:", filename
    reader = open(filename, 'r')

    #key: (f, m)
    #value: list of all runtimes (len repats)
    all_runtimes = defaultdict(list)
    #key: (f, m)
    #value: list of 0/1 values (len repats)
    #   1: problem was satisfiable
    #   0: problem was not satisfiable
    problem_satisfied = defaultdict(list)
    f_vals = set()
    m_vals = set()
    while True:
        line = reader.readline().split()
        if len(line) < 9:
            if len(line) == 8 and line[7] == 'None':
                print "error, solution = None encountered, how did this happen?"
                continue
            else:
                break

        f = float(line[1])
        run_time = float(line[8][0:-1])
        m = int(line[5])
        if line[7] == '(True,':
            satisfied = True
        else:
            assert(line[7] == '(False,')
            satisfied = False

        m_vals.add(m)
        f_vals.add(f)
        all_runtimes[(f,m)].append(run_time)
        problem_satisfied[(f,m)].append(satisfied)

    sorted_m_vals = sorted(m_vals)
    sorted_f_vals = sorted(f_vals)
    print m_vals
    print sorted_m_vals
    mean_runtimes = np.zeros((len(sorted_f_vals), len(sorted_m_vals)))
    fraction_SAT = np.zeros((len(sorted_f_vals), len(sorted_m_vals)))

    for (m_idx, m_val) in enumerate(sorted_m_vals):
        for (f_idx, f_val) in enumerate(sorted_f_vals):
#            assert(len(all_runtimes[(f_val, m_val)]) == repeats or len(all_runtimes[(f_val, m_val)]) == repeats-1), (len(all_runtimes[(f_val, m_val)]), f_val, m_val)
#            assert(len(problem_satisfied[(f_val, m_val)]) == repeats or len(problem_satisfied[(f_val, m_val)]) == repeats-1), (len(problem_satisfied[(f_val, m_val)]), f_val, m_val)
            mean_runtimes[f_idx, m_idx] = np.mean(all_runtimes[(f_val, m_val)])
            fraction_SAT[f_idx, m_idx] = np.mean(problem_satisfied[(f_val, m_val)])
    return(sorted_m_vals, sorted_f_vals, mean_runtimes, fraction_SAT)

def get_runtime_by_f(sorted_m_vals, sorted_f_vals, mean_runtimes, fraction_SAT):
    '''

    Outputs:
    - m_runtime: list of tuples (m_value, minimum runtime over all f values that produced > .5 satisfied problems)
    '''
    m_runtime = []
    for (m_idx, m_val) in enumerate(sorted_m_vals):
        min_runtime = None
        for (f_idx, f_val) in enumerate(sorted_f_vals):
            if fraction_SAT[f_idx, m_idx] > .5 and (min_runtime==None or mean_runtimes[f_idx, m_idx] < min_runtime):
                min_runtime = mean_runtimes[f_idx, m_idx]
        m_runtime.append((m_val, min_runtime))
    return m_runtime


##### original randomness #####
(sorted_m_vals, sorted_f_vals, mean_runtimes, fraction_SAT) = read_file(random_file, repeats=REPEATS)
m_runtime = get_runtime_by_f(sorted_m_vals, sorted_f_vals, mean_runtimes, fraction_SAT)
m_list, original_runtime_list = zip(*m_runtime)
assert(m_list == sorted_m_vals)


##### block diagonal + randomness #####
(sorted_m_vals, sorted_f_vals, mean_runtimes, fraction_SAT) = read_file(regular_file, repeats=REPEATS)
m_runtime = get_runtime_by_f(sorted_m_vals, sorted_f_vals, mean_runtimes, fraction_SAT)
m_list, blockDiag_runtime_list = zip(*m_runtime)
assert(m_list == sorted_m_vals)


##### permutation + randomness #####
(sorted_m_vals, sorted_f_vals, mean_runtimes, fraction_SAT) = read_file(permutation_file, repeats=REPEATS)
m_runtime = get_runtime_by_f(sorted_m_vals, sorted_f_vals, mean_runtimes, fraction_SAT)
m_list, permutation_runtime_list = zip(*m_runtime)
assert(m_list == sorted_m_vals)


plt.plot(m_list, [blockDiag_runtime_list[i]/original_runtime_list[i] for i in range(len(original_runtime_list))], label='blockDiag/orig')
plt.plot(m_list, [permutation_runtime_list[i]/original_runtime_list[i] for i in range(len(original_runtime_list))], label='permutation/orig', hold=True)
plt.xlabel('m')
plt.ylabel('runtime ratio')
plt.legend()
plt.show()

