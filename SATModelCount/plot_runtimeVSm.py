from __future__ import division
from matplotlib import pyplot as plt
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()
from collections import defaultdict
from plot_heatmap import read_file
from plot_heatmap import read_files
import math

#random_file = 'SATModelCount/result/speed20.txt'
#regular_file = 'SATModelCount/result/rspeed20.txt'
#random_file = 'result/speed20.txt'
#regular_file = 'result/rspeed20.txt'

REPEATS = 100
random_file = 'heatmap_result_constantF/speed_smallF_REPEATS=%d_lang12.txt' % REPEATS
#regular_file = 'heatmap_result/rspeed_smallF_REPEATS=%d_lang12.txt' % REPEATS
regular_file = 'heatmap_result_constantF/rspeed_smallF_REPEATS=%d_lang12.txt' % REPEATS
permutation_file = 'heatmap_result_constantF/pspeed_smallF_REPEATS=%d_lang12.txt' % REPEATS


#random_file = 'heatmap_result/speed2_REPEATS=%d_lang12.txt' % REPEATS
##regular_file = 'heat/rspeed2_REPEATS=%d_lang12.txt' % REPEATS
#regular_file = 'heatmap_result/rspeed2_REPEATS=%d_lang12.txt' % REPEATS
#permutation_file = 'heatmap_result/pspeed2_REPEATS=%d_lang12.txt' % REPEATS

REPEATS = 100
random_file = 'heatmap_result_moreModels5/speed_REPEATS=%d_c499.txt' % REPEATS
#regular_file = 'heatmap_result/rspeed_REPEATS=%d_c499.txt' % REPEATS
regular_file = 'heatmap_result_moreModels5/rspeed_REPEATS=%d_c499.txt' % REPEATS
permutation_file = 'heatmap_result_moreModels5/pspeed_REPEATS=%d_c499.txt' % REPEATS


REPEATS = 10
permutedBlockDiag_filebase = 'heatmap_result_fireworks/permutedBlockDiag_speed_REPEATS=%d_c432_duplicates=0_expIdx=' % REPEATS
regular_filebase = 'heatmap_result_fireworks/rspeed_REPEATS=%d_c432_duplicates=0_expIdx=' % REPEATS
original_filebase = 'heatmap_result_fireworks/speed_REPEATS=%d_c432_duplicates=0_expIdx=' % REPEATS

REPEATS = 10
PROBLEM_NAME = 'tire-1'
permutedBlockDiag_filebase = 'heatmap_result_fireworksWED/permutedBlockDiagDeterministic_speed_REPEATS=%d_%s_duplicates=0_expIdx=' % (REPEATS, PROBLEM_NAME)
regular_filebase = 'heatmap_result_fireworksWED/blockDiagDeterministic_speed_REPEATS=%d_%s_duplicates=0_expIdx=' % (REPEATS, PROBLEM_NAME)
original_filebase = 'heatmap_result_fireworksWED/speed_REPEATS=%d_%s_duplicates=0_expIdx=' % (REPEATS, PROBLEM_NAME)


USE_MULTIPLE_FILES = True #aggregate results from multiple files if true
FILE_COUNT = 20
PLOT_BLOCK_DIAG = True
PLOT_BLOCK_DIAG_PERMUTED = True
PLOT_PERMUTATION_K1 = False

def get_runtime_by_f(sorted_m_vals, sorted_f_vals, mean_runtimes, fraction_SAT):
    '''

    Outputs:
    - m_runtime: list of tuples (m_value, minimum runtime over all f values that produced > .5 satisfied problems)
    '''
    mVals_runtimes = []
    for (m_idx, m_val) in enumerate(sorted_m_vals):
        min_runtime = None
        min_fval = None
        for (f_idx, f_val) in enumerate(sorted_f_vals):
            if fraction_SAT[f_idx, m_idx] > .5 and (min_runtime==None or mean_runtimes[f_idx, m_idx] < min_runtime):
                min_runtime = mean_runtimes[f_idx, m_idx]
                min_fval = f_val
        if min_runtime == None:
            print "whoops, min_runtime = None, m_val=", m_val
            min_runtime = 1
        else:
            print 'm =', m_val, 'min_runtime =', min_runtime, 'min_fval =', min_fval
        mVals_runtimes.append((m_val, min_runtime))
    return mVals_runtimes

def get_lower_bound(num_SAT, T, run_times, min_confidence=.95):
    '''
    
    Inputs:
    - num_SAT: int, the number of trials where we found a satisfying solution
    - T: int, the number of trials
    - min_confidence: float, the probability with which the bound holds
    - run_times: list of floats of length num_SAT,
        where each entry corresponds to the runtime for solving that problem instance

    Outputs:
    - bounds: list of floats, bounds on ln(set size) we can compute sorted from worst to best
    - parallel_runtimes: list of floats (same length as bounds), 
        parallel runtime required to compute the corespoding bound (longest runtime for a problem)
    '''
    bounds = []
    parallel_runtimes = []
    delta = 1 - min_confidence
    ln = math.log
    run_times.sort()
    assert(len(run_times) == num_SAT)
    for cur_num_SAT in range(1, num_SAT+1):
        if cur_num_SAT > -ln(delta):
            c = cur_num_SAT/T
            kappa = -3 * ln(delta) + sqrt(ln(delta) ** 2 - 8 * c * T * ln(delta))
            kappa /= 2 * (c * T + ln(delta))
            cur_bound = m * ln(2) + ln(c) - ln(1 + kappa)
            bounds.append(cur_bound)
            parallel_runtimes.append(run_times[cur_num_SAT - 1])
    return(bounds, parallel_runtimes)

def read_files_moreInfo(filename_base, repeats, file_count):
    '''

    Inputs:
    - repeats: (int) the number of experiments run for each f and m value
    - file_count: (int) the number of files containing identical experiment sets
    '''
    #key: (f, m)
    #value: list of all runtimes (len repeats)
    all_runtimes = defaultdict(list)

    #key: (f, m)
    #value: list of runtimes for problems that were found to be satisfiable
    SAT_runtimes = defaultdict(list)

    #key: (f, m)
    #value: list of runtimes for problems that were found to be not satisfiable
    UNSAT_runtimes = defaultdict(list)

    #key: (f, m)
    #value: number of trials performed with these values of f and m (SAT count + UNSAT count)
    trials_count = defaultdict(int)

    #key: (f, m)
    #value: list of 0/1 values (len repeats)
    #   1: problem was satisfiable
    #   0: problem was not satisfiable
    problem_satisfied = defaultdict(list)
    f_vals = set()
    m_vals = set()

    for exp_idx in range(file_count):
        cur_filename = '%s%d.txt' % (filename_base, exp_idx)
        print "reading file:", cur_filename
        reader = open(cur_filename, 'r')
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
                satisfied = 1
                SAT_runtimes[(f,m)].append(run_time)
            else:
                assert(line[7] == '(False,')
                satisfied = 0
                UNSAT_runtimes[(f,m)].append(run_time)


            m_vals.add(m)
            f_vals.add(f)
            all_runtimes[(f,m)].append(run_time)
            problem_satisfied[(f,m)].append(satisfied)
            trials_count[(f,m)] += 1

    sorted_m_vals = sorted(m_vals)
    sorted_f_vals = sorted(f_vals)
    print m_vals
    print sorted_m_vals
    num_SAT = np.zeros((len(sorted_f_vals), len(sorted_m_vals)))

    for (m_idx, m_val) in enumerate(sorted_m_vals):
        for (f_idx, f_val) in enumerate(sorted_f_vals):
            SAT_runtimes[f_idx, m_idx].sort()
            UNSAT_runtimes[f_idx, m_idx].sort()
            num_SAT[f_idx, m_idx] = np.sum(problem_satisfied[(f_val, m_val)])
            assert(len(UNSAT_runtimes[f_idx, m_idx]) + len(SAT_runtimes[f_idx, m_idx]) == trials_count[f_idx, m_idx])
            assert(len(SAT_runtimes[f_idx, m_idx]) == num_SAT[f_idx, m_idx])

    return(sorted_m_vals, sorted_f_vals, SAT_runtimes, num_SAT, trials_count)

if __name__=="__main__":

    ##### original randomness #####
    if USE_MULTIPLE_FILES: 
        (sorted_m_vals, sorted_f_vals, mean_runtimes, fraction_SAT) = read_files(filename_base=original_filebase, repeats=REPEATS, file_count=FILE_COUNT)
    else:
        (sorted_m_vals, sorted_f_vals, mean_runtimes, fraction_SAT) = read_file(random_file, repeats=REPEATS)

    print '-'*30, 'original randomness', '-'*30
    m_runtime = get_runtime_by_f(sorted_m_vals, sorted_f_vals, mean_runtimes, fraction_SAT)
    m_list, original_runtime_list = zip(*m_runtime)
    assert(list(m_list) == list(sorted_m_vals)), (m_list, sorted_m_vals)


    ##### block diagonal + randomness #####
    if USE_MULTIPLE_FILES:
        (sorted_m_vals, sorted_f_vals, mean_runtimes, fraction_SAT) = read_files(filename_base=regular_filebase, repeats=REPEATS, file_count=FILE_COUNT)
    else:
        (sorted_m_vals, sorted_f_vals, mean_runtimes, fraction_SAT) = read_file(regular_file, repeats=REPEATS)

    print '-'*30, 'block diag', '-'*30
    m_runtime = get_runtime_by_f(sorted_m_vals, sorted_f_vals, mean_runtimes, fraction_SAT)
    m_list, blockDiag_runtime_list = zip(*m_runtime)
    assert(list(m_list) == list(sorted_m_vals)), (m_list, sorted_m_vals)


    ##### permuted block diagonal + randomness #####
    if USE_MULTIPLE_FILES:
        (sorted_m_vals, sorted_f_vals, mean_runtimes, fraction_SAT) = read_files(filename_base=permutedBlockDiag_filebase, repeats=REPEATS, file_count=FILE_COUNT)
    else:
        (sorted_m_vals, sorted_f_vals, mean_runtimes, fraction_SAT) = read_file(permutedBlockDiag_file, repeats=REPEATS)    

    print '-'*30, 'permuted block diag', '-'*30
    m_runtime = get_runtime_by_f(sorted_m_vals, sorted_f_vals, mean_runtimes, fraction_SAT)
    m_list, permutedBlockDiag_runtime_list = zip(*m_runtime)
    assert(list(m_list) == list(sorted_m_vals)), (m_list, sorted_m_vals)

    PLOT_PERMUTATION_K1 = False
    if PLOT_PERMUTATION_K1:
        ##### permutation + randomness #####
        (sorted_m_vals, sorted_f_vals, mean_runtimes, fraction_SAT) = read_file(permutation_file, repeats=REPEATS)
        m_runtime = get_runtime_by_f(sorted_m_vals, sorted_f_vals, mean_runtimes, fraction_SAT)
        m_list, permutation_runtime_list = zip(*m_runtime)
        assert(list(m_list) == list(sorted_m_vals)), (m_list, sorted_m_vals)
        
    print zip(m_list, [blockDiag_runtime_list[i]/original_runtime_list[i] for i in range(len(original_runtime_list))])

    plt.plot(m_list, [blockDiag_runtime_list[i]/original_runtime_list[i] for i in range(len(original_runtime_list))], label='blockDiag/orig')
    plt.plot(m_list, [permutedBlockDiag_runtime_list[i]/original_runtime_list[i] for i in range(len(original_runtime_list))], label='permutedBlockDiag/orig')
    if PLOT_PERMUTATION_K1:
        plt.plot(m_list, [permutation_runtime_list[i]/original_runtime_list[i] for i in range(len(original_runtime_list))], label='permutation/orig', hold=True)
    plt.xlabel('m')
    plt.ylabel('runtime ratio')
    plt.legend()
    plt.show()
        
