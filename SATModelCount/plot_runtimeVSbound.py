from __future__ import division
from matplotlib import pyplot as plt
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()
from collections import defaultdict
from plot_heatmap import read_file
from plot_heatmap import read_files
import math
from scipy.spatial import ConvexHull
from adjustText import adjust_text
import os
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
PROBLEM_NAME = 'sat-grid-pbl-0010'
permutedBlockDiag_filebase = 'heatmap_result_fireworksWED/permutedBlockDiagDeterministic_speed_REPEATS=%d_%s_duplicates=0_expIdx=' % (REPEATS, PROBLEM_NAME)
regular_filebase = 'heatmap_result_fireworksWED/blockDiagDeterministic_speed_REPEATS=%d_%s_duplicates=0_expIdx=' % (REPEATS, PROBLEM_NAME)
original_filebase = 'heatmap_result_fireworksWED/speed_REPEATS=%d_%s_duplicates=0_expIdx=' % (REPEATS, PROBLEM_NAME)

REPEATS = 10 #repitions of each (m, f) run during an experiment on a single machine
PROBLEM_NAME = 'lang12'
original_filebase = 'heatmap_result_fireworksTIMEOUTcomplete/%s/f_block=1_permute=False_k=0_allOnesConstraint=False_REPEATS=%d_expIdx=' % (PROBLEM_NAME, REPEATS)
regular_filebase = 'heatmap_result_fireworksTIMEOUTcomplete/%s/f_block=1minusF_permute=False_k=None_allOnesConstraint=False_REPEATS=%d_expIdx=' % (PROBLEM_NAME, REPEATS)
permutedBlockDiag_filebase = 'heatmap_result_fireworksTIMEOUTcomplete/%s/f_block=1minusF_permute=True_k=None_allOnesConstraint=False_REPEATS=%d_expIdx=' % (PROBLEM_NAME, REPEATS)


USE_MULTIPLE_FILES = True #aggregate results from multiple files if true
FILE_COUNT = 10 #number of experiments run on possible different machines
PLOT_BLOCK_DIAG = True
PLOT_BLOCK_DIAG_PERMUTED = True
PLOT_PERMUTATION_K1 = False

PLOT_ACTUAL_POINTS = False 
ANNOTATE_PLOTS = False

log_2_Z = { 'c432': 36.1,
            'c499': 41.0,
            'c880': 60.0,
            'c1355': 41.0,
            'c1908': 33.0,
            'c2670': 233,
            'sat-grid-pbl-0010': 78.9,
            'sat-grid-pbl-0015': 180.9,
            'sat-grid-pbl-0020': 318,
            'ra': 951.0,
            'tire-1': 29.4,
            'tire-2': 39.4,
            'tire-3': 37.7,
            'tire-4': 46.6,
            'log-1': 69.0,
            'log-2': 34.9,
            'lang12': -1,
            }


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

def get_lower_bound(num_SAT, T, m, SAT_run_times, UNSAT_run_times, min_confidence=.95):
    '''
    
    Inputs:
    - num_SAT: int, the number of trials where we found a satisfying solution
    - T: int, the number of trials
    - m: int, the number of parity constrains
    - min_confidence: float, the probability with which the bound holds
    - SAT_run_times: list of floats of length num_SAT,
        where each entry corresponds to the runtime for solving that problem instance
    - SAT_run_times: list of floats of length number of UNSAT problems,
        where each entry corresponds to the runtime for solving that problem instance

    Outputs:
    - parallel_bounds: list of floats, bounds on ln(set size) we can compute sorted from worst to best.
        worst bound uses the minimum number of satisfiable solutions (such that num_SAT > -ln(delta)) and has runtime of the slowest of these
        best bound uses all satisfiable solutions and has the runtime of the slowest 
    - parallel_runtimes: list of floats (same length as parallel_bounds), 
        parallel runtime required to compute the corespoding bound, which is the longest runtime of a single satisfiable problem
    - sat_over_trials_parallel: list of floats, (#SAT used)/#trials for each parallel bound
    - satUsed_over_totalSat_parallel: list of floats, (#SAT used)/(total #SAT) for each parallel bound (will be 1 for best parallel bound)
    - sequential_bound: same as the best parallel bound
    - sequential_runtime: sum of runtimes of all satisfiable problems and unsatisfiable problems
    - sat_over_trials_sequential: float #SAT/#trials (same as sat_over_trials_parallel for best parallel bound)
    '''
    parallel_bounds = []
    parallel_runtimes = []
    sat_over_trials_parallel = []
    satUsed_over_totalSat_parallel = []
    delta = 1 - min_confidence
    ln = math.log
    SAT_run_times.sort()
    assert(len(SAT_run_times) == num_SAT)
    for cur_num_SAT in range(1, num_SAT+1):
        if cur_num_SAT > -ln(delta):
            c = cur_num_SAT/T
            kappa = -3 * ln(delta) + math.sqrt(ln(delta) ** 2 - 8 * c * T * ln(delta))
            kappa /= 2 * (c * T + ln(delta))
            cur_bound = m * ln(2) + ln(c) - ln(1 + kappa)
            parallel_bounds.append(cur_bound)
            parallel_runtimes.append(SAT_run_times[cur_num_SAT - 1])
            sat_over_trials_parallel.append(c)
            satUsed_over_totalSat_parallel.append(cur_num_SAT/num_SAT)

    if len(parallel_bounds) > 0:
        sequential_bound = [parallel_bounds[-1]]
        sequential_runtime = [np.sum(SAT_run_times) + np.sum(UNSAT_run_times)]
        sat_over_trials_sequential = [sat_over_trials_parallel[-1]]
        assert(sat_over_trials_sequential[0] == num_SAT/T)
    else:
        sequential_bound = []
        sequential_runtime = []
        sat_over_trials_sequential = []
    return (parallel_bounds, parallel_runtimes, sat_over_trials_parallel, satUsed_over_totalSat_parallel, 
            sequential_bound, sequential_runtime, sat_over_trials_sequential)

def read_files_moreInfo(filename_base, repeats, file_count):
    '''

    Inputs:
    - repeats: (int) the number of experiments run for each f and m value
    - file_count: (int) the number of files containing identical experiment sets
    '''
    #key: (f, m)
    #value: list of all runtimes (len repeats)
    all_runtimes_dict = defaultdict(list)

    #key: (f, m)
    #value: list of runtimes for problems that were found to be satisfiable
    SAT_runtimes = defaultdict(list)

    #key: (f, m)
    #value: list of runtimes for problems that were found to be not satisfiable
    UNSAT_runtimes = defaultdict(list)

    #key: (f, m)
    #value: number of trials performed with these values of f and m (SAT count + UNSAT count + TIMEOUT count)
    num_trials_dict = defaultdict(int)

    #key: (f, m)
    #value: number of SAT trials
    num_SAT_dict = defaultdict(int)

    #key: (f, m)
    #value: number of UNSAT trials
    num_UNSAT_dict = defaultdict(int)  

    f_vals = set()
    m_vals = set()

    #key: (f, m)
    #value: number of trials that timed out
    num_TIMEOUT_dict = defaultdict(int)

    for exp_idx in range(file_count):
        cur_filename = '%s%d.txt' % (filename_base, exp_idx)
        if os.path.isfile(cur_filename):        
            print "reading file:", cur_filename
            reader = open(cur_filename, 'r')
            while True:
                line = reader.readline().split()
                if len(line) < 9:
                    if len(line) == 8 and line[7] == 'None':
                        #problem timed out
                        f = float(line[1])
                        m = int(line[5])   
                        m_vals.add(m)
                        f_vals.add(f)
                        all_runtimes_dict[(f,m)].append(1000) #timeout of 1000*mean_unperturbed_run_time, normalized by mean_unperturbed_run_time
                        num_trials_dict[(f,m)] += 1
                        num_TIMEOUT_dict[(f,m)] += 1
                        continue

                    elif len(line) == 2:
                        assert(line[0] == 'mean_unperturbed_run_time=')
                        mean_unperturbed_run_time = float(line[1])
                        continue                        

                    else:
                        break
        
                f = float(line[1])
                run_time = float(line[8][0:-1])
                m = int(line[5])
                if line[7] == '(True,':
                    num_SAT_dict[(f,m)] += 1
                    SAT_runtimes[(f,m)].append(run_time/(mean_unperturbed_run_time)) #runtime normalized by mean_unperturbed_run_time
                else:
                    assert(line[7] == '(False,')
                    num_UNSAT_dict[(f,m)] += 1
                    UNSAT_runtimes[(f,m)].append(run_time/(mean_unperturbed_run_time)) #runtime normalized by mean_unperturbed_run_time

                m_vals.add(m)
                f_vals.add(f)
                all_runtimes_dict[(f,m)].append(run_time/(mean_unperturbed_run_time)) #runtime normalized by mean_unperturbed_run_time
                num_trials_dict[(f,m)] += 1
        else:
            print "file doesn't exist:", cur_filename

    sorted_m_vals = sorted(m_vals)
    sorted_f_vals = sorted(f_vals)
    print m_vals
    print sorted_m_vals
    num_SAT = np.zeros((len(sorted_f_vals), len(sorted_m_vals)))
    num_UNSAT = np.zeros((len(sorted_f_vals), len(sorted_m_vals)))
    num_TIMEOUT = np.zeros((len(sorted_f_vals), len(sorted_m_vals)))

    for (m_idx, m_val) in enumerate(sorted_m_vals):
        for (f_idx, f_val) in enumerate(sorted_f_vals):
            SAT_runtimes[(f_val, m_val)].sort()
            UNSAT_runtimes[(f_val, m_val)].sort()
            num_SAT[f_idx, m_idx] = num_SAT_dict[(f_val, m_val)]
            num_UNSAT[f_idx, m_idx] = num_UNSAT_dict[(f_val, m_val)]
            num_TIMEOUT[f_idx, m_idx] = num_TIMEOUT_dict[(f_val, m_val)]
            assert(len(UNSAT_runtimes[(f_val, m_val)]) == num_UNSAT[f_idx, m_idx])
            assert(len(SAT_runtimes[(f_val, m_val)]) == num_SAT[f_idx, m_idx])
            #assert(num_trials_dict[(f_val, m_val)] == 100), num_trials_dict[(f_val, m_val)]
            assert(num_trials_dict[(f_val, m_val)] == num_UNSAT[f_idx, m_idx] + num_SAT[f_idx, m_idx] + num_TIMEOUT[f_idx, m_idx])

    return(sorted_m_vals, sorted_f_vals, SAT_runtimes, UNSAT_runtimes, num_SAT, num_trials_dict, all_runtimes_dict)


if __name__=="__main__":

    ##### original randomness #####

    if USE_MULTIPLE_FILES: 
        (sorted_m_vals, sorted_f_vals, SAT_runtimes, UNSAT_runtimes, num_SAT, num_trials_dict, all_runtimes_dict) = read_files_moreInfo(filename_base=original_filebase, repeats=REPEATS, file_count=FILE_COUNT)
    else:
        (sorted_m_vals, sorted_f_vals, SAT_runtimes, UNSAT_runtimes, num_SAT, num_trials_dict) = read_file(random_file, repeats=REPEATS)

    print '-'*30, 'original randomness', '-'*30
    all_original_parallel_bounds = []
    all_original_parallel_runtimes = []
    all_original_sequential_bounds = []
    all_original_sequential_runtimes = []
    all_original_num_trials = {} #number of trials for every m, f combination
    parallel_orig_M_F = []
    sequential_orig_M_F = []
    orig_sat_over_trials_parallel = []
    orig_satUsed_over_totalSat_parallel = []
    orig_sat_over_trials_sequential = []
    for (m_idx, m_val) in enumerate(sorted_m_vals):
        for (f_idx, f_val) in enumerate(sorted_f_vals):
            print "m =", m_val, "f =", f_val        
            assert(float(int(int(num_SAT[f_idx, m_idx]))) == num_SAT[f_idx, m_idx])
            (parallel_bounds, parallel_runtimes, sat_over_trials_parallel, satUsed_over_totalSat_parallel, \
                sequential_bound, sequential_runtime, sat_over_trials_sequential) = get_lower_bound(num_SAT=int(num_SAT[f_idx, m_idx]), T=num_trials_dict[f_val, m_val], m=m_val, SAT_run_times=SAT_runtimes[f_val, m_val], UNSAT_run_times=UNSAT_runtimes[f_val, m_val])
            all_original_parallel_bounds.extend(parallel_bounds)
            all_original_parallel_runtimes.extend(parallel_runtimes)
            print 'len(parallel_runtimes) =', len(parallel_runtimes)
            all_original_sequential_bounds.extend(sequential_bound)
            all_original_sequential_runtimes.extend(sequential_runtime)
            all_original_num_trials[(f_val, m_val)] = num_trials_dict[f_val, m_val]
            parallel_orig_M_F.extend([(m_val, f_val) for i in range(len(parallel_bounds))])
            sequential_orig_M_F.append((m_val, f_val))
            orig_sat_over_trials_parallel.extend(sat_over_trials_parallel)
            orig_satUsed_over_totalSat_parallel.extend(satUsed_over_totalSat_parallel)
            orig_sat_over_trials_sequential.extend(sat_over_trials_sequential)


    ##### block diagonal + randomness #####
    if USE_MULTIPLE_FILES:
        (sorted_m_vals, sorted_f_vals, SAT_runtimes, UNSAT_runtimes, num_SAT, num_trials_dict, all_runtimes_dict) = read_files_moreInfo(filename_base=regular_filebase, repeats=REPEATS, file_count=FILE_COUNT)
    else:
        (sorted_m_vals, sorted_f_vals, SAT_runtimes, UNSAT_runtimes, num_SAT, num_trials_dict) = read_file(regular_file, repeats=REPEATS)

    print '-'*30, 'block diag', '-'*30
    all_blockDiag_parallel_bounds = []
    all_blockDiag_parallel_runtimes = []
    all_blockDiag_sequential_bounds = []
    all_blockDiag_sequential_runtimes = []
    all_blockDiag_num_trials = {} #number of trials for every m, f combination
    parallel_blockDiag_M_F = []
    sequential_blockDiag_M_F = []
    blockDiag_sat_over_trials_parallel = []
    blockDiag_satUsed_over_totalSat_parallel = []
    blockDiag_sat_over_trials_sequential = []

    for (m_idx, m_val) in enumerate(sorted_m_vals):
        for (f_idx, f_val) in enumerate(sorted_f_vals):
            print "m =", m_val, "f =", f_val        
            assert(float(int(int(num_SAT[f_idx, m_idx]))) == num_SAT[f_idx, m_idx])
            (parallel_bounds, parallel_runtimes, sat_over_trials_parallel, satUsed_over_totalSat_parallel, \
                sequential_bound, sequential_runtime, sat_over_trials_sequential) = get_lower_bound(num_SAT=int(num_SAT[f_idx, m_idx]), T=num_trials_dict[f_val, m_val], m=m_val, SAT_run_times=SAT_runtimes[f_val, m_val], UNSAT_run_times=UNSAT_runtimes[f_val, m_val])
            all_blockDiag_parallel_bounds.extend(parallel_bounds)
            all_blockDiag_parallel_runtimes.extend(parallel_runtimes)
            all_blockDiag_sequential_bounds.extend(sequential_bound)
            all_blockDiag_sequential_runtimes.extend(sequential_runtime)
            all_blockDiag_num_trials[(f_val, m_val)] = num_trials_dict[f_val, m_val]
            parallel_blockDiag_M_F.extend([(m_val, f_val) for i in range(len(parallel_bounds))])
            sequential_blockDiag_M_F.append((m_val, f_val))
            blockDiag_sat_over_trials_parallel.extend(sat_over_trials_parallel)
            blockDiag_satUsed_over_totalSat_parallel.extend(satUsed_over_totalSat_parallel)
            blockDiag_sat_over_trials_sequential.extend(sat_over_trials_sequential)



    ##### permuted block diagonal + randomness #####
    if USE_MULTIPLE_FILES:
        (sorted_m_vals, sorted_f_vals, SAT_runtimes, UNSAT_runtimes, num_SAT, num_trials_dict, all_runtimes_dict) = read_files_moreInfo(filename_base=permutedBlockDiag_filebase, repeats=REPEATS, file_count=FILE_COUNT)
    else:
        (sorted_m_vals, sorted_f_vals, SAT_runtimes, UNSAT_runtimes, num_SAT, num_trials_dict) = read_file(permutedBlockDiag_file, repeats=REPEATS)    

    print '-'*30, 'permuted block diag', '-'*30
    all_permutedBlockDiag_parallel_bounds = []
    all_permutedBlockDiag_parallel_runtimes = []
    all_permutedBlockDiag_sequential_bounds = []
    all_permutedBlockDiag_sequential_runtimes = []
    all_permutedBlockDiag_num_trials = {} #number of trials for every m, f combination
    parallel_permutedBlockDiag_M_F = []
    sequential_permutedBlockDiag_M_F = []
    permutedBlockDiag_sat_over_trials_parallel = []
    permutedBlockDiag_satUsed_over_totalSat_parallel = []
    permutedBlockDiag_sat_over_trials_sequential = []

    for (m_idx, m_val) in enumerate(sorted_m_vals):
        for (f_idx, f_val) in enumerate(sorted_f_vals):
            print "m =", m_val, "f =", f_val
            assert(float(int(int(num_SAT[f_idx, m_idx]))) == num_SAT[f_idx, m_idx])
            (parallel_bounds, parallel_runtimes, sat_over_trials_parallel, satUsed_over_totalSat_parallel, \
                sequential_bound, sequential_runtime, sat_over_trials_sequential) = get_lower_bound(num_SAT=int(num_SAT[f_idx, m_idx]), T=num_trials_dict[f_val, m_val], m=m_val, SAT_run_times=SAT_runtimes[f_val, m_val], UNSAT_run_times=UNSAT_runtimes[f_val, m_val])
            all_permutedBlockDiag_parallel_bounds.extend(parallel_bounds)
            all_permutedBlockDiag_parallel_runtimes.extend(parallel_runtimes)
            all_permutedBlockDiag_sequential_bounds.extend(sequential_bound)
            all_permutedBlockDiag_sequential_runtimes.extend(sequential_runtime)
            all_permutedBlockDiag_num_trials[(f_val, m_val)] = num_trials_dict[f_val, m_val]
            parallel_permutedBlockDiag_M_F.extend([(m_val, f_val) for i in range(len(parallel_bounds))])
            sequential_permutedBlockDiag_M_F.append((m_val, f_val))
            permutedBlockDiag_sat_over_trials_parallel.extend(sat_over_trials_parallel)
            permutedBlockDiag_satUsed_over_totalSat_parallel.extend(satUsed_over_totalSat_parallel)
            permutedBlockDiag_sat_over_trials_sequential.extend(sat_over_trials_sequential)

    if PLOT_ACTUAL_POINTS:
        plt.scatter(all_original_parallel_runtimes, all_original_parallel_bounds, marker='+', c='b')
        plt.scatter(all_blockDiag_parallel_runtimes, all_blockDiag_parallel_bounds, marker='+', c='r')
        plt.scatter(all_permutedBlockDiag_parallel_runtimes, all_permutedBlockDiag_parallel_bounds, marker='+', c='g')
        plt.xlabel('parallel runtime (units: mean unperturbed runtime on machine averaged over 10 trials)')
        plt.ylabel('bound on ln(set size)')
        plt.legend()
    #    plt.show()

    PLOT_PARALLEL_CONVEX_HULLS = True
    if PLOT_PARALLEL_CONVEX_HULLS:
        #plot convex hull of points from original method
        original_points = np.array(zip(all_original_parallel_runtimes,all_original_parallel_bounds))
        original_hull = ConvexHull(original_points)

        for (idx, simplex) in enumerate(original_hull.simplices):
            print "simplex:", simplex
            print "type(simplex):", type(simplex)
            if idx == 0: #add the label to the legend
                plt.plot(original_points[simplex, 0], original_points[simplex, 1], '*--', c='b', label='original')
            else:
                plt.plot(original_points[simplex, 0], original_points[simplex, 1], '*--', c='b')
            if ANNOTATE_PLOTS:
                m_val0 = parallel_orig_M_F[simplex[0]][0]
                f_val0 = parallel_orig_M_F[simplex[0]][1]

                m_val1 = parallel_orig_M_F[simplex[1]][0]
                f_val1 = parallel_orig_M_F[simplex[1]][1]
                plt.annotate(
                    'm:%d,f:%.3f,T:%d,c:%.2f,%%c:%.2f' % (m_val0, f_val0, all_original_num_trials[(f_val0, m_val0)], \
                        orig_sat_over_trials_parallel[simplex[0]], orig_satUsed_over_totalSat_parallel[simplex[0]]),
                    xy=(original_points[simplex[0], 0], original_points[simplex[0], 1]), xytext=(-20, 20),
                    textcoords='offset points', ha='right', va='bottom', color='b',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0', color='b'))
                plt.annotate(
                    'm:%d,f:%.3f,T:%d,c:%.2f,%%c:%.2f'% (m_val1, f_val1, all_original_num_trials[(f_val1, m_val1)], \
                        orig_sat_over_trials_parallel[simplex[1]], orig_satUsed_over_totalSat_parallel[simplex[1]]),
                    xy=(original_points[simplex[1], 0], original_points[simplex[1], 1]), xytext=(-20, 20),
                    textcoords='offset points', ha='right', va='bottom', color='b',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0', color='b'))
        
        #plot convex hull of points from block diagonal method
        blockDiag_points = np.array(zip(all_blockDiag_parallel_runtimes,all_blockDiag_parallel_bounds))
        blockDiag_hull = ConvexHull(blockDiag_points)
        for (idx, simplex) in enumerate(blockDiag_hull.simplices):
            if idx == 0:
                plt.plot(blockDiag_points[simplex, 0], blockDiag_points[simplex, 1], '*--', c='r', label='block diagonal')
            else:
                plt.plot(blockDiag_points[simplex, 0], blockDiag_points[simplex, 1], '*--', c='r')
            if ANNOTATE_PLOTS:
                m_val0 = parallel_blockDiag_M_F[simplex[0]][0]
                f_val0 = parallel_blockDiag_M_F[simplex[0]][1]

                m_val1 = parallel_blockDiag_M_F[simplex[1]][0]
                f_val1 = parallel_blockDiag_M_F[simplex[1]][1]
                plt.annotate(
                    'm:%d,f:%.3f,T:%d,c:%.2f,%%c:%.2f' % (m_val0, f_val0, all_blockDiag_num_trials[(f_val0, m_val0)], \
                        blockDiag_sat_over_trials_parallel[simplex[0]], blockDiag_satUsed_over_totalSat_parallel[simplex[0]]),
                    xy=(blockDiag_points[simplex[0], 0], blockDiag_points[simplex[0], 1]), xytext=(-20, 20),
                    textcoords='offset points', ha='right', va='bottom', color='r',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0', color='r'))
                plt.annotate(
                    'm:%d,f:%.3f,T:%d,c:%.2f,%%c:%.2f'% (m_val1, f_val1, all_blockDiag_num_trials[(f_val1, m_val1)], \
                        blockDiag_sat_over_trials_parallel[simplex[1]], blockDiag_satUsed_over_totalSat_parallel[simplex[1]]),
                    xy=(blockDiag_points[simplex[1], 0], blockDiag_points[simplex[1], 1]), xytext=(-20, 20),
                    textcoords='offset points', ha='right', va='bottom', color='r',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0', color='r'))
        
        #plot convex hull of points from permuted block diagonal method
        permutedBlockDiag_points = np.array(zip(all_permutedBlockDiag_parallel_runtimes, all_permutedBlockDiag_parallel_bounds))
        permutedBlockDiag_hull = ConvexHull(permutedBlockDiag_points)
        for (idx, simplex) in enumerate(permutedBlockDiag_hull.simplices):
            if idx == 0:
                plt.plot(permutedBlockDiag_points[simplex, 0], permutedBlockDiag_points[simplex, 1], '*--', c='g', label='permuted block diagonal')
            else:
                plt.plot(permutedBlockDiag_points[simplex, 0], permutedBlockDiag_points[simplex, 1], '*--', c='g')
            if ANNOTATE_PLOTS:
                m_val0 = parallel_permutedBlockDiag_M_F[simplex[0]][0]
                f_val0 = parallel_permutedBlockDiag_M_F[simplex[0]][1]

                m_val1 = parallel_permutedBlockDiag_M_F[simplex[1]][0]
                f_val1 = parallel_permutedBlockDiag_M_F[simplex[1]][1]
                plt.annotate(
                    'm:%d,f:%.3f,T:%d,c:%.2f,%%c:%.2f' % (m_val0, f_val0, all_permutedBlockDiag_num_trials[(f_val0, m_val0)], \
                        permutedBlockDiag_sat_over_trials_parallel[simplex[0]], permutedBlockDiag_satUsed_over_totalSat_parallel[simplex[0]]),
                    xy=(permutedBlockDiag_points[simplex[0], 0], permutedBlockDiag_points[simplex[0], 1]), xytext=(-20, 20),
                    textcoords='offset points', ha='right', va='bottom', color='g',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0', color='g'))
                plt.annotate(
                    'm:%d,f:%.3f,T:%d,c:%.2f,%%c:%.2f'% (m_val1, f_val1, all_permutedBlockDiag_num_trials[(f_val1, m_val1)], \
                        permutedBlockDiag_sat_over_trials_parallel[simplex[1]], permutedBlockDiag_satUsed_over_totalSat_parallel[simplex[1]]),
                    xy=(permutedBlockDiag_points[simplex[1], 0], permutedBlockDiag_points[simplex[1], 1]), xytext=(-20, 20),
                    textcoords='offset points', ha='right', va='bottom', color='g',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0', color='g'))

        plt.axhline(y=math.log(2)*log_2_Z[PROBLEM_NAME], color='y', label='ground truth ln(set size)') 
        plt.xlabel('parallel runtime (units: mean unperturbed runtime on machine averaged over 10 trials)')
        plt.ylabel('bound on ln(set size)')
        plt.title('minTrialsOrig=%d, minTrialsBlockDiag=%d, minTrialsPermBlockDiag=%d' % (all_original_num_trials[min(all_original_num_trials, key = lambda x: all_original_num_trials.get(x))],
                                                                                          all_blockDiag_num_trials[min(all_blockDiag_num_trials, key = lambda x: all_blockDiag_num_trials.get(x))],
                                                                                          all_permutedBlockDiag_num_trials[min(all_permutedBlockDiag_num_trials, key = lambda x: all_permutedBlockDiag_num_trials.get(x))]))
        plt.legend()    
        plt.show()
        

    PLOT_SEQUENTIAL_CONVEX_HULLS = True
    if PLOT_SEQUENTIAL_CONVEX_HULLS:
        if ANNOTATE_PLOTS:
            texts = []
        #plot convex hull of points from original method
        original_points = np.array(zip(all_original_sequential_runtimes,all_original_sequential_bounds))
        original_hull = ConvexHull(original_points)
        for (idx, simplex) in enumerate(original_hull.simplices):
            if idx == 0: #add the label to the legend
                plt.plot(original_points[simplex, 0], original_points[simplex, 1], '*--', c='b', label='original')
            else:
                plt.plot(original_points[simplex, 0], original_points[simplex, 1], '*--', c='b')
            if ANNOTATE_PLOTS:
                m_val0 = sequential_orig_M_F[simplex[0]][0]
                f_val0 = sequential_orig_M_F[simplex[0]][1]

                m_val1 = sequential_orig_M_F[simplex[1]][0]
                f_val1 = sequential_orig_M_F[simplex[1]][1]

                plt.annotate(
                    'm:%d,f:%.3f,T:%d,c:%.2f' % (m_val0, f_val0, all_original_num_trials[(f_val0, m_val0)], orig_sat_over_trials_sequential[simplex[0]]),
                    xy=(original_points[simplex[0], 0], original_points[simplex[0], 1]), xytext=(-20, 20),
                    textcoords='offset points', ha='right', va='bottom', color='b',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0', color='b'))
                plt.annotate(
                    'm:%d,f:%.3f,T:%d,c:%.2f'% (m_val1, f_val1, all_original_num_trials[(f_val1, m_val1)], orig_sat_over_trials_sequential[simplex[1]]),
                    xy=(original_points[simplex[1], 0], original_points[simplex[1], 1]), xytext=(-20, 20),
                    textcoords='offset points', ha='right', va='bottom', color='b',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0', color='b'))
        
    #            texts.append(plt.text(original_points[simplex[0], 0], 
    #                         original_points[simplex[0], 1], 
    #                         'm:%d,f:%.3f,T:%d,c:%.2f' % (m_val0, f_val0, all_original_num_trials[(f_val0, m_val0)], orig_sat_over_trials_sequential[simplex[0]])))
    #            texts.append(plt.text(original_points[simplex[1], 0], 
    #                         original_points[simplex[1], 1], 
    #                         'm:%d,f:%.3f,T:%d,c:%.2f' % (m_val0, f_val0, all_original_num_trials[(f_val0, m_val0)], orig_sat_over_trials_sequential[simplex[1]])))
        
        #plot convex hull of points from block diagonal method
        blockDiag_points = np.array(zip(all_blockDiag_sequential_runtimes,all_blockDiag_sequential_bounds))
        blockDiag_hull = ConvexHull(blockDiag_points)
        for (idx, simplex) in enumerate(blockDiag_hull.simplices):
            if idx == 0:
                plt.plot(blockDiag_points[simplex, 0], blockDiag_points[simplex, 1], '*--', c='r', label='block diagonal')
            else:
                plt.plot(blockDiag_points[simplex, 0], blockDiag_points[simplex, 1], '*--', c='r')
        
        #plot convex hull of points from permuted block diagonal method
        permutedBlockDiag_points = np.array(zip(all_permutedBlockDiag_sequential_runtimes, all_permutedBlockDiag_sequential_bounds))
        permutedBlockDiag_hull = ConvexHull(permutedBlockDiag_points)
        for (idx, simplex) in enumerate(permutedBlockDiag_hull.simplices):
            if idx == 0:
                plt.plot(permutedBlockDiag_points[simplex, 0], permutedBlockDiag_points[simplex, 1], '*--', c='g', label='permuted block diagonal')
            else:
                plt.plot(permutedBlockDiag_points[simplex, 0], permutedBlockDiag_points[simplex, 1], '*--', c='g')
        
        plt.axhline(y=math.log(2)*log_2_Z[PROBLEM_NAME], color='y', label='ground truth ln(set size)')     
        plt.xlabel('sequential runtime (units: mean unperturbed runtime on machine averaged over 10 trials)')
        plt.ylabel('bound on ln(set size)')
        plt.title('minTrialsOrig=%d, minTrialsBlockDiag=%d, minTrialsPermBlockDiag=%d' % (all_original_num_trials[min(all_original_num_trials, key = lambda x: all_original_num_trials.get(x))],
                                                                                          all_blockDiag_num_trials[min(all_blockDiag_num_trials, key = lambda x: all_blockDiag_num_trials.get(x))],
                                                                                          all_permutedBlockDiag_num_trials[min(all_permutedBlockDiag_num_trials, key = lambda x: all_permutedBlockDiag_num_trials.get(x))]))
        
        plt.legend()  
    #    adjust_text(texts,only_move={'points':'y', 'text':'y'}, force_points=0.15,
    #            arrowprops=dict(arrowstyle="->", color='r', lw=0.5))  
        plt.show()

