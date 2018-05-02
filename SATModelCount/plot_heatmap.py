from __future__ import division
from matplotlib import pyplot as plt
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()
from collections import defaultdict
import os
import math
from file_helpers import read_files_moreInfo_newFormat

#random_file = 'SATModelCount/result/speed20.txt'
#regular_file = 'SATModelCount/result/rspeed20.txt'
#random_file = 'result/speed20.txt'
#regular_file = 'result/rspeed20.txt'

REPEATS = 20
random_file = 'heatmap_result_moreModels/speed_REPEATS=%d_tire-1.txt' % REPEATS
#regular_file = 'heatmap_result/rspeed_REPEATS=%d_tire-1.txt' % REPEATS
regular_file = 'heatmap_result_moreModels/rspeed_REPEATS=%d_tire-1.txt' % REPEATS
permutation_file = 'heatmap_result_moreModels/pspeed_REPEATS=%d_tire-1.txt' % REPEATS


REPEATS = 10
random_file = 'heatmap_result_fireworks/speed_REPEATS=%d_c432_duplicates=0_expIdx=9.txt' % REPEATS
regular_file = 'heatmap_result_duplicateVars_onlyOrig/rspeed_REPEATS=%d_c432_duplicates=1.txt' % REPEATS
permutation_file = 'heatmap_result_duplicateVars_onlyOrig/pspeed_REPEATS=%d_c432_duplicates=1.txt' % REPEATS


#REPEATS = 10
#permutedBlockDiag_filebase = 'heatmap_result_fireworks/permutedBlockDiag_speed_REPEATS=%d_c499_duplicates=0_expIdx=' % REPEATS
#regular_filebase = 'heatmap_result_fireworks/rspeed_REPEATS=%d_c499_duplicates=0_expIdx=' % REPEATS
#original_filebase = 'heatmap_result_fireworks/speed_REPEATS=%d_c499_duplicates=0_expIdx=' % REPEATS
#

REPEATS = 10
PROBLEM_NAME = 'c432'
permutedBlockDiag_filebase = 'heatmap_result_fireworksWED/permutedBlockDiagDeterministic_speed_REPEATS=%d_%s_duplicates=0_expIdx=' % (REPEATS, PROBLEM_NAME)
regular_filebase = 'heatmap_result_fireworksWED/blockDiagDeterministic_speed_REPEATS=%d_%s_duplicates=0_expIdx=' % (REPEATS, PROBLEM_NAME)
original_filebase = 'heatmap_result_fireworksWED/speed_REPEATS=%d_%s_duplicates=0_expIdx=' % (REPEATS, PROBLEM_NAME)

#REPEATS = 10
#PROBLEM_NAME = 'c432'
#original_filebase = 'heatmap_result_fireworksTIMEOUTallFcorrected/%s/f_block=1_permute=False_k=0_allOnesConstraint=False_adjustF_REPEATS=%d_expIdx=' % (PROBLEM_NAME, REPEATS)
#regular_filebase = 'heatmap_result_fireworksTIMEOUTallF/%s/f_block=1minusF_permute=True_k=maxConstant_allOnesConstraint=False_REPEATS=%d_expIdx=' % (PROBLEM_NAME, REPEATS)
#permutedBlockDiag_filebase = 'heatmap_result_fireworksTIMEOUTcomplete/%s/f_block=1minusF_permute=True_k=3_allOnesConstraint=False_REPEATS=%d_expIdx=' % (PROBLEM_NAME, REPEATS)
#
#REPEATS = 10
#PROBLEM_NAME = 'hypercube2'
#original_filebase = 'heatmap_result_fireworksTIMEOUT_fDensity/%s/f_block=1_permute=False_k=0_allOnesConstraint=False_adjustF=True_REPEATS=%d_expIdx=' % (PROBLEM_NAME, REPEATS)
#regular_filebase = 'heatmap_result_fireworksTIMEOUT_fDensity/%s/f_block=1minusF_permute=True_k=maxConstant_allOnesConstraint=False_adjustF=True_REPEATS=%d_expIdx=' % (PROBLEM_NAME, REPEATS)
#permutedBlockDiag_filebase = 'heatmap_result_fireworksTIMEOUTcomplete/%s/f_block=1minusF_permute=True_k=3_allOnesConstraint=False_REPEATS=%d_expIdx=' % (PROBLEM_NAME, REPEATS)

####REPEATS = 10 #repitions of each (m, f) run during an experiment on a single machine
####PROBLEM_NAME = 'c432'
####original_filebase = 'heatmap_result_fireworksTIMEOUT_3_9_secondCopy/%s/f_block=1_permute=False_k=0_allOnesConstraint=False_adjustF=True_REPEATS=%d_expIdx=' % (PROBLEM_NAME, REPEATS)
####permutedBlockDiag_filebase = 'heatmap_result_fireworksTIMEOUT_3_9_secondCopy/%s/f_block=1minusF_permute=True_k=maxConstant_allOnesConstraint=False_adjustF=True_REPEATS=%d_expIdx=' % (PROBLEM_NAME, REPEATS)


#REPEATS = 10
#PROBLEM_NAME = 'c432'
#original_filebase = 'heatmap_result_fireworksTIMEOUT/%s/f_block=1_permute=False_k=0_allOnesConstraint=False_REPEATS=%d_expIdx=' % (PROBLEM_NAME, REPEATS)
#regular_filebase = 'heatmap_result_fireworksTIMEOUT/%s/f_block=1minusF_permute=False_k=None_allOnesConstraint=False_REPEATS=%d_expIdx=' % (PROBLEM_NAME, REPEATS)
#permutedBlockDiag_filebase = 'heatmap_result_fireworksTIMEOUT/%s/f_block=1minusF_permute=True_k=None_allOnesConstraint=False_REPEATS=%d_expIdx=' % (PROBLEM_NAME, REPEATS)


USE_MULTIPLE_FILES = True #aggregate results from multiple files if true
#NEW_FORMAT = True #extra information is stored in SAT results files
FILE_COUNT = 10
PLOT_BLOCK_DIAG = True
PLOT_BLOCK_DIAG_PERMUTED = True
PLOT_PERMUTATION_K1 = False
SHOW_HEATMAP_VALS = True
MAX_TIMEOUT_MULTIPLE = 100 #run at max MAX_TIMEOUT_MULTIPLE*unperturbed runtime
#convert_origF_to_TriF = False   

def read_file(filename, repeats):
    '''

    Inputs:
    - repeats: (int) the number of experiments run for each f and m value
    '''
    print "reading file:", filename
    reader = open(filename, 'r')

    #key: (f, m)
    #value: list of all runtimes (len repeats)
    all_runtimes = defaultdict(list)
    #key: (f, m)
    #value: list of 0/1 values (len repeats)
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

def round_to_1(x):
    return round(x, -int(math.floor(math.log10(abs(x)))))

def round_to_2(x):
    return round(x, -int(math.floor(math.log10(abs(x))))+1)    

def get_n(problem_name):
    problem_filename = "/Users/jkuck/research/winter_2018/low_density_parity_checks/SAT_problems_cnf/%s.cnf" % problem_name

    f = open(problem_filename, 'r')
    for line in f:
        if line[0] == 'p': #parameters line
            params = line.split()
            assert(params[1] == 'cnf') #we should be reading an unweighted SAT problem
            nbvar = int(params[2]) #number of variables
            break            
        else:
            assert(line[0] == 'c')
    f.close()

    return nbvar

def read_files(filename_base, repeats, file_count, convert_origF_to_TriF=False, f_geq=-1, printF_below = 99, m_below=np.inf):
    '''

    Inputs:
    - repeats: (int) the number of experiments run for each f and m value
    - file_count: (int) the number of files containing identical experiment sets
    '''
    #key: (f, m)
    #value: list of all runtimes (len repeats)
    all_runtimes = defaultdict(list)
    #key: (f, m)
    #value: list of 0/1 values (len repeats)
    #   1: problem was found to be satisfiable
    #   0: problem was not found to be satisfiable (UNSAT or timeout)
    problem_satisfied = defaultdict(list)

    #key: (f, m)
    #value: list of 0/1 values (len repeats)
    #   1: problem was found to be unsatisfiable
    #   0: problem was not found to be unsatisfiable (SAT or timeout)
    problem_unsatisfied = defaultdict(list)    
    f_vals = set()
    m_vals = set()
    n = get_n(PROBLEM_NAME)

    mean_unperturbed_run_time = 999999999
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
                        if f>=printF_below or f<f_geq or m>=m_below:
                            continue

                        if convert_origF_to_TriF:
                            n=196
                            k = math.floor(n/m)                     
                            tri_f = (f*n - k)/(n - 2*k)
                            if tri_f < .1:
                                tri_f = round_to_1(tri_f)
                            else:
                                tri_f = round_to_2(tri_f)
                            print 'origF =', f, 'tri_f =', tri_f, 'k =', k
                            f=tri_f
                        m_vals.add(m)
                        f_vals.add(f)
                        all_runtimes[(f,m)].append(MAX_TIMEOUT_MULTIPLE)
                        problem_satisfied[(f,m)].append(0)
                        problem_unsatisfied[(f,m)].append(0)
                        continue
                                     
                    elif len(line) == 2:
                        assert(line[0] == 'mean_unperturbed_run_time=')
                        mean_unperturbed_run_time = float(line[1])
                        continue                        
                    else:
                        assert(len(line) == 0)
                        break
        
                f = float(line[1])
                run_time = float(line[8][0:-1])
                norm_runtime = run_time/mean_unperturbed_run_time
                m = int(line[5])
                if f>=printF_below or f<f_geq or m>=m_below:
                    continue                
                if convert_origF_to_TriF:
                    k = math.floor(n/m)                     
                    unrounded_tri_f = (f*n - k)/(n - 2*k)
                    if unrounded_tri_f < .1:
                        tri_f = round_to_1(unrounded_tri_f)
                    else:
                        tri_f = round_to_2(unrounded_tri_f)
                    print 'origF =', f, 'unrounded_tri_f=', unrounded_tri_f, 'tri_f =', tri_f, 'k =', k
                    f=tri_f

                if norm_runtime > MAX_TIMEOUT_MULTIPLE:
                    norm_runtime = MAX_TIMEOUT_MULTIPLE
                    satisfied = 0
                    unsatisfied = 0
                elif line[7] == '(True,':
                    satisfied = 1
                    unsatisfied = 0
                else:
                    assert(line[7] == '(False,')
                    satisfied = 0
                    unsatisfied = 1

        
                m_vals.add(m)
                f_vals.add(f)
                all_runtimes[(f,m)].append(norm_runtime)
                problem_satisfied[(f,m)].append(satisfied)
                problem_unsatisfied[(f,m)].append(unsatisfied)
        else:
            print "file doesn't exist", cur_filename


    sorted_m_vals = sorted(m_vals)
    sorted_f_vals = sorted(f_vals)
    print m_vals
    print sorted_m_vals
    mean_runtimes = np.zeros((len(sorted_f_vals), len(sorted_m_vals)))
    fraction_SAT = np.zeros((len(sorted_f_vals), len(sorted_m_vals)))
    fraction_UNSAT = np.zeros((len(sorted_f_vals), len(sorted_m_vals)))

    for (m_idx, m_val) in enumerate(sorted_m_vals):
        for (f_idx, f_val) in enumerate(sorted_f_vals):
#            assert(len(all_runtimes[(f_val, m_val)]) == repeats or len(all_runtimes[(f_val, m_val)]) == repeats-1), (len(all_runtimes[(f_val, m_val)]), f_val, m_val)
#            assert(len(problem_satisfied[(f_val, m_val)]) == repeats or len(problem_satisfied[(f_val, m_val)]) == repeats-1), (len(problem_satisfied[(f_val, m_val)]), f_val, m_val)
            mean_runtimes[f_idx, m_idx] = np.mean(all_runtimes[(f_val, m_val)])
            if m_val == 69 and f_val == .04:
                print all_runtimes[(f_val, m_val)]
                print np.mean(all_runtimes[(f_val, m_val)])
                print len(all_runtimes[(f_val, m_val)])
                print '%'*50
            fraction_SAT[f_idx, m_idx] = np.mean(problem_satisfied[(f_val, m_val)])
            fraction_UNSAT[f_idx, m_idx] = np.mean(problem_unsatisfied[(f_val, m_val)])
    return(sorted_m_vals, sorted_f_vals, mean_runtimes, fraction_SAT, fraction_UNSAT)


if __name__=="__main__":

    DEMO_HEATMAP = False
    if DEMO_HEATMAP:
        uniform_data = np.random.rand(10, 12)
        ax = sns.heatmap(uniform_data, xticklabels=[2*i for i in range(12)], yticklabels=[3*i for i in range(10)])
        ax.invert_yaxis()
        plt.show() 
        sleep(2)

    ##### original randomness #####
    if USE_MULTIPLE_FILES: 
        (sorted_m_vals, sorted_f_vals, mean_runtimes_orig, fraction_SAT_orig, fraction_UNSAT) = \
            read_files(filename_base=original_filebase, repeats=REPEATS, file_count=FILE_COUNT, convert_origF_to_TriF=True, f_geq=.001, printF_below=.5, m_below=99)
#        if NEW_FORMAT:
#            (sorted_m_vals, sorted_f_vals, SAT_runtimes, UNSAT_runtimes, num_SAT, num_trials_dict, all_runtimes_dict, f_prime_dict, k_dict) = \
#            read_files_moreInfo_newFormat(filename_base=original_filebase, repeats=REPEATS, file_count=FILE_COUNT)        
#        else:
#            (sorted_m_vals, sorted_f_vals, SAT_runtimes, UNSAT_runtimes, num_SAT, num_trials_dict, all_runtimes_dict) = \
#            read_files_moreInfo(filename_base=filename_base, repeats=REPEATS, file_count=FILE_COUNT)


    else:
        (sorted_m_vals, sorted_f_vals, mean_runtimes_orig, fraction_SAT_orig, fraction_UNSAT) = read_file(random_file, repeats=REPEATS)

    fraction_TIMEOUT = np.zeros(fraction_SAT_orig.shape)
    for r in range(fraction_SAT_orig.shape[0]):
        for c in range(fraction_SAT_orig.shape[1]):
                fraction_TIMEOUT[r,c] = 1 - fraction_SAT_orig[r,c] - fraction_UNSAT[r,c]
                #assert(fraction_TIMEOUT[r,c] >= 0 and fraction_TIMEOUT[r,c] <= 1.0)



##    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
##    ax1.set_title('Mean runtimes over %d runs, original randomness'%REPEATS)
##    ax2.set_title('Fraction satisfied problems over %d runs, original randomness'%REPEATS)
##    ax3.set_title('Fraction unsatisfied problems over %d runs, original randomness'%REPEATS)
##    plt.xlabel('m')
##    plt.ylabel('f')
##    sns.heatmap(mean_runtimes_orig, annot=SHOW_HEATMAP_VALS, xticklabels=sorted_m_vals, yticklabels=sorted_f_vals, ax=ax1)
##    sns.heatmap(fraction_SAT_orig, annot=SHOW_HEATMAP_VALS, xticklabels=sorted_m_vals, yticklabels=sorted_f_vals, ax=ax2)
##    sns.heatmap(fraction_UNSAT, annot=SHOW_HEATMAP_VALS, xticklabels=sorted_m_vals, yticklabels=sorted_f_vals, ax=ax3)
##
##    # Fine-tune figure; make subplots close to each other and hide x ticks for
##    # all but bottom plot.
##    f.subplots_adjust(hspace=0)
##    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
##    ax1.invert_yaxis()
##    ax2.invert_yaxis()
##    ax3.invert_yaxis()
##    plt.show()

    plt.figure(1)
    plt.subplot(211)
    ax = sns.heatmap(mean_runtimes_orig, annot=SHOW_HEATMAP_VALS, xticklabels=sorted_m_vals, yticklabels=sorted_f_vals)
    plt.title('Mean runtimes over %d runs, original randomness'%REPEATS)
    plt.xlabel('m')
    plt.ylabel('f')
    ax.invert_yaxis()
    #plt.show()

    plt.subplot(212)
    ax = sns.heatmap(fraction_SAT_orig, annot=SHOW_HEATMAP_VALS, xticklabels=sorted_m_vals, yticklabels=sorted_f_vals)
    plt.title('Fraction satisfied problems over %d runs, original randomness'%REPEATS)
    plt.xlabel('m')
    plt.ylabel('f')
    ax.invert_yaxis()
    plt.show()   

    if PLOT_BLOCK_DIAG:
        ##### block diagonal + randomness #####
        if USE_MULTIPLE_FILES:
            (sorted_m_vals, sorted_f_vals, mean_runtimes_diag, fraction_SAT_diag, fraction_UNSAT) = \
            read_files(filename_base=regular_filebase, repeats=REPEATS, file_count=FILE_COUNT, f_geq=.001, printF_below=.5, m_below=99)
        else:
            (sorted_m_vals, sorted_f_vals, mean_runtimes_diag, fraction_SAT_diag, fraction_UNSAT) = read_file(regular_file, repeats=REPEATS)
        
        plt.figure(1)
        plt.subplot(211)
        ax = sns.heatmap(mean_runtimes_diag, annot=SHOW_HEATMAP_VALS, xticklabels=sorted_m_vals, yticklabels=sorted_f_vals)
        plt.title('Mean runtimes over %d runs, block diagonal'%REPEATS)
        plt.xlabel('m')
        plt.ylabel('f')
        ax.invert_yaxis()
        #plt.show()
        
        plt.subplot(212)
        ax = sns.heatmap(fraction_SAT_diag, annot=SHOW_HEATMAP_VALS, xticklabels=sorted_m_vals, yticklabels=sorted_f_vals)
        plt.title('Fraction satisfied problems over %d runs, block diagonal'%REPEATS)
        plt.xlabel('m')
        plt.ylabel('f')
        ax.invert_yaxis()
        plt.show()

    PLOT_ORIG_BLOCK_DIAG_DIFF = True
    if PLOT_ORIG_BLOCK_DIAG_DIFF:
        percentChange_runtime = np.zeros(mean_runtimes_orig.shape)
        print 'percentChange_runtime.shape', percentChange_runtime.shape
        print 'mean_runtimes_diag.shape', mean_runtimes_diag.shape
        print 'mean_runtimes_orig.shape', mean_runtimes_orig.shape
        for r in range(mean_runtimes_orig.shape[0]):
            for c in range(mean_runtimes_orig.shape[1]):
                print 'r', r, 'c', c
                percentChange_runtime[r,c] = (mean_runtimes_diag[r,c] - mean_runtimes_orig[r,c])/mean_runtimes_orig[r,c]

        plt.figure(1)
        plt.subplot(211)
        ax = sns.heatmap(percentChange_runtime, annot=SHOW_HEATMAP_VALS, xticklabels=sorted_m_vals, yticklabels=sorted_f_vals)
        plt.title('(runtime NEW - runtime Orig)/(runtime Orig) %d runs, block diagonal'%REPEATS)
        plt.xlabel('m')
        plt.ylabel('f')
        ax.invert_yaxis()
        #plt.show()
        
        percentChange_fractionSAT = np.zeros(fraction_SAT_orig.shape)
        for r in range(fraction_SAT_orig.shape[0]):
            for c in range(fraction_SAT_orig.shape[1]):
                percentChange_fractionSAT[r,c] = (fraction_SAT_diag[r,c] - fraction_SAT_orig[r,c])/fraction_SAT_orig[r,c]

        plt.subplot(212)
        ax = sns.heatmap(percentChange_fractionSAT, annot=SHOW_HEATMAP_VALS, xticklabels=sorted_m_vals, yticklabels=sorted_f_vals)
        plt.title('(Fraction SAT NEW - fraction sat Orig)/(fraction sat Orig) %d runs'%REPEATS)
        plt.xlabel('m')
        plt.ylabel('f')
        ax.invert_yaxis()
        plt.show()



    if PLOT_BLOCK_DIAG_PERMUTED:
        ##### block diagonal + randomness #####
        if USE_MULTIPLE_FILES:
            (sorted_m_vals, sorted_f_vals, mean_runtimes, fraction_SAT, fraction_UNSAT) = read_files(filename_base=permutedBlockDiag_filebase, repeats=REPEATS, file_count=FILE_COUNT)
        else:
            (sorted_m_vals, sorted_f_vals, mean_runtimes, fraction_SAT, fraction_UNSAT) = read_file(permutedBlockDiag_file, repeats=REPEATS)    
        
        plt.figure(1)
        plt.subplot(211)
        ax = sns.heatmap(mean_runtimes, annot=SHOW_HEATMAP_VALS, xticklabels=sorted_m_vals, yticklabels=sorted_f_vals)
        plt.title('Mean runtimes over %d runs, block diagonal permuted'%REPEATS)
        plt.xlabel('m')
        plt.ylabel('f')
        ax.invert_yaxis()
        #plt.show()
        
        plt.subplot(212)
        ax = sns.heatmap(fraction_SAT, annot=SHOW_HEATMAP_VALS, xticklabels=sorted_m_vals, yticklabels=sorted_f_vals)
        plt.title('Fraction satisfied problems over %d runs, block diagonal permuted'%REPEATS)
        plt.xlabel('m')
        plt.ylabel('f')
        ax.invert_yaxis()
        plt.show()
     
    if PLOT_PERMUTATION_K1:
        ##### permutation + randomness #####
        (sorted_m_vals, sorted_f_vals, mean_runtimes, fraction_SAT, fraction_UNSAT) = read_file(permutation_file, repeats=REPEATS)
        
        plt.figure(1)
        plt.subplot(211)
        ax = sns.heatmap(mean_runtimes, annot=SHOW_HEATMAP_VALS, xticklabels=sorted_m_vals, yticklabels=sorted_f_vals)
        plt.title('Mean runtimes over %d runs, permutation matrix'%REPEATS)
        plt.xlabel('m')
        plt.ylabel('f')
        ax.invert_yaxis()
        #plt.show()
        
        plt.subplot(212)
        ax = sns.heatmap(fraction_SAT, annot=SHOW_HEATMAP_VALS, xticklabels=sorted_m_vals, yticklabels=sorted_f_vals)
        plt.title('Fraction satisfied problems over %d runs, permutation matrix'%REPEATS)
        plt.xlabel('m')
        plt.ylabel('f')
        ax.invert_yaxis()
        plt.show()
        