def read_files_moreInfo_newFormat(filename_base, repeats, file_count):
    '''

    Inputs:
    - repeats: (int) the number of experiments run for each f and m value
    - file_count: (int) the number of files containing identical experiment sets
    - new_format: (bool) file contains 
        f: density 
        m: number of parity constraints
        f_prime: probability of flipping all elements
        cur_k: the block size of elements that start at 1 before flipping and permuting
        n: number of variables
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
    #value: f_prime for these f, m values
    f_prime_dict = {}

    #key: (f, m)
    #value: k for these f, m values
    k_dict = {}  

    #key: (f, m)
    #value: number of trials that timed out
    num_TIMEOUT_dict = defaultdict(int)

    for exp_idx in range(file_count):
        cur_filename = '%s%d.txt' % (filename_base, exp_idx)
        if os.path.isfile(cur_filename):        
#            print "reading file:", cur_filename
            reader = open(cur_filename, 'r')
            while True:
                line = reader.readline().split()
                if len(line) < 15:
                    if len(line) == 14 and line[13] == 'None':
                        #problem timed out
                        f = float(line[3])
                        m = int(line[11])   
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
        
                f = float(line[3])
                run_time = float(line[14][0:-1])
                m = int(line[11])
                if line[13] == '(True,':
                    num_SAT_dict[(f,m)] += 1
                    SAT_runtimes[(f,m)].append(run_time/(mean_unperturbed_run_time)) #runtime normalized by mean_unperturbed_run_time
                else:
                    assert(line[13] == '(False,')
                    num_UNSAT_dict[(f,m)] += 1
                    UNSAT_runtimes[(f,m)].append(run_time/(mean_unperturbed_run_time)) #runtime normalized by mean_unperturbed_run_time

                f_prime = float(line[1])
                k = int(float((line[5])))
                f_prime_dict[(f,m)] = f_prime
                k_dict[(f,m)] = k

                m_vals.add(m)
                f_vals.add(f)
                all_runtimes_dict[(f,m)].append(run_time/(mean_unperturbed_run_time)) #runtime normalized by mean_unperturbed_run_time
                num_trials_dict[(f,m)] += 1
        else:
            print "file doesn't exist:", cur_filename

    sorted_m_vals = sorted(m_vals)
    sorted_f_vals = sorted(f_vals)
    #print m_vals
    #print sorted_m_vals
    num_SAT = np.zeros((len(sorted_f_vals), len(sorted_m_vals)))
    num_UNSAT = np.zeros((len(sorted_f_vals), len(sorted_m_vals)))
    num_TIMEOUT = np.zeros((len(sorted_f_vals), len(sorted_m_vals)))

    for (m_idx, m_val) in enumerate(sorted_m_vals):
        for (f_idx, f_val) in enumerate(sorted_f_vals):
            #print 'm:', m_val, 'f:', f_val, "num_trials:", num_trials_dict[(f_val, m_val)]
            SAT_runtimes[(f_val, m_val)].sort()
            UNSAT_runtimes[(f_val, m_val)].sort()
            num_SAT[f_idx, m_idx] = num_SAT_dict[(f_val, m_val)]
            num_UNSAT[f_idx, m_idx] = num_UNSAT_dict[(f_val, m_val)]
            num_TIMEOUT[f_idx, m_idx] = num_TIMEOUT_dict[(f_val, m_val)]
            assert(len(UNSAT_runtimes[(f_val, m_val)]) == num_UNSAT[f_idx, m_idx])
            assert(len(SAT_runtimes[(f_val, m_val)]) == num_SAT[f_idx, m_idx])
            #assert(num_trials_dict[(f_val, m_val)] == 100), num_trials_dict[(f_val, m_val)]
            assert(num_trials_dict[(f_val, m_val)] == num_UNSAT[f_idx, m_idx] + num_SAT[f_idx, m_idx] + num_TIMEOUT[f_idx, m_idx])

    return(sorted_m_vals, sorted_f_vals, SAT_runtimes, UNSAT_runtimes, num_SAT, num_trials_dict, all_runtimes_dict, f_prime_dict, k_dict)
