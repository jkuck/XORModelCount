from __future__ import division
from sat import SAT
import time
import os
import math
from fireworks import Firework, Workflow, FWorker, LaunchPad
from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FWAction, FireTaskBase

#True: run locally
#False: run remotely on cluster
TEST_LOCAL = False

if TEST_LOCAL:
    from fireworks.core.rocket_launcher import rapidfire
else:
    from fireworks.queue.queue_launcher import rapidfire

from fireworks.user_objects.queue_adapters.common_adapter import CommonAdapter
from fw_tutorials.dynamic_wf.fibadd_task import FibonacciAdderTask

from cluster_config import HOME_DIRECTORY, MONGODB_USERNAME, MONGODB_PASSWORD
from experiment_config import MONGODB_HOST, MONGODB_PORT, MONGODB_NAME

# Add the following line to the file ~/.bashrc.user on Atlas:
# export PYTHONPATH="/atlas/u/jkuck/XORModelCount/SATModelCount:$PYTHONPATH"


# $ export PATH=/opt/rh/python27/root/usr/bin:$PATH
# $ export LD_LIBRARY_PATH=/opt/rh/python27/root/usr/lib64/:$LD_LIBRARY_PATH
# $ PACKAGE_DIR=/atlas/u/jkuck/software
# $ export PATH=$PACKAGE_DIR/anaconda2/bin:$PATH
# $ export LD_LIBRARY_PATH=$PACKAGE_DIR/anaconda2/local:$LD_LIBRARY_PATH
# $ source activate anaconda_venv
# $ cd /atlas/u/jkuck/XORModelCount/SATModelCount/fireworks
# $ python fireworks_speed_test_heatmap.py

#used for RunExperimentBatch
MAX_TIME = 360 #max time to run a single SAT problem

#used for RunSpecificExperimentBatch
MAX_TIMEOUT_MULTIPLE = 100 #run at max MAX_TIMEOUT_MULTIPLE*unperturbed runtime

m_ranges = {#'c432.isc': range(25, 42), #log_2(Z) = 36.1
            'c432.isc': range(25, 46), #log_2(Z) = 36.1
            'c499.isc': range(30, 51), #log_2(Z) = 41.0
            'c880.isc': range(50, 71), #log_2(Z) = 60.0
            'c1355.isc': range(30, 51), #log_2(Z) = 41.0
            'c1908.isc': range(20, 44), #log_2(Z) = 33.0
            'c2670.isc': range(220, 265), #log_2(Z) = 233
            'sat-grid-pbl-0010.cnf': range(65, 95), #log_2(Z) = 78.9
            'sat-grid-pbl-0015.cnf': range(170, 210), #log_2(Z) = 180.9
            'sat-grid-pbl-0020.cnf': range(310, 350), #log_2(Z) = 318
            'ra.cnf': range(920, 1000), #log_2(Z) = 951.0
            'tire-1.cnf': range(20, 40), #log_2(Z) = 29.4    #range(27, 32), #range(20, 40), 
            'tire-2.cnf': range(30, 55), #log_2(Z) = 39.4    #range(27, 32), #range(20, 40), 
            'tire-3.cnf': range(25, 55), #log_2(Z) = 37.7    #range(27, 32), #range(20, 40), 
            'tire-4.cnf': range(35, 60), #log_2(Z) = 46.6    #range(27, 32), #range(20, 40), 
            'log-1.cnf': range(60, 85), #log_2(Z) = 69.0
            'log-2.cnf': range(30, 45), #log_2(Z) = 34.9
            'lang12.cnf': range(10, 26) #log_2(Z) =
            }

f_ranges = {'c432.isc': [i/1000.0 for i in range(1,10)] + [i/100.0 for i in range(1, 50)],
            #'c432.isc': [i/1000.0 for i in range(1,10)] + [i/100.0 for i in range(1, 7)],
            #'c432.isc': [.0001, .001],
            'c499.isc': [i/1000.0 for i in range(1,10)] + [i/100.0 for i in range(1, 50)],
            'lang12.cnf': [i/1000.0 for i in range(1,10)] + [i/100.0 for i in range(1, 50)],
            'c880.isc': [i/1000.0 for i in range(1,10)] + [i/100.0 for i in range(1, 50)],
            'c1355.isc': [i/1000.0 for i in range(1,10)] + [i/100.0 for i in range(1, 50)],
            'c1908.isc': [i/1000.0 for i in range(1,10)] + [i/100.0 for i in range(1, 50)],
            'c2670.isc': [i/1000.0 for i in range(1,10)] + [i/100.0 for i in range(1, 50)],
            'sat-grid-pbl-0010.cnf': [i/1000.0 for i in range(1,10)] + [i/100.0 for i in range(1, 50)],
            'sat-grid-pbl-0015.cnf': [i/1000.0 for i in range(1,10)] + [i/100.0 for i in range(1, 50)],
            'sat-grid-pbl-0020.cnf': [i/1000.0 for i in range(1,10)] + [i/100.0 for i in range(1, 50)],
            'ra.cnf': [i/1000.0 for i in range(1,10)] + [i/100.0 for i in range(1, 50)],
            'tire-1.cnf': [i/1000.0 for i in range(1,10)] + [i/100.0 for i in range(1, 50)],
            'tire-2.cnf': [i/1000.0 for i in range(1,10)] + [i/100.0 for i in range(1, 50)],
            'tire-3.cnf': [i/1000.0 for i in range(1,10)] + [i/100.0 for i in range(1, 50)],
            'tire-4.cnf': [i/1000.0 for i in range(1,10)] + [i/100.0 for i in range(1, 50)],
            'log-1.cnf': [i/1000.0 for i in range(1,10)] + [i/100.0 for i in range(1, 50)],
            'log-2.cnf': [i/1000.0 for i in range(1,10)] + [i/100.0 for i in range(1, 50)],
            }

#logger = open('heatmap_result_moreModels2/speed=%d.txt' % (m), "w")
@explicit_serialize
class RunExperimentBatch(FireTaskBase):   
    def run_task(self, fw_spec):
        RESULTS_DIRECTORY = '/atlas/u/jkuck/XORModelCount/SATModelCount/fireworks/heatmap_result_fireworksK3'
        if not os.path.exists(RESULTS_DIRECTORY):
            os.makedirs(RESULTS_DIRECTORY)        
        #for dup_copies in [0, 1, 3, 7, 15, 31, 63, 127]:
        for dup_copies in [0]:
        #for dup_copies in [7, 15, 31, 63, 127]:
        #    for f in [f*(dup_copies+1) for f in f_ranges[fw_spec['problem_name']]]:
            for f in f_ranges[fw_spec['problem_name']]:
        #        for m in m_ranges[fw_spec['problem_name']]:
                for m in [m*(dup_copies+1) for m in m_ranges[fw_spec['problem_name']]]:
                    for repeat in range(fw_spec['repeats']):
                        print fw_spec['problem_name'], 'dup_copies=', dup_copies
                        RUN_BLOCK_DIAG = True
                        if RUN_BLOCK_DIAG:
                            #block diagonal, deterministic 1's on block
                            filename = '%s/blockDiagDeterministic_speed_REPEATS=%d_%s_duplicates=%d_expIdx=%d.txt'%(RESULTS_DIRECTORY, fw_spec['repeats'], fw_spec['problem_name'].split('.')[0], dup_copies, fw_spec['experiment_idx'])
                            if os.path.exists(filename):
                                append_write = 'a' # append if already exists
                            else:
                                append_write = 'w' # make a new file if not
                            logger = open(filename, append_write)
                            sat = SAT("/atlas/u/jkuck/low_density_parity_checks/SAT_problems_cnf/%s"%fw_spec['problem_name'], verbose=False, instance_id=fw_spec['problem_name'], duplicate=dup_copies)
                            sat.add_regular_constraints_constantF_permuted(m=m, f=f, f_block=1.0, permute=False, k=3)
                            start_time = time.time()
                            outcome = sat.solve(MAX_TIME)
                            elapsed_time = time.time() - start_time
                        
                            logger.write("f %f time %f m %d solution %s\n" % (f, elapsed_time, m, outcome))
                            logger.close()
                            print("f = %.3f, time=%.2f, m=%d, solution=%s" % (f, elapsed_time, m, outcome))
        

                            #block diagonal, sample 1's on block w/ probability 1-f
                            filename = '%s/blockDiag1MinusF_speed_REPEATS=%d_%s_duplicates=%d_expIdx=%d.txt'%(RESULTS_DIRECTORY, fw_spec['repeats'], fw_spec['problem_name'].split('.')[0], dup_copies, fw_spec['experiment_idx'])
                            if os.path.exists(filename):
                                append_write = 'a' # append if already exists
                            else:
                                append_write = 'w' # make a new file if not
                            logger = open(filename, append_write)
                            sat = SAT("/atlas/u/jkuck/low_density_parity_checks/SAT_problems_cnf/%s"%fw_spec['problem_name'], verbose=False, instance_id=fw_spec['problem_name'], duplicate=dup_copies)
                            sat.add_regular_constraints_constantF_permuted(m=m, f=f, f_block=1.0-f, permute=False, k=3)
                            start_time = time.time()
                            outcome = sat.solve(MAX_TIME)
                            elapsed_time = time.time() - start_time
                        
                            logger.write("f %f time %f m %d solution %s\n" % (f, elapsed_time, m, outcome))
                            logger.close()
                            print("f = %.3f, time=%.2f, m=%d, solution=%s" % (f, elapsed_time, m, outcome))

        
                        RUN_PERMUTED_BLOCK_DIAG = True
                        if RUN_PERMUTED_BLOCK_DIAG:
                            #permuted block diagonal, deterministic 1's on block 
                            filename = '%s/permutedBlockDiagDeterministic_speed_REPEATS=%d_%s_duplicates=%d_expIdx=%d.txt'%(RESULTS_DIRECTORY, fw_spec['repeats'], fw_spec['problem_name'].split('.')[0], dup_copies, fw_spec['experiment_idx'])
                            if os.path.exists(filename):
                                append_write = 'a' # append if already exists
                            else:
                                append_write = 'w' # make a new file if not
                            logger = open(filename, append_write)
                            sat = SAT("/atlas/u/jkuck/low_density_parity_checks/SAT_problems_cnf/%s"%fw_spec['problem_name'], verbose=False, instance_id=fw_spec['problem_name'], duplicate=dup_copies)
                            sat.add_regular_constraints_constantF_permuted(m=m, f=f, f_block=1.0, permute=True, k=3)
                            start_time = time.time()
                            outcome = sat.solve(MAX_TIME)
                            elapsed_time = time.time() - start_time
                        
                            logger.write("f %f time %f m %d solution %s\n" % (f, elapsed_time, m, outcome))
                            logger.close()
                            print("f = %.3f, time=%.2f, m=%d, solution=%s" % (f, elapsed_time, m, outcome))

                            #permuted block diagonal, sample 1's on block w/ probability 1-f
                            filename = '%s/permutedBlockDiag1MinusF_speed_REPEATS=%d_%s_duplicates=%d_expIdx=%d.txt'%(RESULTS_DIRECTORY, fw_spec['repeats'], fw_spec['problem_name'].split('.')[0], dup_copies, fw_spec['experiment_idx'])
                            if os.path.exists(filename):
                                append_write = 'a' # append if already exists
                            else:
                                append_write = 'w' # make a new file if not
                            logger = open(filename, append_write)
                            sat = SAT("/atlas/u/jkuck/low_density_parity_checks/SAT_problems_cnf/%s"%fw_spec['problem_name'], verbose=False, instance_id=fw_spec['problem_name'], duplicate=dup_copies)
                            sat.add_regular_constraints_constantF_permuted(m=m, f=f, f_block=1.0-f, permute=True, k=3)
                            start_time = time.time()
                            outcome = sat.solve(MAX_TIME)
                            elapsed_time = time.time() - start_time
                        
                            logger.write("f %f time %f m %d solution %s\n" % (f, elapsed_time, m, outcome))
                            logger.close()
                            print("f = %.3f, time=%.2f, m=%d, solution=%s" % (f, elapsed_time, m, outcome))
        
        
                        RUN_PERMUTATION = False
                        if RUN_PERMUTATION:
                            #permutation 
                            filename = '%s/pspeed_REPEATS=%d_%s_duplicates=%d_expIdx=%d.txt'%(RESULTS_DIRECTORY, fw_spec['repeats'], fw_spec['problem_name'].split('.')[0], dup_copies, fw_spec['experiment_idx'])
                            if os.path.exists(filename):
                                append_write = 'a' # append if already exists
                            else:
                                append_write = 'w' # make a new file if not
                            logger = open(filename, append_write)            
            
                            sat = SAT("/atlas/u/jkuck/low_density_parity_checks/SAT_problems_cnf/%s"%fw_spec['problem_name'], verbose=False, instance_id=fw_spec['problem_name'], duplicate=dup_copies)
                            sat.add_permutation_constraints(m, f)
                            start_time = time.time()
                            outcome = sat.solve(MAX_TIME)
                            elapsed_time = time.time() - start_time
                        
                            logger.write("f %f time %f m %d solution %s\n" % (f, elapsed_time, m, outcome))
                            logger.close()
                            print("f = %.3f, time=%.2f, m=%d, solution=%s" % (f, elapsed_time, m, outcome))            
        
        
                        #original
                        filename = '%s/speed_REPEATS=%d_%s_duplicates=%d_expIdx=%d.txt'%(RESULTS_DIRECTORY, fw_spec['repeats'], fw_spec['problem_name'].split('.')[0], dup_copies, fw_spec['experiment_idx'])
                        if os.path.exists(filename):
                            append_write = 'a' # append if already exists
                        else:
                            append_write = 'w' # make a new file if not
                        logger = open(filename, append_write)                      
                        sat = SAT("/atlas/u/jkuck/low_density_parity_checks/SAT_problems_cnf/%s"%fw_spec['problem_name'], verbose=False, instance_id=fw_spec['problem_name'], duplicate=dup_copies)
                        sat.add_parity_constraints(m, f)
                #        sat.add_regular_constraints(m, f)
                #        sat.add_permutation_constraints(m, f)
                        start_time = time.time()
                        outcome = sat.solve(MAX_TIME)
                        elapsed_time = time.time() - start_time        
        
                        logger.write("f %f time %f m %d solution %s\n" % (f, elapsed_time, m, outcome))
                        logger.close()
                        print("f = %.3f, time=%.2f, m=%d, solution=%s" % (f, elapsed_time, m, outcome))

#logger = open('heatmap_result_moreModels2/speed=%d.txt' % (m), "w")
@explicit_serialize
class RunSpecificExperimentBatch(FireTaskBase):   
    def run_task(self, fw_spec):
        RESULTS_DIRECTORY = '/atlas/u/jkuck/XORModelCount/SATModelCount/fireworks/heatmap_result_fireworksTIMEOUTallFcorrected1/%s' % fw_spec['problem_name'].split('.')[0]
        if not os.path.exists(RESULTS_DIRECTORY):
            os.makedirs(RESULTS_DIRECTORY)      

        if fw_spec['adjust_f']:
            filename = '%s/f_block=%s_permute=%s_k=%s_allOnesConstraint=%s_adjustF_REPEATS=%d_expIdx=%d.txt'%\
                (RESULTS_DIRECTORY, fw_spec['f_block'], fw_spec['permute'], fw_spec['k'], fw_spec['ADD_CONSTRAINT_ALL_ONES'],\
                fw_spec['repeats'], fw_spec['experiment_idx'])

        else:
            filename = '%s/f_block=%s_permute=%s_k=%s_allOnesConstraint=%s_REPEATS=%d_expIdx=%d.txt'%\
                (RESULTS_DIRECTORY, fw_spec['f_block'], fw_spec['permute'], fw_spec['k'], fw_spec['ADD_CONSTRAINT_ALL_ONES'],\
                fw_spec['repeats'], fw_spec['experiment_idx'])


        REPEATS = 10
        total_time = 0.0
        for i in range(REPEATS):
            sat = SAT("/atlas/u/jkuck/low_density_parity_checks/SAT_problems_cnf/%s"%fw_spec['problem_name'], verbose=False, instance_id=fw_spec['problem_name'], duplicate=0)
            outcome = sat.solve(3600)
            elapsed_time = outcome[1]
            total_time += elapsed_time
        mean_runtime = total_time/REPEATS
        logger = open(filename, 'w')
        logger.write("mean_unperturbed_run_time= %f\n" % mean_runtime)
        logger.write("MAX_TIMEOUT_MULTIPLE= %d\n" % MAX_TIMEOUT_MULTIPLE)
        logger.close()

        #for dup_copies in [0, 1, 3, 7, 15, 31, 63, 127]:
        for dup_copies in [0]:
        #for dup_copies in [7, 15, 31, 63, 127]:
        #    for f in [f*(dup_copies+1) for f in f_ranges[fw_spec['problem_name']]]:
            for f in f_ranges[fw_spec['problem_name']]:
        #        for m in m_ranges[fw_spec['problem_name']]:
                quit_m_early = False
                last_m_val = -1
                for m in [m*(dup_copies+1) for m in m_ranges[fw_spec['problem_name']]]:
                    if fw_spec['adjust_f']:
                        #compute f such that the original matrix construction using iid entries will
                        #have the same expected number of 1's as when floor(n/m) entries are added with
                        #probability (1-f)
                        sat = SAT("/atlas/u/jkuck/low_density_parity_checks/SAT_problems_cnf/%s"%fw_spec['problem_name'], verbose=False, instance_id=fw_spec['problem_name'], duplicate=0)
                        max_const_k = int(math.floor(float(sat.n) / m))
                        f_orig = (f*(sat.n - 2*max_const_k) + max_const_k)/(sat.n)
                        if f_orig > .5:
                            continue
                    else:
                        f_orig = f
                    failures = 0
                    for repeat in range(fw_spec['repeats']):
                        logger = open(filename, 'a')
                        sat = SAT("/atlas/u/jkuck/low_density_parity_checks/SAT_problems_cnf/%s"%fw_spec['problem_name'], verbose=False, instance_id=fw_spec['problem_name'], duplicate=dup_copies)
                        if fw_spec['f_block'] == '1':
                            sat.add_regular_constraints_constantF_permuted(m=m, f=f_orig, f_block=1.0, permute=fw_spec['permute'], k=fw_spec['k'],\
                                                                       ADD_CONSTRAINT_ALL_ONES=fw_spec['ADD_CONSTRAINT_ALL_ONES'])
                        else:
                            assert(fw_spec['f_block'] == '1minusF')
                            sat.add_regular_constraints_constantF_permuted(m=m, f=f_orig, f_block=1.0-f_orig, permute=fw_spec['permute'], k=fw_spec['k'],\
                                                                       ADD_CONSTRAINT_ALL_ONES=fw_spec['ADD_CONSTRAINT_ALL_ONES'])

                        start_time = time.time()
                        outcome = sat.solve(mean_runtime*MAX_TIMEOUT_MULTIPLE)
                        elapsed_time = time.time() - start_time
                        if outcome == None:
                            failures += 1
                    
                        logger.write("f %f time %f m %d solution %s\n" % (f_orig, elapsed_time, m, outcome))
                        logger.close()
                        print("f = %.3f, time=%.2f, m=%d, solution=%s" % (f_orig, elapsed_time, m, outcome))
    
                    if failures == fw_spec['repeats']:
                        quit_m_early = True
                        last_m_val = m
                        break
                if quit_m_early:
                    #start with largest m and iterate down
                    for m in reversed([m*(dup_copies+1) for m in m_ranges[fw_spec['problem_name']]]):
                        if m <= last_m_val:
                            break
                        if fw_spec['adjust_f']:
                            #compute f such that the original matrix construction using iid entries will
                            #have the same expected number of 1's as when floor(n/m) entries are added with
                            #probability (1-f)
                            sat = SAT("/atlas/u/jkuck/low_density_parity_checks/SAT_problems_cnf/%s"%fw_spec['problem_name'], verbose=False, instance_id=fw_spec['problem_name'], duplicate=0)
                            max_const_k = int(math.floor(float(sat.n) / m))
                            f_orig = (f*(sat.n - 2*max_const_k) + max_const_k)/(sat.n)
                            if f_orig > .5:
                                continue
                        else:
                            f_orig = f

                        failures = 0
                        for repeat in range(fw_spec['repeats']):
                            logger = open(filename, 'a')
                            sat = SAT("/atlas/u/jkuck/low_density_parity_checks/SAT_problems_cnf/%s"%fw_spec['problem_name'], verbose=False, instance_id=fw_spec['problem_name'], duplicate=dup_copies)
                            if fw_spec['f_block'] == '1':
                                sat.add_regular_constraints_constantF_permuted(m=m, f=f_orig, f_block=1.0, permute=fw_spec['permute'], k=fw_spec['k'],\
                                                                           ADD_CONSTRAINT_ALL_ONES=fw_spec['ADD_CONSTRAINT_ALL_ONES'])
                            else:
                                assert(fw_spec['f_block'] == '1minusF')
                                sat.add_regular_constraints_constantF_permuted(m=m, f=f_orig, f_block=1.0-f_orig, permute=fw_spec['permute'], k=fw_spec['k'],\
                                                                           ADD_CONSTRAINT_ALL_ONES=fw_spec['ADD_CONSTRAINT_ALL_ONES'])

                            start_time = time.time()
                            outcome = sat.solve(mean_runtime*MAX_TIMEOUT_MULTIPLE)
                            elapsed_time = time.time() - start_time
                            if outcome == None:
                                failures += 1
                        
                            logger.write("f %f time %f m %d solution %s\n" % (f_orig, elapsed_time, m, outcome))
                            logger.close()
                            print("f = %.3f, time=%.2f, m=%d, solution=%s" % (f_orig, elapsed_time, m, outcome))
        
                        if failures == fw_spec['repeats']:
                            break
def create_launchpad():
    with open('./my_launchpad.yaml', 'w') as f:
        f.write('host: %s\n' % MONGODB_HOST)
        f.write('port: %d\n' % MONGODB_PORT)
        f.write('name: %s\n' % MONGODB_NAME)
        f.write('username: %s\n' % MONGODB_USERNAME)
        f.write('password: %s\n' % MONGODB_PASSWORD)
        f.write('logdir: null\n')
        f.write('strm_lvl: INFO\n')

def run_experiment():
    '''

    '''
    # write new launchpad file
    create_launchpad()

    # set up the LaunchPad and reset it
    launchpad = LaunchPad(host=MONGODB_HOST, port=MONGODB_PORT, name=MONGODB_NAME, username=MONGODB_USERNAME, password=MONGODB_PASSWORD,
                     logdir=None, strm_lvl='INFO', user_indices=None, wf_user_indices=None, ssl_ca_file=None)
    launchpad.reset('', require_password=False)
         

    all_fireworks = [] 

    PROBLEM_NAMES = ['hypercube.cnf', 'hypercube1.cnf', 'hypercube2.cnf', 'c499.isc', 'c432.isc', 'tire-1.cnf', 'tire-2.cnf', 'tire-3.cnf', 'tire-4.cnf', 'lang12.cnf', 'c880.isc', 'c1355.isc', 'c1908.isc', 'c2670.isc', 'sat-grid-pbl-0010.cnf', 'sat-grid-pbl-0015.cnf', 'sat-grid-pbl-0020.cnf', 'log-1.cnf', 'log-2.cnf', 'ra.cnf']
    #PROBLEM_NAMES = ['c432.isc']
    REPEATS_OF_EXPERIMENT = 10

    for problem_name in PROBLEM_NAMES:
        for repeats_per_experiment in [10]:
            for experiment_idx in range(REPEATS_OF_EXPERIMENT): #repeat the same experiment this many times
#                for ADD_CONSTRAINT_ALL_ONES in [True, False]:
#                    for (f_block, permute, k) in [('1', True, None), ('1minusF', True, None), ('1', False, None), ('1minusF', False, None),\
#                                                  ('1', True, 3), ('1minusF', True, 3), ('1', False, 3), ('1minusF', False, 3),\
#                                                  ('1', True, 1), ('1minusF', True, 1), ('1', False, 1), ('1minusF', False, 1),
#                                                  ('1', False, 0)]
                for (f_block, permute, k, ADD_CONSTRAINT_ALL_ONES, adjust_f) in \
                    [('1minusF', True, 'maxConstant', False, False),\
                     ('1', False, 0, False, False),\
                     ('1', True, None, False, False), ('1', False, None, False, False)]:                    
                    #[('1minusF', True, 'maxConstant', False), ('1minusF', True, 'maxConstant', True),\
                    #('1', True, None, True), ('1', False, None, True)]:                    
                    #[('1minusF', True, None, False), ('1minusF', False, None, False), ('1minusF', True, 3, True),\
                    # ('1minusF', True, 3, False),\
                    # ('1', True, 1, False),\
                    # ('1', False, 0, False), ('1', False, 0, True)]:                     
                    cur_spec = {'problem_name': problem_name,
                                'repeats': repeats_per_experiment,
                                'experiment_idx': experiment_idx,
                                'f_block': f_block,
                                'permute': permute,
                                'k': k,
                                'ADD_CONSTRAINT_ALL_ONES':ADD_CONSTRAINT_ALL_ONES,
                                #compute f such that the original matrix construction using iid entries will
                                #have the same expected number of 1's as when floor(n/m) entries are added with
                                #probability (1-f)
                                'adjust_f': adjust_f,  
                                }
                    all_fireworks.append(Firework(RunSpecificExperimentBatch(), spec=cur_spec))

    firework_dependencies = {}
    workflow = Workflow(all_fireworks, firework_dependencies)
    if TEST_LOCAL:
        launchpad.add_wf(workflow)
        rapidfire(launchpad, FWorker())
    else:
        launchpad.add_wf(workflow)
        qadapter = CommonAdapter.from_file("%s/my_qadapter.yaml" % HOME_DIRECTORY)
        rapidfire(launchpad, FWorker(), qadapter, launch_dir='.', nlaunches='infinite', njobs_queue=200,
                      njobs_block=500, sleep_time=None, reserve=False, strm_lvl='INFO', timeout=None,
                      fill_mode=False)

if __name__=="__main__":
    run_experiment()

######################### Fireworks info copied from anothor project #########################
# If the database thinks a firework is still running, but no jobs are running on the cluster, try:
# $ lpad detect_lostruns --time 1 --refresh
#
# If a firework fizzles and you are trying to find the error/output, note the fireworks fw_id
# in the online database, then search for this fw_id in the launcher block, e.g.:
# $ cd block_2017-11-01-07-30-53-457640
# $ pt 'fw_id: 34'
# or on atlas-ws-6 use silver searcher:
# $ ag 'fw_id: 34'
#
#Note, on Atlas before this script:
# start a krbscreen session:
# $ krbscreen #reattach using $ screen -rx
# $ reauth #important so that jobs can be submitted after logging out, enter password
#
# $ export PATH=/opt/rh/python27/root/usr/bin:$PATH
# $ export LD_LIBRARY_PATH=/opt/rh/python27/root/usr/lib64/:$LD_LIBRARY_PATH
# $ PACKAGE_DIR=/atlas/u/jkuck/software
# $ export PATH=$PACKAGE_DIR/anaconda2/bin:$PATH
# $ export LD_LIBRARY_PATH=$PACKAGE_DIR/anaconda2/local:$LD_LIBRARY_PATH
# $ source activate anaconda_venv
# $ cd /atlas/u/jkuck/rbpf_fireworks/
#
# To install anaconda packages run, e.g.:
# $ conda install -c matsci fireworks=1.3.9
#
#May need to run $ kinit -r 30d
#
# Add the following line to the file ~/.bashrc.user on Atlas:
# export PYTHONPATH="/atlas/u/jkuck/rbpf_fireworks:$PYTHONPATH"
# Weird, but to run commands like "lpad -l my_launchpad.yaml get_fws",
# add the following line to the file ~/.bashrc.user on Atlas:
# export PYTHONPATH="${PYTHONPATH}:/atlas/u/jkuck/rbpf_fireworks/KITTI_helpers/"
#
# To install cvxpy on atlas run (hopefully):
#
#$ export PATH=/opt/rh/python27/root/usr/bin:$PATH
#$ export LD_LIBRARY_PATH=/opt/rh/python27/root/usr/lib64/:$LD_LIBRARY_PATH
#$ pip install --user numpy
#$ pip install --user cvxpy
#
# Install pymatgen:
#$ pip install --user pymatgen
##########################################################################################
#
#Note, on Sherlock before this script:
#$ ml load python/2.7.5
#$ easy_install-2.7 --user pip
#$ export PATH=~/.local/bin:$PATH
#$ pip2.7 install --user fireworks #and others
#$ pip2.7 install --user filterpy
#$ pip2.7 install --user scipy --upgrade
#$ pip2.7 install --user munkres
#$ pip2.7 install --user pymatgen
#$ cd /scratch/users/kuck/rbpf_fireworks/
#
# Add the following line to the file ~/.bashrc on Sherlock:
# export PYTHONPATH="/scratch/users/kuck/rbpf_fireworks:$PYTHONPATH"
# Weird, but to run commands like "lpad -l my_launchpad.yaml get_fws",
# add the following line to the file ~/.bashrc.user on Atlas:
# export PYTHONPATH="${PYTHONPATH}:/scratch/users/kuck/rbpf_fireworks/KITTI_helpers/"
#
#
# When setting up:
# - make cluster_config.py file
# - make my_qadapter.yaml file (look at fireworks workflow manager website for info)
#
# To install cvxpy on sherlock run:
# $ pip2.7 install --user cvxpy
