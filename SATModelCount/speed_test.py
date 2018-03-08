from sat import SAT
import time

#sat = SAT("examples/wff.3.100.150.cnf", verbose=False)
sat = SAT("examples/lang12.cnf", verbose=False)
#sat = SAT("../../low_density_parity_checks/SAT_problems_cnf/sat-grid-pbl-0010.cnf", verbose=False)
m = 17
logger = open('result/pspeed_m=%d.txt' % (m), "w")
#logger = open('result/rspeed%d.txt' % m, "w")

average_time = 0.0
true_count = 0

f = 0.07
REPEATS = 100
#for f_index in range(40):
#for f_index in range(28):
for f_index in range(1):
    for repeat in range(REPEATS):
        sat.add_parity_constraints(m, f)
#        sat.add_regular_constraints(m, f)
#        sat.add_permutation_constraints(m, f)
        start_time = time.time()
        outcome = sat.solve(3600)
        elapsed_time = time.time() - start_time

        average_time += outcome[1]

        if outcome is True:
            solution = "true"
        elif outcome is False:
            solution = "false"
        else:
            solution = "timeout"

        if outcome[0] == True:
            true_count += 1


        logger.write("f %f time %f solution %s\n" % (f, elapsed_time, outcome))
        print("f = %.3f, time=%.2f, solution=%s" % (f, elapsed_time, outcome))
    f *= 0.92

print "number of times satisfying solution found =", true_count
print "average_time =", average_time/REPEATS