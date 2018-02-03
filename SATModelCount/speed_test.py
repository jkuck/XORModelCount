from sat import SAT
import time

sat = SAT("examples/wff.3.100.150.cnf", verbose=False)
m = 20
logger = open('result/speed%d.txt' % m, "w")

f = 0.5
for f_index in range(40):
    for repeat in range(20):
        sat.add_parity_constraints(m, f)
        start_time = time.time()
        outcome = sat.solve(3600)
        elapsed_time = time.time() - start_time

        if outcome is True:
            solution = "true"
        elif outcome is False:
            solution = "false"
        else:
            solution = "timeout"

        logger.write("f %f time %f solution %s\n" % (f, elapsed_time, outcome))
        print("f = %.3f, time=%.2f, solution=%s" % (f, elapsed_time, outcome))
    f *= 0.92