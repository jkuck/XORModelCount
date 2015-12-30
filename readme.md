#Usage
python main.py cnf_file_name problem_type f_value output_file_name max_time

problem type can be:
lbo: obtain a lower bound of the model count, using an exploration strategy that is fast, mostly reliable, but have no guarantees
lbe: obtain a lower bound of the model count, using an enumeration strategy that is slow but reliable
ubo: obtain an upper bound of the model count, using the exploration strategy
ube: obtain an upper bound of the model count, using the enumeration strategy
