DATA=res1u.pkl res2u.pkl res5u.pkl
DEPS:=mud_problem.py convergence.py experiments.py plotting.py poisson.py

# n is number of trials (repeatitions of experiment to avg out results)
# r stands for proportion of time elapsed
# t stands for tolerance (of measurement equipment)
ode: mud_problem.py
	./run_decay.sh
	mkdir -p ode/
	mv ode*.png ode/

pde: $(DATA) $(DEPS)
	./run_poisson.sh

$(DATA): generate_pde_data.sh pp.py $(DEPS)
	./generate_pde_data.sh

data: $(DATA)

.PHONY: ode pde deps data
