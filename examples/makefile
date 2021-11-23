all: heatrod linear
	echo "Built all"

linear: examples_linear.sh models.py
	sh examples_linear.sh


clean: clean-heat

# the nuclear option: wipe all examples
clean-heat: \
	clean-heat-a \
	clean-heat-b

# ch 2+3 heatrod example for fifty and five-hundred parameter samples
# ch2 has dependence on size
# ch3 has the introduction of a second set of QoI (measurement locations)
heatrod: heatrod-a heatrod-b models.py
# not sure the Dockerfile symlink depenency works. maybe swap them?

heatrod-a: \
	fig_heatrod_q1/HeatrodModel--sample_N50_mc.png \
	fig_heatrod_q1/HeatrodModel--set_N50_em.png \
	fig_heatrod_q1/HeatrodModel--sample_N500_mc.png \
	fig_heatrod_q1/HeatrodModel--set_N500_em.png

heatrod-b: \
	fig_heatrod_q2/HeatrodModel--sample_N50_mc.png \
	fig_heatrod_q2/HeatrodModel--set_N50_em.png \
	fig_heatrod_q2/HeatrodModel--sample_N500_mc.png \
	fig_heatrod_q2/HeatrodModel--set_N500_em.png

clean-heat-a:
	rm -f fig_heatrod_q1/*

clean-heat-b:
	rm -f fig_heatrod_q2/*

.PHONY: \
	all clean \
	identity linear heatrod \
	heatrod-50 heatrod-500 \
	clean-heat clean-heat-a clean-heat-b \
	decay decay-a decay-b \
	clean-decay clean-decay-a clean-decay-b \

# Note: for fenics-based examples, need to add `|| true` to end of python statement.
# This is because of a bad exit of MPI throwing an error. If the figure truly failed,
# the following instruction would cause the target to fail since there would be no
# file to move into a subfolder. An unfortunate slopiness of the official fenics container.
# TO-DO: see if a conda-based image that installs fenics could actually suffice.

# first QoI map

fig_heatrod_q1/HeatrodModel--set_N500_em.png: heatrod_example.sh solve_problem.py
	. ./heatrod_example.sh && \
	NUM_SAMPLES=500 && \
	LOC_1=0.25 && \
	LOC_2=0.51 && \
	python solve_problem.py -m $${MODEL} --fontsize $${FONTSIZE} --figsize $${FIGSIZE} \
    	--set --seed $${SEED} -o $${M} -n $${NUM_SAMPLES} \
    	--mc_points $${NUM_MC_SAMPLES}  \
    	--t0 $${LOC_1} --t1 $${LOC_2} \
    	--lam1 $${LAM_1} --lam2 $${LAM_2} || true
	mv HeatrodModel*N$${NUM_SAMPLES}* fig_heatrod_q1/
	echo "figure generated"

fig_heatrod_q1/HeatrodModel--sample_N500_mc.png: heatrod_example.sh solve_problem.py
	. ./heatrod_example.sh && \
	NUM_SAMPLES=500 && \
	LOC_1=0.25 && \
	LOC_2=0.51 && \
	python solve_problem.py -m $${MODEL} --fontsize $${FONTSIZE} --figsize $${FIGSIZE} \
		--noeval --sample --seed $${SEED} -n $${NUM_SAMPLES} \
		--t0 $${LOC_1} --t1 $${LOC_2} \
		--lam1 $${LAM_1} --lam2 $${LAM_2} || true
	mv HeatrodModel*N$${NUM_SAMPLES}* fig_heatrod_q1/
	echo "figure generated"

fig_heatrod_q1/HeatrodModel--set_N50_em.png: heatrod_example.sh solve_problem.py
	. ./heatrod_example.sh && \
	NUM_SAMPLES=50 && \
	LOC_1=0.25 && \
	LOC_2=0.51 && \
	python solve_problem.py -m $${MODEL} --fontsize $${FONTSIZE} --figsize $${FIGSIZE} \
    	--set --seed $${SEED} -o $${M} -n $${NUM_SAMPLES} \
    	--mc_points $${NUM_MC_SAMPLES}  \
    	--t0 $${LOC_1} --t1 $${LOC_2} \
    	--lam1 $${LAM_1} --lam2 $${LAM_2} || true
	mv HeatrodModel*N$${NUM_SAMPLES}* fig_heatrod_q1/
	echo "figure generated"

fig_heatrod_q1/HeatrodModel--sample_N50_mc.png: heatrod_example.sh solve_problem.py
	. ./heatrod_example.sh && \
	NUM_SAMPLES=50 && \
	LOC_1=0.25 && \
	LOC_2=0.51 && \
	python solve_problem.py -m $${MODEL} --fontsize $${FONTSIZE} --figsize $${FIGSIZE} \
		--noeval --sample --seed $${SEED} -n $${NUM_SAMPLES} \
		--t0 $${LOC_1} --t1 $${LOC_2} \
		--lam1 $${LAM_1} --lam2 $${LAM_2} || true
	mv HeatrodModel*N$${NUM_SAMPLES}* fig_heatrod_q1/
	echo "figure generated"

# second QoI map

fig_heatrod_q2/HeatrodModel--set_N500_em.png: heatrod_example.sh solve_problem.py
	. ./heatrod_example.sh && \
	NUM_SAMPLES=500 && \
	LOC_1=0.67 && \
	LOC_2=0.98 && \
	python solve_problem.py -m $${MODEL} --fontsize $${FONTSIZE} --figsize $${FIGSIZE} \
    	--set --seed $${SEED} -o $${M} -n $${NUM_SAMPLES} \
    	--mc_points $${NUM_MC_SAMPLES}  \
    	--t0 $${LOC_1} --t1 $${LOC_2} \
    	--lam1 $${LAM_1} --lam2 $${LAM_2} || true
	mv HeatrodModel*N$${NUM_SAMPLES}* fig_heatrod_q2/
	echo "figure generated"

fig_heatrod_q2/HeatrodModel--sample_N500_mc.png: heatrod_example.sh solve_problem.py
	. ./heatrod_example.sh && \
	NUM_SAMPLES=500 && \
	LOC_1=0.67 && \
	LOC_2=0.98 && \
	python solve_problem.py -m $${MODEL} --fontsize $${FONTSIZE} --figsize $${FIGSIZE} \
		--noeval --sample --seed $${SEED} -n $${NUM_SAMPLES} \
		--t0 $${LOC_1} --t1 $${LOC_2} \
		--lam1 $${LAM_1} --lam2 $${LAM_2} || true
	mv HeatrodModel*N$${NUM_SAMPLES}* fig_heatrod_q2/
	echo "figure generated"

fig_heatrod_q2/HeatrodModel--set_N50_em.png: heatrod_example.sh solve_problem.py
	. ./heatrod_example.sh && \
	NUM_SAMPLES=50 && \
	LOC_1=0.67 && \
	LOC_2=0.98 && \
	python solve_problem.py -m $${MODEL} --fontsize $${FONTSIZE} --figsize $${FIGSIZE} \
    	--set --seed $${SEED} -o $${M} -n $${NUM_SAMPLES} \
    	--mc_points $${NUM_MC_SAMPLES}  \
    	--t0 $${LOC_1} --t1 $${LOC_2} \
    	--lam1 $${LAM_1} --lam2 $${LAM_2} || true
	mv HeatrodModel*N$${NUM_SAMPLES}* fig_heatrod_q2/
	echo "figure generated"

fig_heatrod_q2/HeatrodModel--sample_N50_mc.png: heatrod_example.sh solve_problem.py
	. ./heatrod_example.sh && \
	NUM_SAMPLES=50 && \
	LOC_1=0.67 && \
	LOC_2=0.98 && \
	python solve_problem.py -m $${MODEL} --fontsize $${FONTSIZE} --figsize $${FIGSIZE} \
		--noeval --sample --seed $${SEED} -n $${NUM_SAMPLES} \
		--t0 $${LOC_1} --t1 $${LOC_2} \
		--lam1 $${LAM_1} --lam2 $${LAM_2} || true
	mv HeatrodModel*N$${NUM_SAMPLES}* fig_heatrod_q2/
	echo "figure generated"

# Note: t0/t1 are "thermometer locations"

#####

decay: decay-a decay-b

decay-a: \
	fig_decay_q1/DecayModel--sample_N50_mc.png \
	fig_decay_q1/DecayModel--set_N50_em.png \
	fig_decay_q1/DecayModel--sample_N500_mc.png \
	fig_decay_q1/DecayModel--set_N500_em.png

decay-b : \
	fig_decay_q2/DecayModel--sample_N50_mc.png \
	fig_decay_q2/DecayModel--set_N50_em.png \
	fig_decay_q2/DecayModel--sample_N500_mc.png \
	fig_decay_q2/DecayModel--set_N500_em.png

# first at a low sample size

fig_decay_q1/DecayModel--sample_N50_mc.png: decay_example.sh solve_problem.py
	. ./decay_example.sh && \
	NUM_SAMPLES=50 && \
	TIME_0=1 && \
	TIME_1=2 && \
	python solve_problem.py -m $${MODEL} --fontsize $${FONTSIZE} --figsize $${FIGSIZE} \
		--noeval --sample --seed $${SEED} -n $${NUM_SAMPLES} \
		--t0 $${TIME_0} --t1 $${TIME_1} \
		--lam1 $${LAM_1} --lam2 $${LAM_2}
	mv DecayModel*N$${NUM_SAMPLES}* fig_decay_q1/
	echo "figure generated"

fig_decay_q1/DecayModel--set_N50_em.png: decay_example.sh solve_problem.py
	. ./decay_example.sh && \
	NUM_SAMPLES=50 && \
	TIME_0=1 && \
	TIME_1=2 && \
	python solve_problem.py -m $${MODEL} --fontsize $${FONTSIZE} --figsize $${FIGSIZE} \
		--noeval --set --seed $${SEED} -n $${NUM_SAMPLES} \
		--t0 $${TIME_0} --t1 $${TIME_1} \
  	--mc_points $${NUM_MC_SAMPLES}  \
		--lam1 $${LAM_1} --lam2 $${LAM_2}
	mv DecayModel*N$${NUM_SAMPLES}* fig_decay_q1/
	echo "figure generated"

fig_decay_q2/DecayModel--sample_N50_mc.png: decay_example.sh solve_problem.py
	. ./decay_example.sh && \
	NUM_SAMPLES=50 && \
	TIME_0=0.5 && \
	TIME_1=1.0 && \
	python solve_problem.py -m $${MODEL} --fontsize $${FONTSIZE} --figsize $${FIGSIZE} \
		--noeval --sample --seed $${SEED} -n $${NUM_SAMPLES} \
		--t0 $${TIME_0} --t1 $${TIME_1} \
		--lam1 $${LAM_1} --lam2 $${LAM_2}
	mv DecayModel*N$${NUM_SAMPLES}* fig_decay_q2/
	echo "figure generated"

fig_decay_q2/DecayModel--set_N50_em.png: decay_example.sh solve_problem.py
	. ./decay_example.sh && \
	NUM_SAMPLES=50 && \
	TIME_0=0.5 && \
	TIME_1=1.0 && \
	python solve_problem.py -m $${MODEL} --fontsize $${FONTSIZE} --figsize $${FIGSIZE} \
		--noeval --set --seed $${SEED} -n $${NUM_SAMPLES} \
		--t0 $${TIME_0} --t1 $${TIME_1} \
  	--mc_points $${NUM_MC_SAMPLES}  \
		--lam1 $${LAM_1} --lam2 $${LAM_2}
	mv DecayModel*N$${NUM_SAMPLES}* fig_decay_q2/
	echo "figure generated"

# now for a higher sample size ...
fig_decay_q1/DecayModel--sample_N500_mc.png: decay_example.sh solve_problem.py
	. ./decay_example.sh && \
	NUM_SAMPLES=500 && \
	TIME_0=1 && \
	TIME_1=2 && \
	python solve_problem.py -m $${MODEL} --fontsize $${FONTSIZE} --figsize $${FIGSIZE} \
		--noeval --sample --seed $${SEED} -n $${NUM_SAMPLES} \
		--t0 $${TIME_0} --t1 $${TIME_1} \
		--lam1 $${LAM_1} --lam2 $${LAM_2}
	mv DecayModel*N$${NUM_SAMPLES}* fig_decay_q1/
	echo "figure generated"

fig_decay_q1/DecayModel--set_N500_em.png: decay_example.sh solve_problem.py
	. ./decay_example.sh && \
	NUM_SAMPLES=500 && \
	TIME_0=1 && \
	TIME_1=2 && \
	python solve_problem.py -m $${MODEL} --fontsize $${FONTSIZE} --figsize $${FIGSIZE} \
		--noeval --set --seed $${SEED} -n $${NUM_SAMPLES} \
		--t0 $${TIME_0} --t1 $${TIME_1} \
  	--mc_points $${NUM_MC_SAMPLES}  \
		--lam1 $${LAM_1} --lam2 $${LAM_2}
	mv DecayModel*N$${NUM_SAMPLES}* fig_decay_q1/
	echo "figure generated"

fig_decay_q2/DecayModel--sample_N500_mc.png: decay_example.sh solve_problem.py
	. ./decay_example.sh && \
	NUM_SAMPLES=500 && \
	TIME_0=0.5 && \
	TIME_1=1.0 && \
	python solve_problem.py -m $${MODEL} --fontsize $${FONTSIZE} --figsize $${FIGSIZE} \
		--noeval --sample --seed $${SEED} -n $${NUM_SAMPLES} \
		--t0 $${TIME_0} --t1 $${TIME_1} \
		--lam1 $${LAM_1} --lam2 $${LAM_2}
	mv DecayModel*N$${NUM_SAMPLES}* fig_decay_q2/
	echo "figure generated"

fig_decay_q2/DecayModel--set_N500_em.png: decay_example.sh solve_problem.py
	. ./decay_example.sh && \
	NUM_SAMPLES=500 && \
	TIME_0=0.5 && \
	TIME_1=1.0 && \
	python solve_problem.py -m $${MODEL} --fontsize $${FONTSIZE} --figsize $${FIGSIZE} \
		--noeval --set --seed $${SEED} -n $${NUM_SAMPLES} \
		--t0 $${TIME_0} --t1 $${TIME_1} \
  	--mc_points $${NUM_MC_SAMPLES}  \
		--lam1 $${LAM_1} --lam2 $${LAM_2}
	mv DecayModel*N$${NUM_SAMPLES}* fig_decay_q2/
	echo "figure generated"

clean-decay: clean-decay-a clean-decay-b

clean-decay-a:
	rm -f fig_decay_q1/*

clean-decay-b:
	rm -f fig_decay_q2/*
