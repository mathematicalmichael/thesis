all: heatrod linear
	echo "Built all"

linear: examples_linear.sh models.py
	sh examples_linear.sh


# the nuclear option: wipe all examples
clean: \
	clean_heat_q1 \
	clean_heat_q2


# ch 2+3 heatrod example for fifty and five-hundred parameter samples
# ch2 has dependence on size
# ch3 has the introduction of a second set of QoI (measurement locations)
heatrod: heatrod-a heatrod-b models.py

heatrod-a: \
	fig_heatrod_q1/tHeatrodModel--sample_N50_mc.png \
	fig_heatrod_q1/tHeatrodModel--set_N50_em.png \
	fig_heatrod_q1/tHeatrodModel--sample_N500_mc.png \
	fig_heatrod_q1/tHeatrodModel--set_N500_em.png

heatrod-b: \
	fig_heatrod_q2/tHeatrodModel--sample_N50_mc.png \
	fig_heatrod_q2/tHeatrodModel--set_N50_em.png \
	fig_heatrod_q2/tHeatrodModel--sample_N500_mc.png \
	fig_heatrod_q2/tHeatrodModel--set_N500_em.png

clean_heat_a:
	rm -f fig_heatrod_q1/*
	
clean_heat_b:
	rm -f fig_heatrod_q1/*

.PHONY: \
	all clean \
	identity linear heatrod \
	heatrod-50 heatrod-500 \
	clean_heat_q1 clean_heat_q2

# first QoI map

fig_heatrod_q1/tHeatrodModel--set_N500_em.png: heatrod_example.sh solve_problem.py
	. ./heatrod_example.sh && \
	NUM_SAMPLES=500 && \
	LOC_1=0.25 && \
	LOC_2=0.51 && \
	python solve_problem.py -m $${MODEL} --fontsize $${FONTSIZE} --figsize $${FIGSIZE} \
    	--set --seed $${SEED} -o $${M} -n $${NUM_SAMPLES} \
    	--mc_points $${NUM_MC_SAMPLES}  \
    	--t0 $${LOC_1} --t1 $${LOC_2} \
    	--lam1 $${LAM_1} --lam2 $${LAM_2} || true
	mv tHeatrodModel*N$${NUM_SAMPLES}* fig_heatrod_q1/
	echo "figure generated"

fig_heatrod_q1/tHeatrodModel--sample_N500_mc.png: heatrod_example.sh solve_problem.py
	. ./heatrod_example.sh && \
	NUM_SAMPLES=500 && \
	LOC_1=0.25 && \
	LOC_2=0.51 && \
	python solve_problem.py -m $${MODEL} --fontsize $${FONTSIZE} --figsize $${FIGSIZE} \
		--noeval --sample --seed $${SEED} -n $${NUM_SAMPLES} \
		--t0 $${LOC_1} --t1 $${LOC_2} \
		--lam1 $${LAM_1} --lam2 $${LAM_2} || true
	mv tHeatrodModel*N$${NUM_SAMPLES}* fig_heatrod_q1/
	echo "figure generated"

fig_heatrod_q1/tHeatrodModel--set_N50_em.png: heatrod_example.sh solve_problem.py
	. ./heatrod_example.sh && \
	NUM_SAMPLES=50 && \
	LOC_1=0.25 && \
	LOC_2=0.51 && \
	python solve_problem.py -m $${MODEL} --fontsize $${FONTSIZE} --figsize $${FIGSIZE} \
    	--set --seed $${SEED} -o $${M} -n $${NUM_SAMPLES} \
    	--mc_points $${NUM_MC_SAMPLES}  \
    	--t0 $${LOC_1} --t1 $${LOC_2} \
    	--lam1 $${LAM_1} --lam2 $${LAM_2} || true
	mv tHeatrodModel*N$${NUM_SAMPLES}* fig_heatrod_q1/
	echo "figure generated"

fig_heatrod_q1/tHeatrodModel--sample_N50_mc.png: heatrod_example.sh solve_problem.py
	. ./heatrod_example.sh && \
	NUM_SAMPLES=50 && \
	LOC_1=0.25 && \
	LOC_2=0.51 && \
	python solve_problem.py -m $${MODEL} --fontsize $${FONTSIZE} --figsize $${FIGSIZE} \
		--noeval --sample --seed $${SEED} -n $${NUM_SAMPLES} \
		--t0 $${LOC_1} --t1 $${LOC_2} \
		--lam1 $${LAM_1} --lam2 $${LAM_2} || true
	mv tHeatrodModel*N$${NUM_SAMPLES}* fig_heatrod_q1/
	echo "figure generated"

# second QoI map

fig_heatrod_q2/tHeatrodModel--set_N500_em.png: heatrod_example.sh solve_problem.py
	. ./heatrod_example.sh && \
	NUM_SAMPLES=500 && \
	LOC_1=0.67 && \
	LOC_2=0.98 && \
	python solve_problem.py -m $${MODEL} --fontsize $${FONTSIZE} --figsize $${FIGSIZE} \
    	--set --seed $${SEED} -o $${M} -n $${NUM_SAMPLES} \
    	--mc_points $${NUM_MC_SAMPLES}  \
    	--t0 $${LOC_1} --t1 $${LOC_2} \
    	--lam1 $${LAM_1} --lam2 $${LAM_2} || true
	mv tHeatrodModel*N$${NUM_SAMPLES}* fig_heatrod_q2/
	echo "figure generated"

fig_heatrod_q2/tHeatrodModel--sample_N500_mc.png: heatrod_example.sh solve_problem.py
	. ./heatrod_example.sh && \
	NUM_SAMPLES=500 && \
	LOC_1=0.67 && \
	LOC_2=0.98 && \
	python solve_problem.py -m $${MODEL} --fontsize $${FONTSIZE} --figsize $${FIGSIZE} \
		--noeval --sample --seed $${SEED} -n $${NUM_SAMPLES} \
		--t0 $${LOC_1} --t1 $${LOC_2} \
		--lam1 $${LAM_1} --lam2 $${LAM_2} || true
	mv tHeatrodModel*N$${NUM_SAMPLES}* fig_heatrod_q2/
	echo "figure generated"

fig_heatrod_q2/tHeatrodModel--set_N50_em.png: heatrod_example.sh solve_problem.py
	. ./heatrod_example.sh && \
	NUM_SAMPLES=50 && \
	LOC_1=0.67 && \
	LOC_2=0.98 && \
	python solve_problem.py -m $${MODEL} --fontsize $${FONTSIZE} --figsize $${FIGSIZE} \
    	--set --seed $${SEED} -o $${M} -n $${NUM_SAMPLES} \
    	--mc_points $${NUM_MC_SAMPLES}  \
    	--t0 $${LOC_1} --t1 $${LOC_2} \
    	--lam1 $${LAM_1} --lam2 $${LAM_2} || true
	mv tHeatrodModel*N$${NUM_SAMPLES}* fig_heatrod_q2/
	echo "figure generated"

fig_heatrod_q2/tHeatrodModel--sample_N50_mc.png: heatrod_example.sh solve_problem.py
	. ./heatrod_example.sh && \
	NUM_SAMPLES=50 && \
	LOC_1=0.67 && \
	LOC_2=0.98 && \
	python solve_problem.py -m $${MODEL} --fontsize $${FONTSIZE} --figsize $${FIGSIZE} \
		--noeval --sample --seed $${SEED} -n $${NUM_SAMPLES} \
		--t0 $${LOC_1} --t1 $${LOC_2} \
		--lam1 $${LAM_1} --lam2 $${LAM_2} || true
	mv tHeatrodModel*N$${NUM_SAMPLES}* fig_heatrod_q2/
	echo "figure generated"

