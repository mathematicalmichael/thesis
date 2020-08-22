all: heatrod
	echo "Built all"

linear:
	sh examples_linear.sh

heatrod: \
	fig_heatrod_q1/tHeatrodModel--sample_N500_mc.png \
	fig_heatrod_q1/tHeatrodModel--set_N500_em.png \
	fig_heatrod_q2/tHeatrodModel--sample_N500_mc.png \
	fig_heatrod_q2/tHeatrodModel--set_N500_em.png
	echo "Heatrod up to date."

.PHONY: \
	all clean \
	identity linear heatrod \
	clean_heat_q1 clean_heat_q2


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

clean_heat_q1:
	rm -f fig_heatrod_q1/*
	
clean_heat_q2:
	rm -f fig_heatrod_q1/*

# the nuclear option: wipe all examples
clean: \
	clean_heat_q1 \
	clean_heat_q2
