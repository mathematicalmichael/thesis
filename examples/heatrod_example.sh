#!/bin/sh
FONTSIZE=16
FIGSIZE=5
MODEL='heatrod'
SEED=21
NUM_SAMPLES=50
NUM_MC_SAMPLES=10000

# just paying attention to characertistic-function densities.
M=1

# locations
# [0.25, 0.51, 0.67, 0.98]

LAM_1=0.15
LAM_2=0.05

#python solve_problem.py -m $MODEL --plot --fontsize $FONTSIZE --figsize $FIGSIZE $@

# low number of samples for input space
# sample-based
solve(){
  python solve_problem.py -m $MODEL --fontsize $FONTSIZE --figsize $FIGSIZE \
    --noeval --sample --seed $SEED -n $NUM_SAMPLES \
    --t0 $LOC_1 --t1 $LOC_2 \
    --lam1 $LAM_1 --lam2 $LAM_2 $@

  # set-based
  # python solve_problem.py -m $MODEL --fontsize $FONTSIZE --figsize $FIGSIZE \
  #   --set --seed $SEED -o $M -n $NUM_SAMPLES \
  #   --t0 $LOC_1 --t1 $LOC_2 \
  #   --lam1 $LAM_1 --lam2 $LAM_2 $@

  # with volume emulation
  python solve_problem.py -m $MODEL --fontsize $FONTSIZE --figsize $FIGSIZE \
    --set --seed $SEED -o $M -n $NUM_SAMPLES \
    --mc_points $NUM_MC_SAMPLES  \
    --t0 $LOC_1 --t1 $LOC_2 \
    --lam1 $LAM_1 --lam2 $LAM_2 $@
}

# qoi in each column
# two different sample sizes
# two pages: low sample size and high sample size
solve_q1q2(){
  # first qoi
  LOC_1=0.25
  LOC_2=0.51
  solve
  mkdir -p fig_heatrod_q1/
  mv tHeatrodModel*N$NUM_SAMPLES* fig_heatrod_q1/
  # second qoi
  LOC_1=0.67
  LOC_2=0.98
  solve
  mkdir -p fig_heatrod_q2/
  mv tHeatrodModel*N$NUM_SAMPLES* fig_heatrod_q2/
}

solve_q1q2
NUM_SAMPLES=500
solve_q1q2
