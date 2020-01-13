#!/bin/sh
FONTSIZE=16
FIGSIZE=5
# MODEL='identity'
SEED=21
# NUM_SAMPLES=100
NUM_MC_SAMPLES=10000
M=1

#python solve_problem.py -m $MODEL --plot --fontsize $FONTSIZE --figsize $FIGSIZE $@
LAM_1=0.5
LAM_2=0.5

#python solve_problem.py -m $MODEL --plot --fontsize $FONTSIZE --figsize $FIGSIZE $@

# low number of samples for input space
# sample-based
solve(){
  python solve_problem.py -m $MODEL --fontsize $FONTSIZE --figsize $FIGSIZE \
    --sample --seed $SEED -n $NUM_SAMPLES \
    --lam1 $LAM_1 --lam2 $LAM_2 $@

  # set-based
  python solve_problem.py -m $MODEL --fontsize $FONTSIZE --figsize $FIGSIZE \
    --set --seed $SEED -o $M -n $NUM_SAMPLES \
    --lam1 $LAM_1 --lam2 $LAM_2 $@

  # with volume emulation
  python solve_problem.py -m $MODEL --fontsize $FONTSIZE --figsize $FIGSIZE \
    --set --seed $SEED -o $M -n $NUM_SAMPLES \
    --mc_points $NUM_MC_SAMPLES  \
    --lam1 $LAM_1 --lam2 $LAM_2 $@
}

solve_varying_samples(){
  FOLDER_NAME=figs_$MODEL/
  mkdir -p $FOLDER_NAME

  NUM_SAMPLES=100
  solve
  NUM_SAMPLES=1000
  solve

  mv t*N$NUM_SAMPLES* $FOLDER_NAME
}

MODEL='skew'
solve_varying_samples --skew=2
solve_varying_samples --skew=1

MODEL='identity'
solve_varying_samples

MODEL='random'
solve_varying_samples
