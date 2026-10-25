#!/bin/bash
# Slurm job for G-CMPS.jl (gauged cMPS ground-state search).
# Code lives in /scratch200/amiror1/cmps. Submit from there:
#     cd /scratch200/amiror1/cmps && mkdir -p logs && sbatch run_gcmps.slurm
# Output (figures + G-CMPS_results.txt) lands in results/<strategy>/ next to G-CMPS.jl;
# Slurm's own log (Julia errors, optimiser warnings) in logs/.

#SBATCH --job-name=gcmps
# TAU "power" cluster (hpcguide.tau.ac.il); allowed account/partition/QOS from
# `check_my_partitions` on the login node
#SBATCH --account=public-erezzohar_v2
#SBATCH --partition=power-general-shared-pool
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1              # one optimisation chain is serial, and BLAS runs on 1 thread (see G-CMPS.jl)
#SBATCH --mem-per-cpu=4G
#SBATCH --time=2-00:00:00              # D = 25 with maxiter = 3000 can take long; TAU maximum is 7-00:00:00
# logs/ must exist before submitting (mkdir -p logs), Slurm does not create it
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=amiror1@mail.tau.ac.il

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# the cluster has no julia module: julia, its versions and all packages are installed
# once on the login node in scratch (home quota is small), with juliaup:
#     export JULIA_DEPOT_PATH=/scratch200/amiror1/.julia JULIAUP_DEPOT_PATH=/scratch200/amiror1/.julia
#     curl -fsSL https://install.julialang.org | sh -s -- --yes --path /scratch200/amiror1/.juliaup
SCRATCH=/scratch200/amiror1
export PATH="$SCRATCH/.juliaup/bin:$PATH"
export JULIA_DEPOT_PATH="$SCRATCH/.julia"
export JULIAUP_DEPOT_PATH="$SCRATCH/.julia"
command -v julia >/dev/null || { echo "julia not found in $SCRATCH/.juliaup/bin, install it with juliaup first"; exit 1; }

# single-threaded linear algebra (small blocks); GR (Plots backend) without a display
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export GKSwstype=100

# set up the project environment here with the packages the code uses; this runs on the
# compute node (precompiling on the login node gets killed) and is quick once installed,
# also completing an install that was interrupted half way
julia --project=. -e 'using Pkg
    pkgs = ["TensorKit", "Zygote", "OptimKit", "Plots", "QuadGK"]
    missing_pkgs = filter(p -> !haskey(Pkg.project().dependencies, p), pkgs)
    isempty(missing_pkgs) || Pkg.add(missing_pkgs)
    Pkg.instantiate(); Pkg.precompile()'

echo "job $SLURM_JOB_ID on $(hostname), $SLURM_CPUS_PER_TASK cpus, started $(date)"
julia --project=. G-CMPS.jl
echo "finished $(date)"
