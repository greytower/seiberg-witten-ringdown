# seiberg-witten-ringdown

seiberg-Witten Ringdown Constraints (O4b)This repository contains the standalone Bayesian analysis pipeline and data ingestion scripts used in the manuscript "Constraining Quantum Horizon Geometry with Gravitational-Wave Ringdowns: A Multi-Event Seiberg-Witten Analysis."Our pipeline utilizes a hierarchical Whittle likelihood and the dynesty nested sampler to directly constrain the quantum topological mutation parameter ($\epsilon_{\mathrm{qg}}$) using raw GWOSC strain data, bypassing surrogate model biases present in standard CBC pipelines.Directory Structuredata/: Scripts to fetch HDF5 strain data and native Welch PSDs from the Gravitational Wave Open Science Center (GWOSC). Includes configurations for GW150914, GW190521, GW200129, GW190412, and GW250114.models/: The core physics module containing the Seiberg-Witten connection formulas and exact WKB frequency/damping shifts for the $(2,2,1)$ overtone.pipeline/: The Whittle likelihood engine, data-tapering utilities, and hierarchical dynesty sampler configurations.results/: Output directories for MCMC chains, Bayes factors, and scripts to generate the posterior evolution tables and corner plots.InstallationClone this repository and install the required dependencies (we recommend using a virtual environment):git clone https://github.com/greytower/seiberg-witten-ringdown.git
cd seiberg-witten-ringdown
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
Reproducing the ResultsTo reproduce the 5-event hierarchical stack presented in Table I of the manuscript (agnostic priors):python pipeline/run_hierarchical_stack.py --events all --priors agnostic --nlive 1500
To run the Numerical Relativity (NR) anchored analysis to test the physical limits of the overtone damping time:python pipeline/run_hierarchical_stack.py --events all --priors nr_anchored --nlive 1500
Note: Depending on your hardware, a full 31-dimensional nested sampling run with 1500 live points may take several hours to converge.CitationIf you use this code, pipeline, or exact WKB methodology in your research, please cite our paper:@article{Brant_2026_SW_Ringdown,
  title = {Constraining Quantum Horizon Geometry with Gravitational-Wave Ringdowns: A Multi-Event Seiberg-Witten Analysis},
  author = {Brant, William},
  journal = {Physical Review D},
  year = {2026},
  note = {Submitted}
}
