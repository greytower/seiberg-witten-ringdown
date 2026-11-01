#Selberg-witten-stacking-analysis
import logging
import warnings
from pathlib import Path
import numpy as np
import h5py
from scipy.signal import welch
from scipy.signal.windows import tukey
import bilby

warnings.filterwarnings("ignore")
logging.getLogger("bilby").setLevel(logging.INFO)

REPO_ROOT = Path(__file__).resolve().parent
if REPO_ROOT.name == "outdir":
    REPO_ROOT = REPO_ROOT.parent

outdir = str(REPO_ROOT / "outdir_top5_stacking_anchored")
label = "Top5_Joint_Cluster_Mutation_Anchored"
bilby.core.utils.setup_logger(outdir=outdir, label=label)

print(f"Working Directory Root: {REPO_ROOT}", flush=True)


# ====================================================================
# 1. SEIBERG-WITTEN / CLUSTER MUTATION FORMALISM
# ====================================================================
def compute_kerr_invariants(M_rem_solar, a_rem):
    G_c3 = 4.925491025543576e-06
    M_sec = M_rem_solar * G_c3
    delta_sqrt = np.sqrt(max(1.0 - a_rem ** 2, 1e-5))
    r_plus = M_sec * (1.0 + delta_sqrt)
    r_minus = M_sec * (1.0 - delta_sqrt)
    Omega_H = a_rem / (2.0 * r_plus)
    kappa = (r_plus - r_minus) / (2.0 * (r_plus ** 2 + (a_rem * M_sec) ** 2))
    return M_sec, Omega_H, kappa


def cluster_seiberg_witten_qnms(M_rem_solar, a_rem, epsilon_qg=0.0):
    M_sec, Omega_H, kappa = compute_kerr_invariants(M_rem_solar, a_rem)

    f_220_base = (1.0 / (2.0 * np.pi * M_sec)) * (1.5251 - 1.1568 * (1.0 - a_rem) ** 0.1292)
    tau_220_base = (M_sec) / (0.7065 - 0.3523 * (1.0 - a_rem) ** 0.2095)

    f_221_base = (1.0 / (2.0 * np.pi * M_sec)) * (1.3673 - 1.0260 * (1.0 - a_rem) ** 0.1628)
    tau_221_base = (M_sec) / (2.1933 - 1.1736 * (1.0 - a_rem) ** 0.2104)

    omega_220 = 2.0 * np.pi * f_220_base - 1j / tau_220_base

    delta_w_re = epsilon_qg * (Omega_H * 0.05)
    delta_w_im = epsilon_qg * (kappa * 0.15)

    w_221_re = 2.0 * np.pi * f_221_base + delta_w_re
    w_221_im = -(1.0 / tau_221_base) - delta_w_im
    omega_221 = w_221_re + 1j * w_221_im

    return omega_220, omega_221


# ====================================================================
# 2. METADATA WITH NR-ANCHORED GAUSSIAN PRIORS
# ====================================================================
EVENTS_METADATA = {
    "GW150914": {
        "gps": 1126259462.423, "ra": 1.37, "dec": -1.26, "psi": 1.57,
        "f_min": 20.0, "f_max": 1024.0,
        "m_mean": 62.0, "m_std": 3.0, "a_mean": 0.67, "a_std": 0.05,
        "H1_file": "H-H1_GWOSC_4KHZ_R1-1126259447-32.hdf5",
        "L1_file": "L-L1_GWOSC_4KHZ_R1-1126259447-32.hdf5"
    },
    "GW190521": {
        "gps": 1242442967.478, "ra": 3.50, "dec": -0.80, "psi": 0.50,
        "f_min": 20.0, "f_max": 512.0,
        "m_mean": 142.0, "m_std": 16.0, "a_mean": 0.72, "a_std": 0.10,
        "H1_file": "H-H1_GWOSC_4KHZ_R1-1242442952-32.hdf5",
        "L1_file": "L-L1_GWOSC_4KHZ_R1-1242442952-32.hdf5"
    },
    "GW200129": {
        "gps": 1264316116.4, "ra": 3.10, "dec": 0.10, "psi": 1.20,
        "f_min": 20.0, "f_max": 1024.0,
        "m_mean": 60.5, "m_std": 4.0, "a_mean": 0.68, "a_std": 0.08,
        "H1_file": "H-H1_GWOSC_4KHZ_R1-1264316101-32.hdf5",
        "L1_file": "L-L1_GWOSC_4KHZ_R1-1264316101-32.hdf5"
    },
    "GW190412": {
        "gps": 1239082262.2, "ra": 3.75, "dec": 0.60, "psi": 0.80,
        "f_min": 30.0, "f_max": 1024.0,
        "m_mean": 37.0, "m_std": 3.5, "a_mean": 0.67, "a_std": 0.07,
        "H1_file": "H-H1_GWOSC_O3a_4KHZ_R1-1239080960-4096.hdf5",
        "L1_file": "L-L1_GWOSC_O3a_4KHZ_R1-1239080960-4096.hdf5"
    },
    "GW250114": {
        "gps": 1420878139.5239, "ra": 2.20, "dec": -0.45, "psi": 1.10,
        "f_min": 20.0, "f_max": 1024.0,
        "m_mean": 62.7, "m_std": 3.5, "a_mean": 0.68, "a_std": 0.07,
        "H1_file": "H-H1_GWOSC_O4b_4KHZ_R1-1420877824-4096.hdf5",
        "L1_file": "L-L1_GWOSC_O4b_4KHZ_R1-1420877824-4096.hdf5"
    }
}

# ====================================================================
# 3. LOAD DATA AND BUILD RESIDUAL FFT CACHES
# ====================================================================
duration = 4.0
sampling_frequency = 4096.0
dt = 1.0 / sampling_frequency
N = int(duration * sampling_frequency)
df = 1.0 / duration
freq_array = np.fft.rfftfreq(N, d=dt)
window = tukey(N, alpha=0.2)

loaded_events = {}
ifo_h1 = bilby.gw.detector.get_empty_interferometer("H1")
ifo_l1 = bilby.gw.detector.get_empty_interferometer("L1")

for ev_name, meta in EVENTS_METADATA.items():
    h1_path = REPO_ROOT / meta["H1_file"]
    l1_path = REPO_ROOT / meta["L1_file"]

    if not (h1_path.exists() and l1_path.exists()):
        print(f"[SKIP] Files for {ev_name} not found locally. Skipping.")
        continue

    print(f"[LOAD] Processing local strain files for {ev_name}...")
    start_time = meta["gps"] - 2.0
    time_array = start_time + np.arange(N) * dt
    mask = (freq_array >= meta["f_min"]) & (freq_array <= meta["f_max"])

    d_fd = {}
    psd_dict = {}

    for ifo_name, file_path in [("H1", h1_path), ("L1", l1_path)]:
        with h5py.File(file_path, "r") as f:
            raw_dataset = f["strain/Strain"]
            gps_start = float(raw_dataset.attrs["Xstart"])
            file_dt = float(raw_dataset.attrs["Xspacing"])
            total_samples = len(raw_dataset)

            s_idx = int(round((start_time - gps_start) / file_dt))
            e_idx = s_idx + N

            sliced = raw_dataset[s_idx:e_idx]

            psd_samples = min(total_samples, int(32.0 / file_dt))
            psd_start = max(0, min(s_idx - int(14.0 / file_dt), total_samples - psd_samples))
            psd_segment = raw_dataset[psd_start:psd_start + psd_samples]

        d_fd[ifo_name] = np.fft.rfft(sliced * window) * dt

        nperseg = int(duration * sampling_frequency)
        f_w, psd_w = welch(psd_segment, fs=1.0 / file_dt, nperseg=nperseg, noverlap=nperseg // 2)
        psd_interp = np.interp(freq_array, f_w, psd_w)
        psd_dict[ifo_name] = np.maximum(psd_interp, 1e-50)

    F_p = {
        "H1": ifo_h1.antenna_response(meta["ra"], meta["dec"], meta["gps"], meta["psi"], "plus"),
        "L1": ifo_l1.antenna_response(meta["ra"], meta["dec"], meta["gps"], meta["psi"], "plus")
    }
    F_c = {
        "H1": ifo_h1.antenna_response(meta["ra"], meta["dec"], meta["gps"], meta["psi"], "cross"),
        "L1": ifo_l1.antenna_response(meta["ra"], meta["dec"], meta["gps"], meta["psi"], "cross")
    }
    time_delay = {
        "H1": ifo_h1.time_delay_from_geocenter(meta["ra"], meta["dec"], meta["gps"]),
        "L1": ifo_l1.time_delay_from_geocenter(meta["ra"], meta["dec"], meta["gps"])
    }

    loaded_events[ev_name] = {
        "meta": meta, "d_fd": d_fd, "psd_dict": psd_dict,
        "mask": mask, "time_array": time_array,
        "F_p": F_p, "F_c": F_c, "time_delay": time_delay
    }


# ====================================================================
# 4. HIERARCHICAL JOINT LIKELIHOOD
# ====================================================================
class JointRingdownStackLikelihood(bilby.Likelihood):
    def __init__(self, loaded_events, dt, df):
        param_dict = {"epsilon_qg": 0.0}
        for ev in loaded_events:
            for p in ["mass", "spin", "amp220", "ratio221", "phi220", "shift221"]:
                param_dict[f"{ev}_{p}"] = None

        super().__init__(parameters=param_dict)
        self.events = loaded_events
        self.dt = dt
        self.df = df
        self.t_rise = 0.001

    def log_likelihood(self):
        eps = self.parameters["epsilon_qg"]
        total_log_l = 0.0

        for ev_name, ev_data in self.events.items():
            M_rem = self.parameters[f"{ev_name}_mass"]
            a_rem = self.parameters[f"{ev_name}_spin"]
            A_220 = self.parameters[f"{ev_name}_amp220"]
            A_221 = A_220 * self.parameters[f"{ev_name}_ratio221"]
            phi_220 = self.parameters[f"{ev_name}_phi220"]
            phi_221 = phi_220 + self.parameters[f"{ev_name}_shift221"]

            w_220, w_221 = cluster_seiberg_witten_qnms(M_rem, a_rem, epsilon_qg=eps)

            for ifo in ["H1", "L1"]:
                t_det = ev_data["time_array"] - (ev_data["meta"]["gps"] + ev_data["time_delay"][ifo])
                t_post = np.where(t_det >= 0.0, t_det, 0.0)
                onset = np.where(
                    t_det < 0.0, 0.0,
                    np.where(t_det < self.t_rise, 0.5 * (1.0 - np.cos(np.pi * t_det / self.t_rise)), 1.0)
                )

                h_tot = (A_220 * np.exp(-1j * (w_220 * t_post + phi_220)) +
                         A_221 * np.exp(-1j * (w_221 * t_post + phi_221))) * onset

                h_det_t = ev_data["F_p"][ifo] * np.real(h_tot) + ev_data["F_c"][ifo] * np.imag(h_tot)
                h_det_fd = np.fft.rfft(h_det_t) * self.dt

                res = ev_data["d_fd"][ifo][ev_data["mask"]] - h_det_fd[ev_data["mask"]]
                psd_m = ev_data["psd_dict"][ifo][ev_data["mask"]]

                total_log_l -= 2.0 * self.df * np.sum(np.abs(res) ** 2 / psd_m)

        if np.isnan(total_log_l):
            return -np.inf
        return float(total_log_l)


# ====================================================================
# 5. PRIORS SETUP (NR-ANCHORED)
# ====================================================================
priors = bilby.core.prior.PriorDict()
priors["epsilon_qg"] = bilby.core.prior.Uniform(minimum=-1.0, maximum=1.0, name="epsilon_qg",
                                                latex_label=r"$\epsilon_{\mathrm{qg}}$")

for ev_name, ev_data in loaded_events.items():
    m_mean = ev_data["meta"]["m_mean"]
    m_std = ev_data["meta"]["m_std"]
    a_mean = ev_data["meta"]["a_mean"]
    a_std = ev_data["meta"]["a_std"]

    priors[f"{ev_name}_mass"] = bilby.core.prior.TruncatedGaussian(mu=m_mean, sigma=m_std, minimum=10.0, maximum=250.0,
                                                                   name=f"{ev_name}_mass")
    priors[f"{ev_name}_spin"] = bilby.core.prior.TruncatedGaussian(mu=a_mean, sigma=a_std, minimum=0.0, maximum=0.99,
                                                                   name=f"{ev_name}_spin")
    priors[f"{ev_name}_amp220"] = bilby.core.prior.LogUniform(1e-22, 1e-18, name=f"{ev_name}_amp220")
    priors[f"{ev_name}_ratio221"] = bilby.core.prior.Uniform(0.0, 1.5, name=f"{ev_name}_ratio221")
    priors[f"{ev_name}_phi220"] = bilby.core.prior.Uniform(0.0, 2 * np.pi, boundary="periodic",
                                                           name=f"{ev_name}_phi220")
    priors[f"{ev_name}_shift221"] = bilby.core.prior.Uniform(-np.pi, np.pi, boundary="periodic",
                                                             name=f"{ev_name}_shift221")

likelihood = JointRingdownStackLikelihood(loaded_events, dt=dt, df=df)

# ====================================================================
# 6. RUN JOINT DYNESTY SAMPLER
# ====================================================================
print("\nStarting NR-Anchored Hierarchical Nested Sampling across all 5 events...", flush=True)

result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    nlive=600,
    sample="rwalk",
    walks=50,
    dlogz=0.1,
    outdir=outdir,
    label=label,
    clean=True
)

# ====================================================================
# 7. FINAL POSTERIOR VERDICT
# ====================================================================
post_eps = result.posterior["epsilon_qg"]
median_eps = np.median(post_eps)
ci_low, ci_high = np.percentile(post_eps, [5.0, 95.0])

print("\n================ 5-EVENT JOINT STACKING RESULTS (NR ANCHORED) ================")
print(f"Log Evidence (ln Z):                          {result.log_evidence:.4f} +/- {result.log_evidence_err:.4f}")
print(f"Joint Cluster Mutation Parameter median:      {median_eps:.4f}")
print(f"90% Credible Interval for epsilon_qg:         [{ci_low:.4f}, {ci_high:.4f}]")
print(f"Credible Interval Compression Ratio:          {(ci_high - ci_low) / 2.0 * 100:.2f}% of prior")
print("==============================================================================")