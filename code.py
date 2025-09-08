import numpy as np
import subprocess
import damask
import shutil
import os
import json
import yaml
import logging
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.plots import plot_convergence
import sys

# ==================================
#          LOGGING SETUP
# ==================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimization_damask3_thermo_mech_corrected.log', mode='w'), # New log file
        logging.StreamHandler()
    ]
)

# ==================================
#          CONFIGURATION
# ==================================
class Config:
    MAX_PARALLEL_JOBS = 1
    CORES_PER_JOB = 100
    MAX_ITER = 100
    
    SPLIT_STRAIN = 0.25 
    WEIGHT_STRESS_LOW, WEIGHT_STRESS_HIGH = 0.3, 0.7
    WEIGHT_HARD_LOW, WEIGHT_HARD_HIGH = 0.3, 0.7
    LAMBDA_BIAS = 5.0
    STRESS_ERROR_WEIGHT, HARDENING_ERROR_WEIGHT = 0.4, 0.6
    HARDENING_SMOOTHING_WINDOW = 10 
    EXP_HARDENING_STEP = 20 
    SIM_HARDENING_STEP = 1
    STRAIN_THRESHOLD = 0.5 
    BASE_DIR = Path(os.getcwd())
    RESULTS_DIR = BASE_DIR / "optimization_results_damask3_thermo_mech_corrected_v4" # New results directory for fresh start
    PLOTS_DIR = RESULTS_DIR / "plots"
    MATERIAL_TEMPLATE = BASE_DIR / "material.yaml"
    GEOM_FILE = BASE_DIR / "geom.vti"
    LOAD_FILE = BASE_DIR / "tensionX.yaml"
    EXP_DATA_FILE = BASE_DIR / "exp.txt"
    
    # REVISED PARAMETER SPACE for much better match
PARAM_SPACE = [
    # CRITICAL: Lower this range to match the experimental yield stress
    Real(name='tau_0', low=20e6, high=40e6, prior='log-uniform'),

    Real(name='Q_sl', low=1.0e-19, high=3.0e-19, prior='log-uniform'),
    Real(name='p_sl', low=0.1, high=0.9),
    Real(name='Q_cl', low=2.0e-19, high=8.0e-19),
    Real(name='q_sl', low=1.0, high=2.5),
    
    # The previous range for D_a worked well, let's keep it
    Real(name='D_a',  low=2.0, high=12.0),
    
    # Refine this to prevent excessively high initial stress
    Real(name='rho_dip_0', low=5.0e10, high=5.0e11, prior='log-uniform'),

    # The range for B is good, it allows the optimizer to find the twinning onset
    Real(name='B', low=0.001, high=0.05, prior='log-uniform'),
    
    # CRITICAL: Drastically reduce the hardening coefficients to a more physical range.
    # This will lower the hardening rate and overall stress.
    Real(name='h_val_1', low=0.5, high=4.0),
    Real(name='h_val_2', low=1.0, high=8.0)
]

YAML_PLASTIC_PATH = ['phase', 'Austenite', 'mechanical', 'plastic']

    @classmethod
    def setup(cls):
        cls.RESULTS_DIR.mkdir(exist_ok=True)
        cls.PLOTS_DIR.mkdir(exist_ok=True)
        if not (cls.RESULTS_DIR / "optimization_history.json").exists():
            with open(cls.RESULTS_DIR / "optimization_history.json", 'w') as f:
                json.dump({"iterations": []}, f)
        logging.info(f"Corrected run. Results will be stored in: {cls.RESULTS_DIR}")
        logging.info(f"Plots will be stored in: {cls.PLOTS_DIR}")

# ==================================
#       I/O AND PREPARATION
# ==================================
def read_experimental_data():
    try:
        data = np.genfromtxt(Config.EXP_DATA_FILE, delimiter='\t', skip_header=1)
        strain_percent, stress_mpa = data[:, 0], data[:, 1]
        logging.info(f"Loaded {len(strain_percent)} experimental data points.")
        return {'stress_strain': np.column_stack((strain_percent, stress_mpa))}
    except Exception as e:
        logging.error(f"Error reading experimental data: {e}", exc_info=True)
        raise

def update_material_config(params, run_dir):
    material_file_path = run_dir / Config.MATERIAL_TEMPLATE.name
    try:
        with open(Config.MATERIAL_TEMPLATE, 'r') as f:
            material_data = yaml.safe_load(f)

        # Pop the special hardening values from the params dictionary
        h_val_1 = params.pop('h_val_1')
        h_val_2 = params.pop('h_val_2')

        # Reconstruct the h_sl-sl list based on the original pattern
        # This correctly builds the list to be written to the YAML file
        scaled_h_sl_sl = [h_val_1, h_val_1, h_val_2, h_val_1, h_val_2, h_val_2, h_val_1]

        # Navigate to the plastic properties section
        current_level = material_data
        for key in Config.YAML_PLASTIC_PATH:
            current_level = current_level[key]

        # Explicitly set the newly constructed h_sl-sl list
        current_level['h_sl-sl'] = scaled_h_sl_sl

        # Apply the rest of the parameters from the optimizer
        for pname, pvalue in params.items():
            if pname in current_level:
                formatted_value = float(f'{pvalue:.6e}')
                if isinstance(current_level[pname], list):
                    current_level[pname] = [formatted_value] * len(current_level[pname])
                else:
                    current_level[pname] = formatted_value

        with open(material_file_path, 'w') as f:
            yaml.dump(material_data, f, default_flow_style=None, sort_keys=False)
        return material_file_path
    except Exception as e:
        logging.error(f"Error during material config update: {e}", exc_info=True)
        raise

def run_damask_simulation(run_dir, material_file_path_in_run_dir):
    geom_file_src, load_file_src, material_file_src = Config.GEOM_FILE, Config.LOAD_FILE, Config.MATERIAL_TEMPLATE
    geom_file_dest, load_file_dest = run_dir / geom_file_src.name, run_dir / load_file_src.name
    shutil.copy(geom_file_src, geom_file_dest); shutil.copy(load_file_src, load_file_dest)
    cmd = ['DAMASK_grid', '--load', load_file_dest.name, '--geom', geom_file_dest.name, '--material', material_file_path_in_run_dir.name]
    output_base_name = f"{geom_file_src.stem}_{load_file_src.stem}_{material_file_src.stem}"
    hdf5_file = run_dir / f"{output_base_name}.hdf5"
    logging.info(f"Executing: {' '.join(map(str, cmd))}")
    logging.info(f"Expecting HDF5 at: {hdf5_file}")
    try:
        env = os.environ.copy(); env['DAMASK_NUM_THREADS'] = str(Config.CORES_PER_JOB)
        process = subprocess.run(cmd, cwd=run_dir, capture_output=True, text=True, encoding='utf-8')
        if process.stdout: logging.debug(f"DAMASK STDOUT:\n{process.stdout}")
        if process.stderr: logging.warning(f"DAMASK STDERR:\n{process.stderr}")
        if process.returncode == 0 and hdf5_file.exists():
            logging.info("DAMASK_grid completed successfully.")
            return True, hdf5_file
        else:
            logging.error(f"DAMASK_grid failed. Return code: {process.returncode}.")
            if not hdf5_file.exists(): logging.error(f"Expected output file {hdf5_file} was NOT created.")
            return False, None
    except Exception as e:
        logging.error(f"An error occurred running DAMASK: {e}", exc_info=True)
        return False, None

# ==================================
#      POST-PROCESSING & OBJECTIVE
# ==================================
def process_damask_results(hdf5_file_path):
    logging.info(f"Processing DAMASK results from: {hdf5_file_path}")
    try:
        res = damask.Result(str(hdf5_file_path))
        res.add_stress_Cauchy()
        res.add_strain()
        res.add_equivalent_Mises('sigma')
        res.add_equivalent_Mises('epsilon_V^0.0(F)')
        incs = res.increments
        if not incs:
            logging.error("No increments found in the result file.")
            return None
        stress_key = 'sigma_vM'
        strain_key = 'epsilon_V^0.0(F)_vM'
        mises_stress_data = res.get(stress_key)
        mises_strain_data = res.get(strain_key)
        if mises_stress_data is None or mises_strain_data is None:
            logging.error(f"Failed to get data for keys '{stress_key}' or '{strain_key}'.")
            return None
        avg_S = np.array([np.average(val) for val in mises_stress_data.values()]) / 1e6
        avg_s = np.array([np.average(val) for val in mises_strain_data.values()]) * 100
        sim_data = np.column_stack((avg_S, avg_s))
        logging.info(f"Successfully extracted {len(sim_data)} simulation data points.")
        return sim_data
    except Exception as e:
        logging.error(f"Error processing HDF5 file {hdf5_file_path}: {e}", exc_info=True)
        return None

# ==================================
#      STRESS–STRAIN & HARDENING
# ==================================
def compute_seg_masks(s, ss): return (s >= 0) & (s < ss), (s >= ss)

def calculate_segmented_stress_error(sim, exp):
    exp_s, exp_S = exp['stress_strain'][:, 0], exp['stress_strain'][:, 1]
    if sim.size == 0 or sim[-1, 1] < exp_s[0] or sim[0, 1] > exp_s[-1]: return 1e6
    sim_S_i = np.interp(exp_s, sim[:, 1], sim[:, 0], left=np.nan, right=np.nan)
    v = ~np.isnan(sim_S_i); diff = sim_S_i[v] - exp_S[v]
    if diff.size == 0: return 1e6
    lm, hm = compute_seg_masks(exp_s[v], Config.SPLIT_STRAIN)
    mse_l = np.mean(diff[lm]**2) if np.any(lm) else 0; mse_h = np.mean(diff[hm]**2) if np.any(hm) else 0
    rmse = np.sqrt(Config.WEIGHT_STRESS_LOW * mse_l + Config.WEIGHT_STRESS_HIGH * mse_h)
    logging.info(f"Segmented stress RMSE: {rmse:.3f}")
    return rmse

def compute_stress_bias(sim, exp):
    exp_s, exp_S = exp['stress_strain'][:, 0], exp['stress_strain'][:, 1]
    if sim.size == 0 or sim[-1, 1] < exp_s[0] or sim[0, 1] > exp_s[-1]: return 0.0
    sim_S_i = np.interp(exp_s, sim[:, 1], sim[:, 0], left=np.nan, right=np.nan)
    diff = sim_S_i[~np.isnan(sim_S_i)] - exp_S[~np.isnan(sim_S_i)]; bias = np.mean(diff) if diff.size > 0 else 0.0
    logging.info(f"Signed bias (sim - exp): {bias:.3f} MPa"); return bias

def calculate_hardening_curve(s, S, step):
    n = len(s);
    if n <= step: return np.array([]), np.array([])
    v = np.where(np.diff(s) > 1e-9)[0] + 1; v = np.insert(v, 0, 0); s, S = s[v], S[v]
    if len(s) <= step: return np.array([]), np.array([])
    ds, dE = S[step:] - S[:-step], s[step:] - s[:-step]; vd = dE > 1e-9
    if not np.any(vd): return np.array([]), np.array([])
    h = np.full_like(dE, np.nan); h[vd] = ds[vd] / dE[vd]; mid_s = 0.5 * (s[:-step] + s[step:])
    return mid_s[vd], h[vd]

def calculate_hardening_error(sim, exp):
    exp_s, exp_S = exp['stress_strain'][:, 0], exp['stress_strain'][:, 1]
    exp_mid, exp_h_raw = calculate_hardening_curve(exp_s, exp_S, Config.EXP_HARDENING_STEP)
    if exp_mid.size == 0: return 1e6, *[np.array([])]*3
    w = Config.HARDENING_SMOOTHING_WINDOW
    exp_h = np.convolve(exp_h_raw, np.ones(w)/w, 'same') if w > 0 and len(exp_h_raw) >= w else exp_h_raw
    if sim.size == 0: return 1e6, exp_mid, exp_h, np.full_like(exp_mid, np.nan)
    sim_s, sim_S = sim[:, 1], sim[:, 0]
    sim_mid, sim_h = calculate_hardening_curve(sim_s, sim_S, Config.SIM_HARDENING_STEP)
    if sim_mid.size == 0: return 1e6, exp_mid, exp_h, np.full_like(exp_mid, np.nan)
    im = (exp_mid >= sim_mid[0]) & (exp_mid <= sim_mid[-1])
    exp_mid_i, exp_h_i = exp_mid[im], exp_h[im]
    if exp_mid_i.size == 0: return 1e6, exp_mid, exp_h, np.full_like(exp_mid, np.nan)
    sim_h_i = np.interp(exp_mid_i, sim_mid, sim_h); diff = sim_h_i - exp_h_i
    lm, hm = compute_seg_masks(exp_mid_i, Config.SPLIT_STRAIN)
    mse_l, mse_h = (np.mean(diff[m]**2) if np.any(m) else 0 for m in [lm, hm])
    error = np.sqrt(Config.WEIGHT_HARD_LOW * mse_l + Config.WEIGHT_HARD_HIGH * mse_h)
    logging.info(f"Segmented hardening RMSE: {error:.3f}")
    sim_h_plot = np.full_like(exp_mid, np.nan); sim_h_plot[im] = sim_h_i
    return error, exp_mid, exp_h, sim_h_plot

# ==================================
#       PLOTTING AND HISTORY
# ==================================
def plot_current_results(sim, exp, p, run_id, exp_hm, exp_hs, sim_h):
    fig, ax1 = plt.subplots(figsize=(12, 7)); lines, labels = [], []
    le, = ax1.plot(exp['stress_strain'][:, 0], exp['stress_strain'][:, 1], 'b-', lw=2, label='Exp. S-S')
    lines.append(le); labels.append(le.get_label())
    if sim.size > 0:
        ls, = ax1.plot(sim[:, 1], sim[:, 0], 'r--', lw=2, label=f'Sim (Run {run_id}) S-S')
        lines.append(ls); labels.append(ls.get_label())
    ax1.set_xlabel("Strain (%)"); ax1.set_ylabel("Stress (MPa)", color='b'); ax1.tick_params(axis='y', labelcolor='b'); ax1.grid(True, linestyle=':')
    ax2 = ax1.twinx()
    if exp_hm.size > 0:
        leh, = ax2.plot(exp_hm, exp_hs, 'g-', lw=2, label='Exp. Hardening'); lines.append(leh); labels.append(leh.get_label())
        lsh, = ax2.plot(exp_hm, sim_h, 'm--', lw=2, label=f'Sim (Run {run_id}) Hardening'); lines.append(lsh); labels.append(lsh.get_label())
        ax2.set_ylabel("Hardening Rate", color='g'); ax2.tick_params(axis='y', labelcolor='g')
        vh = (exp_hm >= Config.STRAIN_THRESHOLD) & np.isfinite(exp_hs) & np.isfinite(sim_h)
        if np.any(vh):
            vals = np.concatenate([exp_hs[vh], sim_h[vh]])
            if vals.size > 0: vmin, vmax = np.min(vals), np.max(vals); ax2.set_ylim(vmin - 0.1*(vmax-vmin), vmax + 0.1*(vmax-vmin))
    ax1.legend(lines, labels, loc='center left'); p_txt = "Params:\n" + "\n".join(f"{k}: {v:.4e}" for k, v in p.items())
    plt.figtext(0.99, 0.5, p_txt, va='center', ha='left', bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    plt.title(f"DAMASK 3 Opt: Run {run_id}"); plt.subplots_adjust(right=0.80)
    out_file = Config.PLOTS_DIR / f"comparison_run_{run_id}.png"
    plt.savefig(out_file, bbox_inches='tight'); plt.close(fig); logging.info(f"Saved comparison plot: {out_file}")

def update_optimization_history(p, s_err, h_err, b_pen, t_err, run_id):
    hf = Config.RESULTS_DIR / "optimization_history.json"
    try:
        with open(hf, 'r') as f: hist = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError): hist = {"iterations": []}
    # Re-add h_val_1 and h_val_2 to params for logging
    full_params = p.copy()
    if 'h_val_1' not in full_params: # A bit of a workaround to log all params
        pass # In a more complex setup, you might pass the full set down
    it_data = {"run_id": run_id, "timestamp": datetime.now().isoformat(), "parameters": {k: f"{v:.6e}" for k,v in p.items()}, "stress_error_rmse": s_err, "bias_penalty": b_pen, "hardening_error_rmse": h_err, "total_error_objective": t_err}
    hist["iterations"].append(it_data)
    with open(hf, 'w') as f: json.dump(hist, f, indent=2)

def plot_optimization_progress():
    history_file = Config.RESULTS_DIR / "optimization_history.json"
    try:
        with open(history_file, 'r') as f: df = pd.DataFrame(json.load(f)['iterations'])
        if df.empty: return
        for col in ['stress_error_rmse', 'hardening_error_rmse', 'bias_penalty', 'total_error_objective']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['total_error_objective'], inplace=True)
    except Exception as e:
        logging.error(f"Could not plot history: {e}"); return
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(df['run_id'], df['stress_error_rmse'], 'b-o', label="Stress RMSE"); axes[0].plot(df['run_id'], df['hardening_error_rmse'], 'r-s', label="Hardening RMSE"); axes[0].plot(df['run_id'], df['bias_penalty'], 'c-^', label="Bias Penalty", alpha=0.6); axes[0].plot(df['run_id'], df['total_error_objective'], 'k-x', ms=8, lw=2, label="Total Objective")
    axes[0].set_ylabel("Error / Objective Value"); axes[0].set_title("Optimization Progress"); axes[0].legend(); axes[0].grid(True, linestyle=':');
    if any(e > 0 for e in df['total_error_objective']): axes[0].set_yscale('log')
    param_names = [p.name for p in Config.PARAM_SPACE]
    # Ensure parameters are dictionaries of floats
    df['parameters'] = df['parameters'].apply(lambda p: {k: float(v) for k, v in p.items()})
    for name in param_names:
        axes[1].plot(df['run_id'], [p.get(name) for p in df['parameters']], '-o', label=name)
    axes[1].set_xlabel("Iteration (Run ID)"); axes[1].set_ylabel("Parameter Value"); axes[1].set_title("Parameter Evolution"); axes[1].legend(loc='best', ncol=max(1, len(param_names)//3)); axes[1].grid(True, linestyle=':')
    plt.tight_layout(); plt.savefig(Config.PLOTS_DIR / 'optimization_progress.png'); plt.close(fig); logging.info("Saved optimization progress plot.")

# ==================================
#         EVALUATION AND MAIN
# ==================================
def run_single_evaluation(params, exp_data, run_id):
    logging.info(f"\n{'='*10} Starting Evaluation {run_id} {'='*10}")
    # Log all parameters, including h_val_1 and h_val_2 before they are popped
    all_params_for_logging = params.copy()
    logging.info(f"Parameters: {', '.join(f'{k}={v:.4e}' for k,v in all_params_for_logging.items())}")
    
    run_dir = Config.RESULTS_DIR / f"run_{run_id}"; run_dir.mkdir(exist_ok=True)
    t_err, s_err, h_err, b_pen = 1e7, 1e6, 1e6, 0.0
    sim_data_for_plot = np.array([])
    hardening_data_for_plot = [np.array([])]*3
    try:
        mat_file = update_material_config(params, run_dir)
        success, hdf5 = run_damask_simulation(run_dir, mat_file)
        if success and hdf5:
            sim_data = process_damask_results(hdf5)
            if sim_data is not None and sim_data.size > 0:
                sim_data_for_plot = sim_data
                s_err = calculate_segmented_stress_error(sim_data, exp_data)
                bias = compute_stress_bias(sim_data, exp_data); b_pen = Config.LAMBDA_BIAS * abs(bias)
                h_err, exp_mid, exp_h, sim_h = calculate_hardening_error(sim_data, exp_data)
                hardening_data_for_plot = [exp_mid, exp_h, sim_h]
                t_err = (Config.STRESS_ERROR_WEIGHT * (s_err + b_pen) + Config.HARDENING_ERROR_WEIGHT * h_err)
                logging.info(f"Eval {run_id} done: StressRMSE={s_err:.4f}, BiasPen={b_pen:.4f}, HardRMSE={h_err:.4f} ==> TotalObjective={t_err:.4f}")
            else: logging.error(f"Evaluation {run_id} failed during results processing.")
        else: logging.error(f"Evaluation {run_id} failed during simulation execution.")
    except Exception as e:
        logging.exception(f"A critical error occurred during evaluation {run_id}: {e}")
    finally:
        plot_current_results(sim_data_for_plot, exp_data, all_params_for_logging, run_id, *hardening_data_for_plot)
        update_optimization_history(all_params_for_logging, s_err, h_err, b_pen, t_err, run_id)
    return t_err

@use_named_args(Config.PARAM_SPACE)
def objective(**params):
    try:
        exp_data = read_experimental_data()
    except Exception: return 1e10
    hf = Config.RESULTS_DIR / "optimization_history.json"
    try:
        with open(hf, 'r') as f: run_id = len(json.load(f)['iterations']) + 1
    except (FileNotFoundError, json.JSONDecodeError): run_id = 1
    total_error = run_single_evaluation(params, exp_data, run_id)
    plot_optimization_progress()
    return total_error

def main():
    logging.info("="*30 + "\n Starting DAMASK 3 Thermo-Mechanical Parameter Optimization (Corrected Run) \n" + "="*30)
    Config.setup()
    for f in [Config.MATERIAL_TEMPLATE, Config.GEOM_FILE, Config.LOAD_FILE, Config.EXP_DATA_FILE]:
        if not f.exists():
            logging.critical(f"CRITICAL: File not found: {f}. Exiting."); return
    try:
        read_experimental_data()
    except Exception:
        logging.critical("Could not read exp data. Exiting.", exc_info=True); return
    logging.info("Config and files OK. Starting optimization...")
    start_time = datetime.now()
    res = gp_minimize(func=objective, dimensions=Config.PARAM_SPACE, n_calls=Config.MAX_ITER,
                      n_initial_points=max(1, Config.MAX_ITER // 5), verbose=True) # n_initial_points can be adjusted
    end_time = datetime.now()
    logging.info("="*30 + f"\n Optimization completed in: {end_time - start_time}\n" + "="*30)
    logging.info(f"Minimum objective value found: {res.fun:.6f}")
    best_params = {p.name: val for p, val in zip(Config.PARAM_SPACE, res.x)}
    logging.info("Best parameter set found:")
    for name, val in best_params.items(): logging.info(f"  {name}: {val:.6e}")
    with open(Config.RESULTS_DIR / 'best_parameters.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    logging.info(f"Best parameters saved to: {Config.RESULTS_DIR / 'best_parameters.json'}")
    try:
        plot_convergence(res); plt.savefig(Config.PLOTS_DIR / 'final_convergence.png'); plt.close()
        logging.info(f"Final skopt convergence plot saved to: {Config.PLOTS_DIR / 'final_convergence.png'}")
    except Exception as e:
        logging.warning(f"Could not generate final skopt convergence plot: {e}")
    logging.info("Optimization process finished.")

if __name__ == "__main__":
    main()