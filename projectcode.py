import numpy as np
import pandas as pd
import random
import time
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

# ML & Metrics
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    f1_score, roc_curve, confusion_matrix, roc_auc_score, precision_recall_curve
)
from sklearn.impute import SimpleImputer

# Balancing
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    from sklearn.utils import resample

# Boosting
import xgboost as xgb

# Configure
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()
plt.style.use('seaborn-v0_8-darkgrid')

# ===========================================================
# 1. CONFIGURATION
# ===========================================================

class Config:
    # --- Experiment Setup ---
    SEEDS = [42, 101, 777]

    # --- Simulation ---
    FREQ_HZ = 1
    DURATION_MIN = 30
    TOTAL_STEPS = DURATION_MIN * 60

    # --- Hierarchy ---
    NUM_TROOPS = 4
    SOLDIERS_PER_TROOP = 12  # Total 48 soldiers

    # --- Physics ---
    WEARABLE_BATTERY_MAH = 500

    # --- Thresholds ---
    # UPDATED: Changed from 175 to 120 as requested
    HR_CRITICAL = 120
    SPO2_CRITICAL = 85

    # --- AI Configuration ---
    LSTM_WINDOW = 12
    LABEL_NOISE_RATE = 0.05

    # --- TIER 1: UDP (2.4 GHz) ---
    UDP_TX_POWER_DBM = 14.5
    UDP_SENSITIVITY_DBM = -92
    PACKET_SIZE_BYTES = 48
    MAX_RETRIES = 1
    TRANSMIT_SLOTTING = 2

    # --- TIER 2: SATCOM ---
    SAT_BANDWIDTH_BPS = 2400
    SAT_OUTAGE_START = 600
    SAT_OUTAGE_END = 900
    SAT_BURST_PROB = 0.05

    # --- Environments ---
    ENVIRONMENTS = {
        "URBAN": {"n": 3.0, "sigma": 6.8},
        "RURAL": {"n": 2.2, "sigma": 3}
    }
    CURRENT_ENV = "RURAL"

# ===========================================================
# 2. PHYSICS & NETWORK MODELS
# ===========================================================

class NetworkModel:
    @staticmethod
    def calc_link_budget(dist_m, num_active, tx_power_dbm, current_shadowing):
        env = Config.ENVIRONMENTS[Config.CURRENT_ENV]
        n = env["n"]
        pl_d0 = 40.0
        # Path loss with correlated shadowing
        path_loss = pl_d0 + 10 * n * np.log10(max(dist_m, 1.0)) + current_shadowing

        interference = np.random.normal(0, 3.0)
        rssi = tx_power_dbm - path_loss - abs(interference)

        # Sigmoid PDR curve
        pdr = 1.0 / (1.0 + np.exp(-(rssi - Config.UDP_SENSITIVITY_DBM) / 1.5))

        # Burst Loss
        if np.random.random() < 0.02:
            pdr *= 0.6

        # FEC Uplift
        fec_redundancy = 0.06
        pdr = 1.0 - (1.0 - pdr) ** (1.0 + fec_redundancy)

        # Collision Model
        collision_prob = 0.007 * (num_active ** 0.8)
        if np.random.random() < collision_prob:
            pdr = 0.0

        success = 1 if (np.random.random() < pdr) else 0
        return rssi, success

    @staticmethod
    def attempt_transmission(dist_m, num_active, battery_pct, current_shadowing):
        current_tx_power = Config.UDP_TX_POWER_DBM

        if battery_pct < 15.0:
            current_tx_power -= 6.0

        final_rssi = -120

        for attempt in range(Config.MAX_RETRIES + 1):
            effective_active = max(1, int(num_active * (0.6 ** attempt)))
            rssi, success = NetworkModel.calc_link_budget(dist_m, effective_active, current_tx_power, current_shadowing)
            final_rssi = max(final_rssi, rssi)

            if success:
                burst_delay = 10.0 if np.random.random() < 0.05 else 0.0
                latency = 1.0 + (0.08 * dist_m) + np.random.exponential(2.0) + (attempt * 5.0) + burst_delay
                return final_rssi, 0, latency

        return final_rssi, 1, np.nan

class BaseNode:
    def __init__(self, troop_id):
        self.id = troop_id; self.buffer_bytes = 0

    def process_uplink(self, t, incoming_bytes):
        # Buffer overflow simulation
        if self.buffer_bytes > 500000:
            incoming_bytes = int(incoming_bytes * 0.8)

        self.buffer_bytes += incoming_bytes
        sat_connected = not (Config.SAT_OUTAGE_START <= t <= Config.SAT_OUTAGE_END)
        bytes_sent = 0

        if sat_connected:
            bw_jitter = Config.SAT_BANDWIDTH_BPS * np.random.uniform(0.8, 1.2)
            max_bytes_per_sec = bw_jitter / 8
            bytes_sent = min(self.buffer_bytes, max_bytes_per_sec)
            self.buffer_bytes -= bytes_sent
        else:
            self.buffer_bytes = min(self.buffer_bytes, 500000)

        return {"sat_status": 1 if sat_connected else 0, "buffer_current": self.buffer_bytes}

class HealthModel:
    def __init__(self):
        self.hr = 70.0; self.spo2 = 98.0; self.temp = 37.0; self.stress = 10.0
        self.target_hr = 70.0

    def update(self, speed, scenario, fatigue):
        base_hr = 65 + (speed * 12) + (fatigue * 10)

        if scenario == "PANIC":
            self.target_hr = 172 + np.random.normal(0, 8) + (0.1 * self.stress**1.5)
            self.stress += 0.25
            if self.stress > 20: self.spo2 -= 0.05
        elif scenario == "INJURED":
            self.target_hr = 45; self.spo2 -= 0.1
        elif scenario == "NORMAL":
            self.target_hr = base_hr
            self.stress = max(15, self.stress - 0.1)

        noise_hr = np.random.normal(0, 4.0)
        self.hr = (self.hr * 0.85) + (self.target_hr * 0.15) + noise_hr

        if random.random() < 0.015: self.hr += np.random.uniform(15, 25)

        noise_spo2 = np.random.normal(0, 1.0)
        if self.hr > Config.HR_CRITICAL:
            self.spo2 -= np.random.uniform(0.05, 0.2)
        else:
            self.spo2 += (99 - self.spo2) * 0.05

        self.spo2 += noise_spo2
        self.hr = np.clip(self.hr, 30, 230); self.spo2 = np.clip(self.spo2, 60, 100)
        return {"hr": self.hr, "spo2": self.spo2, "temp": self.temp}

class BatteryModel:
    def __init__(self): self.mah = Config.WEARABLE_BATTERY_MAH
    def drain(self, is_tx):
        self.mah = max(0, self.mah - (0.05 if is_tx else 0.01))
        return (self.mah / Config.WEARABLE_BATTERY_MAH) * 100

# ===========================================================
# 3. SIMULATOR CORE
# ===========================================================

class Soldier:
    def __init__(self, uid, troop_id, base_node, scenario_schedule):
        self.id = uid; self.troop_id = troop_id
        self.rel_x = np.random.uniform(-100, 100); self.rel_y = np.random.uniform(-100, 100)
        self.vel_x = 0.0; self.vel_y = 0.0
        self.hlth = HealthModel(); self.batt = BatteryModel()
        self.timeline = ["NORMAL"] * Config.TOTAL_STEPS
        self.shadowing = 0.0

        for start_t, s in sorted(scenario_schedule.items()):
            dur = random.randint(60, 400)
            end_t = min(start_t + dur, Config.TOTAL_STEPS)
            for i in range(start_t, end_t): self.timeline[i] = s

    def step(self, t, troop_center):
        scen = self.timeline[t]
        fatigue_factor = t / Config.TOTAL_STEPS

        target_speed = 5.0 if scen == "RUNNING" else (0.0 if scen == "FALL_EVENT" else 1.0)
        target_speed *= max(0.6, 1.0 - (fatigue_factor * 0.4))

        if scen == "RUNNING":
            self.vel_x += np.random.normal(0, 0.5); self.vel_y += np.random.normal(0, 0.5)
        elif t % 60 == 0 and scen != "FALL_EVENT":
            self.vel_x += np.random.normal(0, 0.8); self.vel_y += np.random.normal(0, 0.8)

        cx, cy = troop_center
        self.vel_x += 0.02 * (cx - self.rel_x); self.vel_y += 0.02 * (cy - self.rel_y)
        curr_speed = np.sqrt(self.vel_x**2 + self.vel_y**2)
        if curr_speed > 0: self.vel_x = (self.vel_x/curr_speed)*target_speed; self.vel_y = (self.vel_y/curr_speed)*target_speed

        self.rel_x += self.vel_x; self.rel_y += self.vel_y
        dist = np.sqrt(self.rel_x**2 + self.rel_y**2)

        vit = self.hlth.update(curr_speed, scen, fatigue_factor)
        should_transmit = ((t + self.id) % Config.TRANSMIT_SLOTTING == 0)
        batt_pct = self.batt.drain(should_transmit)

        fall_risk = 0.001 + (0.003 * fatigue_factor)
        if dist > 100 and scen == "RUNNING" and random.random() < fall_risk:
            scen = "FALL_EVENT"

        acc = [0.0, 0.0, 9.8]
        acc = [a + np.random.normal(0, 0.6) for a in acc]

        if scen == "FALL_EVENT":
            acc = [np.random.uniform(-30, 30) for _ in range(2)] + [np.random.uniform(0, 20)]
        elif scen == "INJURED":
            acc = [9.8, 0.0, 0.5]; acc = [a + np.random.normal(0, 0.1) for a in acc]
        elif curr_speed > 3.0:
            acc[0] += np.random.normal(0, 6.0); acc[1] += np.random.normal(0, 6.0)
            if random.random() < 0.025: acc[2] += np.random.uniform(20, 50)
            else: acc[2] += np.random.normal(0, 8.0)

        # AR1 Shadowing Update
        env_sigma = Config.ENVIRONMENTS[Config.CURRENT_ENV]["sigma"]
        self.shadowing = (0.9 * self.shadowing) + (0.1 * np.random.normal(0, env_sigma))

        if should_transmit:
            rssi, udp_loss, latency = NetworkModel.attempt_transmission(dist, Config.SOLDIERS_PER_TROOP, batt_pct, self.shadowing)
        else:
            rssi, udp_loss, latency = -120, -1, np.nan

        return {
            "id": self.id, "troop_id": self.troop_id, "time": t, "scenario": scen,
            "rel_x": self.rel_x, "rel_y": self.rel_y, "dist_to_base": dist, "speed": curr_speed,
            "hr": vit["hr"], "spo2": vit["spo2"], "temp": vit["temp"],
            "acc_x": acc[0], "acc_y": acc[1], "acc_z": acc[2],
            "udp_rssi": rssi, "udp_loss": udp_loss, "udp_latency": latency, "batt": batt_pct,
            "label_fall": 1 if scen == "FALL_EVENT" else 0,
            "label_emerg": 1 if (vit["hr"] > Config.HR_CRITICAL or vit["spo2"] < Config.SPO2_CRITICAL) else 0
        }

def run_single_simulation(seed_val):
    random.seed(seed_val); np.random.seed(seed_val)
    base_nodes = {i: BaseNode(i) for i in range(1, Config.NUM_TROOPS + 1)}
    soldiers = []
    sid_c = 1
    troops = {i: [] for i in range(1, Config.NUM_TROOPS + 1)}

    for tid in range(1, Config.NUM_TROOPS + 1):
        for _ in range(Config.SOLDIERS_PER_TROOP):
            events = {0: "NORMAL"}
            if random.random() < 0.3: events[random.randint(100, Config.TOTAL_STEPS - 600)] = "PANIC"
            if random.random() < 0.35: events[random.randint(300, Config.TOTAL_STEPS - 400)] = "FALL_EVENT"
            if random.random() < 0.5: events[random.randint(100, Config.TOTAL_STEPS - 600)] = "RUNNING"
            s = Soldier(sid_c, tid, base_nodes[tid], events)
            soldiers.append(s); troops[tid].append(s); sid_c += 1

    data = []
    for t in tqdm(range(Config.TOTAL_STEPS), desc=f"Simulating Seed {seed_val}", leave=False):
        bn_load = {i: 0 for i in range(1, Config.NUM_TROOPS + 1)}
        t_centers = {tid: (np.mean([x.rel_x for x in m]), np.mean([x.rel_y for x in m])) for tid, m in troops.items()}

        current_step_data = []
        for s in soldiers:
            pkt = s.step(t, t_centers[s.troop_id])
            if pkt["udp_loss"] == 0:
                bn_load[s.troop_id] += Config.PACKET_SIZE_BYTES
            current_step_data.append(pkt)

        for tid, bn in base_nodes.items():
            sat = bn.process_uplink(t, bn_load[tid])
            for pkt in current_step_data:
                if pkt["troop_id"] == tid:
                    pkt["sat_status"] = sat["sat_status"]
                    pkt["base_buffer_bytes"] = sat["buffer_current"]

        data.extend(current_step_data)

    return pd.DataFrame(data)

# ===========================================================
# 4. HYBRID AI HUB
# ===========================================================

class HybridAIHub:
    def __init__(self, df, seed=42):
        self.df = df.sort_values(by=['id', 'time']).copy() if not df.empty else df
        self.models = {}
        self.seed = seed

    def feature_eng(self):
        if self.df.empty: return self.df
        self.df["acc_mag"] = np.sqrt(self.df.acc_x**2 + self.df.acc_y**2 + self.df.acc_z**2)
        g = self.df.groupby("id")

        # Requested Engineered Features
        self.df["acc_mean"] = g["acc_mag"].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
        self.df["acc_var"] = g["acc_mag"].rolling(3, min_periods=1).var().fillna(0).reset_index(0, drop=True)
        self.df["acc_peak"] = g["acc_mag"].rolling(3).max().fillna(0).reset_index(0, drop=True)
        self.df["jerk"] = g["acc_mag"].diff().abs().rolling(5).mean().fillna(0).reset_index(0, drop=True)
        self.df["lying_flag"] = (self.df.acc_z.abs() < 5.0).astype(int)

        self.df["hr_delta"] = g["hr"].diff().fillna(0)
        self.df["temp_norm"] = (self.df["temp"] - 37.0) / 2.0
        self.df["hrv"] = g["hr"].diff().rolling(5).std().fillna(0).reset_index(0, drop=True)
        self.df["stress_hr_ratio"] = self.df.hr / (self.df.hrv + 1e-4)

        # Add Noise
        self.df["acc_mean"] += np.random.normal(0, 0.15, size=len(self.df))
        self.df["hrv"] += np.random.normal(0, 0.05, size=len(self.df))
        return self.df

    def create_windows(self):
        if self.df.empty: return [], [], [], []
        seq, y_f, y_h, grp = [], [], [], []
        cols = ["acc_mean", "acc_var", "acc_peak", "jerk", "lying_flag", "hr", "hr_delta", "hrv", "stress_hr_ratio", "temp_norm"]
        self.df[cols] = SimpleImputer(strategy='median').fit_transform(self.df[cols])

        for sid, g in self.df.groupby("id"):
            vals = g[cols].values
            f_tgt, h_tgt = g["label_fall"].values, g["label_emerg"].values
            if len(g) < Config.LSTM_WINDOW + 1: continue

            for i in range(0, len(g) - Config.LSTM_WINDOW, 4):
                win = vals[i:i+Config.LSTM_WINDOW]
                feats = np.concatenate([np.mean(win, 0), np.std(win, 0), np.max(win, 0) - np.min(win, 0)])
                seq.append(feats)

                f_thresh = np.random.randint(3, 5)
                h_thresh = np.random.randint(3, 5)

                y_f.append(1 if sum(f_tgt[i:i+Config.LSTM_WINDOW]) >= f_thresh else 0)
                y_h.append(1 if sum(h_tgt[i:i+Config.LSTM_WINDOW]) >= h_thresh else 0)
                grp.append(sid)
        return np.array(seq), np.array(y_f), np.array(y_h), np.array(grp), cols

    def train(self):
        X, y_f, y_h, grp, feat_names_base = self.create_windows()
        feat_names = []
        for stat in ['mean', 'std', 'range']:
            for f in feat_names_base: feat_names.append(f"{f}_{stat}")

        if len(X) == 0: return {}
        res_f = self._train_model(X, y_f, grp, "Fall_Detection", feat_names)
        res_h = self._train_model(X, y_h, grp, "Health_Alert", feat_names)
        return {**res_f, **res_h}

    def _train_model(self, X, y, grp, name, feat_names):
        if len(X) == 0: return {}
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=self.seed)
        try: train_idx, test_idx = next(gss.split(X, y, grp))
        except: return {}
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        noise_idx = np.random.choice(len(y_tr), int(len(y_tr) * Config.LABEL_NOISE_RATE), replace=False)
        y_tr[noise_idx] = 1 - y_tr[noise_idx]

        pos_count = sum(y_tr==1)
        if pos_count > 5 and HAS_SMOTE:
            k_n = min(3, max(1, pos_count - 1))
            sm = SMOTE(random_state=self.seed, k_neighbors=k_n)
            X_bal, y_bal = sm.fit_resample(X_tr, y_tr)
        else:
            if pos_count < 2: return {}
            X_pos_up, y_pos_up = resample(X_tr[y_tr==1], y_tr[y_tr==1], replace=True, n_samples=int(pos_count*3), random_state=self.seed)
            X_bal, y_bal = np.vstack([X_tr[y_tr==0], X_pos_up]), np.concatenate([y_tr[y_tr==0], y_pos_up])

        clf = xgb.XGBClassifier(n_estimators=350, learning_rate=0.03, max_depth=4,
                                scale_pos_weight=2, eval_metric='logloss', random_state=self.seed)
        clf.fit(X_bal, y_bal)
        y_probs = clf.predict_proba(X_te)[:, 1]

        precision, recall, thresholds = precision_recall_curve(y_te, y_probs)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-6)
        best_thresh = thresholds[np.argmax(f1_scores)] if len(thresholds) > 0 else 0.5
        y_pred = (y_probs > best_thresh).astype(int)

        self.models[name] = {
            "model": clf, "X_test": X_te, "y_test": y_te, "y_probs": y_probs,
            "y_pred": y_pred, "best_thresh": best_thresh, "feature_names": feat_names
        }

        auc_score = roc_auc_score(y_te, y_probs) if len(np.unique(y_te)) > 1 else np.nan

        return {name: {
            'auc': auc_score,
            'f1': f1_score(y_te, y_pred, zero_division=0)
        }}

# ===========================================================
# 5. VISUALIZATION & REPORTING (UPDATED SPECIFICATION)
# ===========================================================

class VisualizationSuite:
    def __init__(self, df, models):
        self.df = df
        self.models = models
        plt.style.use('seaborn-v0_8-darkgrid')

    def generate_full_report(self, agg_metrics):
        if self.df.empty: print("CRITICAL: No data."); return

        tx_attempts = self.df[self.df['udp_loss'] != -1]
        total_packets = len(tx_attempts)
        udp_success = len(tx_attempts[tx_attempts['udp_loss'] == 0])
        tier1_pdr = (udp_success / total_packets) * 100 if total_packets > 0 else 0

        max_buffer = self.df['base_buffer_bytes'].max()

        successful_packets = tx_attempts[tx_attempts['udp_loss'] == 0]
        avg_latency = successful_packets['udp_latency'].mean() if not successful_packets.empty else 0

        falls = self.df[self.df['label_fall'] == 1]['id'].nunique()
        emerg = self.df[self.df['label_emerg'] == 1]['id'].nunique()

        # 1. Textual Outputs (Console Report)
        print("\n" + "="*60)
        print("   WATCHGUARD MISSION REPORT")
        print("="*60)
        print("Mission Status:")
        print(f"  Duration: {Config.DURATION_MIN} minutes.")
        print(f"  Personnel: {Config.NUM_TROOPS} Troops, {Config.NUM_TROOPS*Config.SOLDIERS_PER_TROOP} Soldiers total.")
        print(f"  Environment: {Config.CURRENT_ENV} (based on config).")
        print(f"  Packet Count: Total number of data packets generated (~{total_packets:,}).")
        print("\nNetwork Health Metrics:")
        print(f"  UDP PDR (Packet Delivery Ratio): {tier1_pdr:.2f}% (Percentage of packets successfully received).")
        print(f"  Avg Latency: {avg_latency:.2f} ms (Average time for packet transmission).")
        print(f"  Max Sat Buffer: {max_buffer/1024:.2f} KB (Peak usage during satellite outages).")
        print("\nIncident Report:")
        print(f"  Fall Events: {falls} soldiers affected.")
        print(f"  Critical Health: {emerg} soldiers affected (High HR / Low SpO2).")
        print("\nAI Diagnostics (Aggregated Results):")
        print("  Returns the Mean AUC and Mean F1 Score averaged across the 3 random seeds:")
        for name, stats in agg_metrics.items():
            print(f"  > {name:16s} | Mean AUC: {stats['auc_mean']:.3f} | Mean F1: {stats['f1_mean']:.3f}")
        print("="*60 + "\n")

    def plot_all(self):
        # 2. Graphical Outputs (Visualization Dashboard)
        viz_df = self.df.copy()
        # Jitter for map visualization
        viz_df['rel_x'] += np.random.normal(0, 1.5, len(viz_df))
        viz_df['rel_y'] += np.random.normal(0, 1.5, len(viz_df))

        self.plot_figure_1_tactical_map(viz_df)
        self.plot_figure_2_network_deep_dive()
        self.plot_figure_3_health_dashboard()
        self.plot_figure_4_troop_dynamics()
        self.plot_figure_5_ai_analysis()
        self.plot_figure_6_forensics()

    def plot_figure_1_tactical_map(self, viz_df):
        # Figure 1: Tactical Map
        plt.figure(figsize=(10, 8))
        plt.scatter([0], [0], c='black', marker='^', s=250, label='Base Node (0,0)')
        subset = viz_df.iloc[::20] # Downsample for clarity
        sns.scatterplot(data=subset, x='rel_x', y='rel_y', hue='troop_id', style='scenario', palette='bright', alpha=0.6)

        falls = viz_df[viz_df['label_fall'] == 1]
        if not falls.empty:
            plt.scatter(falls['rel_x'], falls['rel_y'], c='red', marker='x', s=100, label='Fall Events', zorder=10)

        plt.title("Figure 1: Tactical Map (Troop Positions & Incidents)")
        plt.xlabel("Relative X Position (m)")
        plt.ylabel("Relative Y Position (m)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    def plot_figure_2_network_deep_dive(self):
        # Figure 2: Network Deep Dive (2x2 Grid)
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        tx_data = self.df[self.df['udp_loss'] != -1]

        # 1. UDP Link Quality
        sns.scatterplot(data=tx_data.iloc[::20], x='dist_to_base', y='udp_rssi', hue='udp_loss', palette='coolwarm', ax=axes[0,0], alpha=0.5)
        axes[0,0].axhline(Config.UDP_SENSITIVITY_DBM, color='red', ls='--')
        axes[0,0].set_title("UDP Link Quality (RSSI vs Distance)")
        axes[0,0].set_ylabel("RSSI (dBm)")

        # 2. Latency Distribution
        success_tx = tx_data[tx_data['udp_loss']==0]
        if not success_tx.empty:
            sns.histplot(data=success_tx, x='udp_latency', kde=True, ax=axes[0,1], color='green')
        axes[0,1].set_title("Latency Distribution (ms)")

        # 3. Satellite Buffer Usage
        buf = self.df.groupby("time")["base_buffer_bytes"].mean()
        axes[1,0].plot(buf.index, buf.values, color='orange')
        axes[1,0].axvspan(Config.SAT_OUTAGE_START, Config.SAT_OUTAGE_END, color='gray', alpha=0.3, label='Outage Period')
        axes[1,0].set_title("Satellite Buffer Usage (Tier 2)")
        axes[1,0].set_ylabel("Buffer Size (Bytes)")
        axes[1,0].legend()

        # 4. Packet Loss Rate
        loss_t = tx_data.groupby("time")['udp_loss'].mean()
        loss_smooth = loss_t.rolling(window=20, min_periods=1).mean()
        axes[1,1].plot(loss_t.index, loss_t.values, color='red', alpha=0.2)
        axes[1,1].plot(loss_smooth.index, loss_smooth.values, color='darkred', lw=2)
        axes[1,1].set_title("Network Packet Loss Rate (Moving Avg)")
        axes[1,1].set_ylabel("Loss Ratio")

        plt.suptitle("Figure 2: Network Deep Dive")
        plt.tight_layout()
        plt.show()

    def plot_figure_3_health_dashboard(self):
        # Figure 3: Health Dashboard (2x2 Grid)
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        plot_df = self.df.iloc[::15].copy() # Downsample

        # 1. Heart Rate Trends
        sns.lineplot(data=plot_df, x='time', y='hr', hue='troop_id', palette='tab10', ax=axes[0,0])

        # UPDATED: Dynamic label for critical threshold
        axes[0,0].axhline(Config.HR_CRITICAL, color='red', ls='--', label=f'Critical Threshold ({Config.HR_CRITICAL} BPM)')

        axes[0,0].set_title("Heart Rate Trends")
        axes[0,0].legend()

        # 2. SpO2 Levels
        sns.lineplot(data=plot_df, x='time', y='spo2', hue='troop_id', palette='tab10', ax=axes[0,1], legend=False)
        axes[0,1].set_title("SpO2 Levels (%)")
        axes[0,1].set_ylim(80, 100)

        # 3. Heart Rate Variability (HRV)
        sns.lineplot(data=plot_df, x='time', y='hrv', color='purple', ax=axes[1,0])
        axes[1,0].set_title("Heart Rate Variability (HRV)")
        axes[1,0].set_ylabel("HRV (std dev)")

        # 4. Battery Drain
        sns.lineplot(data=self.df.iloc[::100], x='time', y='batt', color='gray', ax=axes[1,1])
        axes[1,1].set_title("Wearable Battery Drain (%)")

        plt.suptitle("Figure 3: Health Dashboard")
        plt.tight_layout()
        plt.show()

    def plot_figure_4_troop_dynamics(self):
        # Figure 4: Troop Dynamics (1x2 Grid)
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        # 1. Movement Speed
        sns.lineplot(data=self.df.iloc[::20], x='time', y='speed', hue='scenario', ax=axes[0])
        axes[0].set_title("Movement Speed (m/s) by Scenario")

        # 2. Troop Dispersion
        sns.boxplot(data=self.df, x='troop_id', y='dist_to_base', ax=axes[1])
        axes[1].set_title("Troop Dispersion (Distance to Base)")

        plt.suptitle("Figure 4: Troop Dynamics")
        plt.tight_layout()
        plt.show()

    def plot_figure_5_ai_analysis(self):
        # Figure 5: AI Analysis (1x3 Grid per Model)
        if not self.models: return

        for name, data in self.models.items():
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # 1. ROC Curve
            fpr, tpr, _ = roc_curve(data['y_test'], data['y_probs'])
            axes[0].plot(fpr, tpr, color='darkorange', lw=2)
            axes[0].plot([0,1],[0,1], 'k--')
            axes[0].set_title(f"{name}: ROC Curve")
            axes[0].set_xlabel("False Positive Rate")
            axes[0].set_ylabel("True Positive Rate")

            # 2. Confusion Matrix
            cm = confusion_matrix(data['y_test'], data['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1])
            axes[1].set_title(f"{name}: Confusion Matrix")
            axes[1].set_xlabel("Predicted")
            axes[1].set_ylabel("True")

            # 3. Feature Importance
            imp = getattr(data['model'], 'feature_importances_', None)
            if imp is not None:
                names = np.array(data['feature_names'])
                indices = np.argsort(imp)[::-1][:10]
                axes[2].barh(range(10), imp[indices], align='center')
                axes[2].set_yticks(range(10))
                axes[2].set_yticklabels(names[indices])
                axes[2].invert_yaxis()
                axes[2].set_title(f"{name}: Top 10 Features")

            plt.suptitle(f"Figure 5: AI Analysis - {name}")
            plt.tight_layout()
            plt.show()

    def plot_figure_6_forensics(self):
        # Figure 6: Forensics (Event Reconstruction)
        fall_df = self.df[self.df['label_fall'] == 1]
        if fall_df.empty: return

        # Pick the first fall event to visualize
        sid = fall_df.iloc[0]['id']
        t_fall = fall_df.iloc[0]['time']

        # Window +/- 10 seconds
        win = self.df[(self.df['id'] == sid) & (self.df['time'].between(t_fall - 10, t_fall + 10))]

        plt.figure(figsize=(12, 5))
        plt.plot(win['time'], win['acc_z'], label='Acc Z-axis', color='blue')
        plt.plot(win['time'], win['acc_mag'], label='Acc Magnitude', color='black', ls='--')


# ===========================================================
# 6. MULTI-SEED EXECUTION
# ===========================================================

if __name__ == "__main__":
    metrics_history = {"Fall_Detection": {"auc": [], "f1": []}, "Health_Alert": {"auc": [], "f1": []}}
    last_df = pd.DataFrame()
    last_models = {}

    print(f"Running robustness check across {len(Config.SEEDS)} seeds: {Config.SEEDS}")

    for i, seed in enumerate(Config.SEEDS):
        logger.info(f"Starting Run {i+1}/{len(Config.SEEDS)} (Seed {seed})...")
        df_sim = run_single_simulation(seed)

        hub = HybridAIHub(df_sim, seed=seed)
        hub.feature_eng()
        metrics = hub.train()

        for k in metrics_history.keys():
            if k in metrics:
                metrics_history[k]["auc"].append(metrics[k]["auc"])
                metrics_history[k]["f1"].append(metrics[k]["f1"])

        # Save the last run for visualization
        if i == len(Config.SEEDS) - 1:
            last_df = hub.df
            last_models = hub.models

    # Aggregate Metrics
    agg_metrics = {}
    for name, vals in metrics_history.items():
        clean_auc = [v for v in vals["auc"] if not np.isnan(v)]
        clean_f1 = [v for v in vals["f1"] if not np.isnan(v)]

        if clean_auc:
            agg_metrics[name] = {
                "auc_mean": np.mean(clean_auc),
                "f1_mean": np.mean(clean_f1)
            }
        else:
             agg_metrics[name] = {"auc_mean": 0.0, "f1_mean": 0.0}

    # Generate Reports
    viz = VisualizationSuite(last_df, last_models)

    # 1. Textual Report
    viz.generate_full_report(agg_metrics)

    # 2. Graphical Dashboard
    print("Generating Visualization Dashboard...")
    viz.plot_all()

    logger.info("WATCHGUARD Simulation Complete.")
