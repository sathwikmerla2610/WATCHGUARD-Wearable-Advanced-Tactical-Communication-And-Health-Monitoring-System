import streamlit as st
import numpy as np
import pandas as pd
import random
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import deque
import warnings
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, roc_curve, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
import xgboost as xgb
from datetime import datetime

warnings.filterwarnings('ignore')

# ===========================================================
# CONFIGURATION
# ===========================================================

class Config:
    SEEDS = [42, 101, 777]
    FREQ_HZ = 1
    DURATION_MIN = 30
    TOTAL_STEPS = DURATION_MIN * 60
    NUM_TROOPS = 4
    SOLDIERS_PER_TROOP = 12
    WEARABLE_BATTERY_MAH = 500
    HR_CRITICAL = 120
    SPO2_CRITICAL = 85
    LSTM_WINDOW = 12
    LABEL_NOISE_RATE = 0.05
    UDP_TX_POWER_DBM = 14.5
    UDP_SENSITIVITY_DBM = -92
    PACKET_SIZE_BYTES = 48
    MAX_RETRIES = 1
    TRANSMIT_SLOTTING = 2
    SAT_BANDWIDTH_BPS = 2400
    SAT_OUTAGE_START = 600
    SAT_OUTAGE_END = 900
    SAT_BURST_PROB = 0.05
    ENVIRONMENTS = {"URBAN": {"n": 3.0, "sigma": 6.8}, "RURAL": {"n": 2.2, "sigma": 3}}
    CURRENT_ENV = "RURAL"

# ===========================================================
# SIMULATION MODELS
# ===========================================================

class NetworkModel:
    @staticmethod
    def calc_link_budget(dist_m, num_active, tx_power_dbm, current_shadowing):
        env = Config.ENVIRONMENTS[Config.CURRENT_ENV]
        n = env["n"]
        pl_d0 = 40.0
        path_loss = pl_d0 + 10 * n * np.log10(max(dist_m, 1.0)) + current_shadowing
        interference = np.random.normal(0, 3.0)
        rssi = tx_power_dbm - path_loss - abs(interference)
        pdr = 1.0 / (1.0 + np.exp(-(rssi - Config.UDP_SENSITIVITY_DBM) / 1.5))
        if np.random.random() < 0.02: pdr *= 0.6
        fec_redundancy = 0.06
        pdr = 1.0 - (1.0 - pdr) ** (1.0 + fec_redundancy)
        collision_prob = 0.007 * (num_active ** 0.8)
        if np.random.random() < collision_prob: pdr = 0.0
        success = 1 if (np.random.random() < pdr) else 0
        return rssi, success

    @staticmethod
    def attempt_transmission(dist_m, num_active, battery_pct, current_shadowing):
        current_tx_power = Config.UDP_TX_POWER_DBM
        if battery_pct < 15.0: current_tx_power -= 6.0
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
        if self.buffer_bytes > 500000: incoming_bytes = int(incoming_bytes * 0.8)
        self.buffer_bytes += incoming_bytes
        sat_connected = not (Config.SAT_OUTAGE_START <= t <= Config.SAT_OUTAGE_END)
        bytes_sent = 0
        if sat_connected:
            bw_jitter = Config.SAT_BANDWIDTH_BPS * np.random.uniform(0.8, 1.2)
            max_bytes_per_sec = bw_jitter / 8
            bytes_sent = min(self.buffer_bytes, max_bytes_per_sec)
            self.buffer_bytes -= bytes_sent
        else: self.buffer_bytes = min(self.buffer_bytes, 500000)
        return {"sat_status": 1 if sat_connected else 0, "buffer_current": self.buffer_bytes}

class HealthModel:
    def __init__(self):
        self.hr = 70.0; self.spo2 = 98.0; self.temp = 37.0; self.stress = 10.0; self.target_hr = 70.0
    def update(self, speed, scenario, fatigue):
        base_hr = 70 + (speed * 15) + (fatigue * 8)
        if scenario == "PANIC":
            self.target_hr = 115 + np.random.normal(0, 10) + (0.15 * self.stress)
            self.stress = min(30, self.stress + 0.3)
            if self.stress > 25: self.spo2 -= 0.02
        elif scenario == "INJURED": 
            self.target_hr = 55 + np.random.normal(0, 3)
            self.spo2 -= 0.08
        elif scenario == "RUNNING":
            self.target_hr = base_hr + 25
            self.stress = max(10, self.stress - 0.05)
        elif scenario == "NORMAL": 
            self.target_hr = base_hr
            self.stress = max(10, self.stress - 0.1)
        noise_hr = np.random.normal(0, 3.0)
        self.hr = (self.hr * 0.88) + (self.target_hr * 0.12) + noise_hr
        if random.random() < 0.008: self.hr += np.random.uniform(8, 15)
        noise_spo2 = np.random.normal(0, 0.5)
        if self.hr > Config.HR_CRITICAL: self.spo2 -= np.random.uniform(0.02, 0.08)
        elif scenario != "INJURED": self.spo2 += (98 - self.spo2) * 0.04
        self.spo2 += noise_spo2
        self.hr = np.clip(self.hr, 40, 200); self.spo2 = np.clip(self.spo2, 75, 100)
        return {"hr": self.hr, "spo2": self.spo2, "temp": self.temp}

class BatteryModel:
    def __init__(self): self.mah = Config.WEARABLE_BATTERY_MAH
    def drain(self, is_tx):
        self.mah = max(0, self.mah - (0.05 if is_tx else 0.01))
        return (self.mah / Config.WEARABLE_BATTERY_MAH) * 100

class Soldier:
    def __init__(self, uid, troop_id, base_node, scenario_schedule):
        self.id = uid; self.troop_id = troop_id
        self.rel_x = np.random.uniform(-100, 100); self.rel_y = np.random.uniform(-100, 100)
        self.vel_x = 0.0; self.vel_y = 0.0; self.hlth = HealthModel(); self.batt = BatteryModel()
        self.timeline = ["NORMAL"] * Config.TOTAL_STEPS; self.shadowing = 0.0
        for start_t, s in sorted(scenario_schedule.items()):
            if s == "PANIC": dur = random.randint(45, 180)
            elif s == "RUNNING": dur = random.randint(60, 240)
            elif s == "FALL_EVENT": dur = random.randint(30, 120)
            elif s == "INJURED": dur = random.randint(120, 600)
            else: dur = random.randint(60, 300)
            end_t = min(start_t + dur, Config.TOTAL_STEPS)
            for i in range(start_t, end_t): self.timeline[i] = s
    def step(self, t, troop_center):
        scen = self.timeline[t]; fatigue_factor = t / Config.TOTAL_STEPS
        target_speed = 4.5 if scen == "RUNNING" else (0.0 if scen == "FALL_EVENT" else 1.2)
        target_speed *= max(0.7, 1.0 - (fatigue_factor * 0.3))
        if scen == "RUNNING": self.vel_x += np.random.normal(0, 0.4); self.vel_y += np.random.normal(0, 0.4)
        elif t % 60 == 0 and scen != "FALL_EVENT": self.vel_x += np.random.normal(0, 0.6); self.vel_y += np.random.normal(0, 0.6)
        cx, cy = troop_center
        self.vel_x += 0.015 * (cx - self.rel_x); self.vel_y += 0.015 * (cy - self.rel_y)
        curr_speed = np.sqrt(self.vel_x**2 + self.vel_y**2)
        if curr_speed > 0: self.vel_x = (self.vel_x/curr_speed)*target_speed; self.vel_y = (self.vel_y/curr_speed)*target_speed
        self.rel_x += self.vel_x; self.rel_y += self.vel_y; dist = np.sqrt(self.rel_x**2 + self.rel_y**2)
        vit = self.hlth.update(curr_speed, scen, fatigue_factor)
        should_transmit = ((t + self.id) % Config.TRANSMIT_SLOTTING == 0)
        batt_pct = self.batt.drain(should_transmit)
        fall_risk = 0.0003 + (0.0008 * fatigue_factor)
        if dist > 150 and scen == "RUNNING" and random.random() < fall_risk: scen = "FALL_EVENT"
        acc = [0.0, 0.0, 9.8]; acc = [a + np.random.normal(0, 0.4) for a in acc]
        if scen == "FALL_EVENT": acc = [np.random.uniform(-25, 25) for _ in range(2)] + [np.random.uniform(0, 15)]
        elif scen == "INJURED": acc = [9.8, 0.0, 0.3]; acc = [a + np.random.normal(0, 0.1) for a in acc]
        elif curr_speed > 3.0:
            acc[0] += np.random.normal(0, 4.5); acc[1] += np.random.normal(0, 4.5)
            if random.random() < 0.015: acc[2] += np.random.uniform(12, 30)
            else: acc[2] += np.random.normal(0, 5.0)
        env_sigma = Config.ENVIRONMENTS[Config.CURRENT_ENV]["sigma"]
        self.shadowing = (0.9 * self.shadowing) + (0.1 * np.random.normal(0, env_sigma))
        if should_transmit: rssi, udp_loss, latency = NetworkModel.attempt_transmission(dist, Config.SOLDIERS_PER_TROOP, batt_pct, self.shadowing)
        else: rssi, udp_loss, latency = -120, -1, np.nan
        return {"id": self.id, "troop_id": self.troop_id, "time": t, "scenario": scen, "rel_x": self.rel_x, "rel_y": self.rel_y, 
                "dist_to_base": dist, "speed": curr_speed, "hr": vit["hr"], "spo2": vit["spo2"], "temp": vit["temp"],
                "acc_x": acc[0], "acc_y": acc[1], "acc_z": acc[2], "udp_rssi": rssi, "udp_loss": udp_loss, 
                "udp_latency": latency, "batt": batt_pct, "label_fall": 1 if scen == "FALL_EVENT" else 0,
                "label_emerg": 1 if (vit["hr"] > Config.HR_CRITICAL or vit["spo2"] < Config.SPO2_CRITICAL) else 0}

class HybridAIHub:
    def __init__(self, df, seed=42):
        self.df = df.sort_values(by=['id', 'time']).copy() if not df.empty else df
        self.models = {}; self.seed = seed; self.trained = False
    def feature_eng(self):
        if self.df.empty: return self.df
        self.df["acc_mag"] = np.sqrt(self.df.acc_x**2 + self.df.acc_y**2 + self.df.acc_z**2)
        g = self.df.groupby("id")
        self.df["acc_mean"] = g["acc_mag"].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
        self.df["acc_var"] = g["acc_mag"].rolling(3, min_periods=1).var().fillna(0).reset_index(0, drop=True)
        self.df["acc_peak"] = g["acc_mag"].rolling(3).max().fillna(0).reset_index(0, drop=True)
        self.df["jerk"] = g["acc_mag"].diff().abs().rolling(5).mean().fillna(0).reset_index(0, drop=True)
        self.df["lying_flag"] = (self.df.acc_z.abs() < 5.0).astype(int)
        self.df["hr_delta"] = g["hr"].diff().fillna(0)
        self.df["temp_norm"] = (self.df["temp"] - 37.0) / 2.0
        self.df["hrv"] = g["hr"].diff().rolling(5).std().fillna(0).reset_index(0, drop=True)
        self.df["stress_hr_ratio"] = self.df.hr / (self.df.hrv + 1e-4)
        return self.df
    def create_windows(self):
        if self.df.empty: return [], [], [], [], []
        seq, y_f, y_h, grp = [], [], [], []
        cols = ["acc_mean", "acc_var", "acc_peak", "jerk", "lying_flag", "hr", "hr_delta", "hrv", "stress_hr_ratio", "temp_norm"]
        self.df[cols] = SimpleImputer(strategy='median').fit_transform(self.df[cols])
        for sid, g in self.df.groupby("id"):
            vals = g[cols].values; f_tgt, h_tgt = g["label_fall"].values, g["label_emerg"].values
            if len(g) < Config.LSTM_WINDOW + 1: continue
            for i in range(0, len(g) - Config.LSTM_WINDOW, 2):
                win = vals[i:i+Config.LSTM_WINDOW]
                feats = np.concatenate([np.mean(win, 0), np.std(win, 0), np.max(win, 0) - np.min(win, 0)])
                seq.append(feats); y_f.append(1 if sum(f_tgt[i:i+Config.LSTM_WINDOW]) >= 1 else 0)
                y_h.append(1 if sum(h_tgt[i:i+Config.LSTM_WINDOW]) >= 2 else 0); grp.append(sid)
        return np.array(seq), np.array(y_f), np.array(y_h), np.array(grp), cols
    def train(self):
        X, y_f, y_h, grp, feat_names_base = self.create_windows()
        feat_names = []; [feat_names.append(f"{f}_{stat}") for stat in ['mean', 'std', 'range'] for f in feat_names_base]
        if len(X) == 0: return {}
        res_f = self._train_model(X, y_f, grp, "Fall_Detection", feat_names)
        res_h = self._train_model(X, y_h, grp, "Health_Alert", feat_names)
        if res_f or res_h: self.trained = True
        return {**res_f, **res_h}
    def _train_model(self, X, y, grp, name, feat_names):
        if len(X) == 0 or sum(y == 1) < 5: return {}
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=self.seed)
        try: train_idx, test_idx = next(gss.split(X, y, grp))
        except: return {}
        X_tr, X_te = X[train_idx], X[test_idx]; y_tr, y_te = y[train_idx], y[test_idx]
        if sum(y_te == 1) == 0 or sum(y_te == 0) == 0: return {}
        pos_count = sum(y_tr == 1); neg_count = sum(y_tr == 0)
        if pos_count < 2 or neg_count < 2: return {}
        if pos_count < neg_count:
            target_pos = min(neg_count, pos_count * 4)
            X_pos_up, y_pos_up = resample(X_tr[y_tr==1], y_tr[y_tr==1], replace=True, n_samples=target_pos, random_state=self.seed)
            X_bal, y_bal = np.vstack([X_tr[y_tr==0], X_pos_up]), np.concatenate([y_tr[y_tr==0], y_pos_up])
        else: X_bal, y_bal = X_tr, y_tr
        clf = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, 
                                scale_pos_weight=neg_count/pos_count if pos_count > 0 else 1,
                                eval_metric='logloss', random_state=self.seed, use_label_encoder=False)
        try: clf.fit(X_bal, y_bal); y_probs = clf.predict_proba(X_te)[:, 1]
        except: return {}
        precision, recall, thresholds = precision_recall_curve(y_te, y_probs)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-6)
        best_thresh = thresholds[np.argmax(f1_scores)] if len(thresholds) > 0 else 0.5
        y_pred = (y_probs > best_thresh).astype(int)
        self.models[name] = {"model": clf, "X_test": X_te, "y_test": y_te, "y_probs": y_probs, 
                            "y_pred": y_pred, "best_thresh": best_thresh, "feature_names": feat_names}
        try: auc_score = roc_auc_score(y_te, y_probs) if len(np.unique(y_te)) > 1 else 0.5
        except: auc_score = 0.5
        return {name: {'auc': auc_score, 'f1': f1_score(y_te, y_pred, zero_division=0)}}

def classify_health_status(hr, spo2, scenario):
    if scenario == "FALL_EVENT": return "FALL DETECTED", "critical", "#DC2626"
    elif scenario == "INJURED": return "INJURED", "critical", "#DC2626"
    elif hr > Config.HR_CRITICAL or spo2 < Config.SPO2_CRITICAL or hr < 50: return "EMERGENCY", "critical", "#DC2626"
    elif scenario == "PANIC" or hr > 100: return "HIGH STRESS", "warning", "#F59E0B"
    elif hr > 90 or spo2 < 92: return "ELEVATED", "caution", "#EAB308"
    elif hr < 60: return "LOW HR", "warning", "#F59E0B"
    else: return "NOMINAL", "normal", "#10B981"

def get_network_health(loss_rate, avg_rssi, buffer_pct):
    if loss_rate > 0.3 or buffer_pct > 80: return "CRITICAL", "#DC2626"
    elif loss_rate > 0.15 or buffer_pct > 60 or avg_rssi < -85: return "DEGRADED", "#F59E0B"
    elif loss_rate > 0.05 or buffer_pct > 40: return "MODERATE", "#EAB308"
    else: return "OPTIMAL", "#10B981"

def simulation_generator(seed_val):
    random.seed(seed_val); np.random.seed(seed_val)
    base_nodes = {i: BaseNode(i) for i in range(1, Config.NUM_TROOPS + 1)}
    soldiers = []; sid_c = 1; troops = {i: [] for i in range(1, Config.NUM_TROOPS + 1)}
    for tid in range(1, Config.NUM_TROOPS + 1):
        for _ in range(Config.SOLDIERS_PER_TROOP):
            events = {0: "NORMAL"}
            if random.random() < 0.25: events[random.randint(200, Config.TOTAL_STEPS - 500)] = "PANIC"
            if random.random() < 0.08: events[random.randint(400, Config.TOTAL_STEPS - 300)] = "FALL_EVENT"
            if random.random() < 0.45: events[random.randint(150, Config.TOTAL_STEPS - 500)] = "RUNNING"
            if random.random() < 0.05: events[random.randint(500, Config.TOTAL_STEPS - 200)] = "INJURED"
            s = Soldier(sid_c, tid, base_nodes[tid], events); soldiers.append(s); troops[tid].append(s); sid_c += 1
    all_data = []
    for t in range(Config.TOTAL_STEPS):
        bn_load = {i: 0 for i in range(1, Config.NUM_TROOPS + 1)}
        t_centers = {tid: (np.mean([x.rel_x for x in m]), np.mean([x.rel_y for x in m])) for tid, m in troops.items()}
        current_step_data = []
        for s in soldiers:
            pkt = s.step(t, t_centers[s.troop_id])
            if pkt["udp_loss"] == 0: bn_load[s.troop_id] += Config.PACKET_SIZE_BYTES
            current_step_data.append(pkt)
        for tid, bn in base_nodes.items():
            sat = bn.process_uplink(t, bn_load[tid])
            for pkt in current_step_data:
                if pkt["troop_id"] == tid: pkt["sat_status"] = sat["sat_status"]; pkt["base_buffer_bytes"] = sat["buffer_current"]
        all_data.extend(current_step_data)
        yield {'step': t, 'seed': seed_val, 'current_data': current_step_data, 'all_data': all_data.copy(),
               'base_nodes': {tid: {'buffer': bn.buffer_bytes} for tid, bn in base_nodes.items()}, 'soldiers': soldiers}

# ===========================================================
# STREAMLIT APP
# ===========================================================

st.set_page_config(page_title="WatchGuard Command Center", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Dark Professional Theme */
    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    }
    
    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #334155 100%);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        margin: 10px 0;
    }
    
    /* Alerts */
    .alert-critical {
        background: linear-gradient(135deg, #DC2626 0%, #991B1B 100%);
        color: white;
        padding: 16px 20px;
        border-radius: 12px;
        margin: 8px 0;
        font-weight: 600;
        font-size: 14px;
        border-left: 4px solid #FEE2E2;
        box-shadow: 0 4px 16px rgba(220, 38, 38, 0.4);
        animation: pulse 2s ease-in-out infinite;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
        color: white;
        padding: 14px 20px;
        border-radius: 12px;
        margin: 6px 0;
        font-weight: 600;
        font-size: 13px;
        border-left: 4px solid #FEF3C7;
        box-shadow: 0 4px 16px rgba(245, 158, 11, 0.3);
    }
    
    .alert-info {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
        color: white;
        padding: 12px 20px;
        border-radius: 12px;
        margin: 4px 0;
        font-weight: 500;
        font-size: 13px;
        border-left: 4px solid #DBEAFE;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.3);
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.85; }
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-critical { background: #DC2626; color: white; }
    .status-warning { background: #F59E0B; color: white; }
    .status-normal { background: #10B981; color: white; }
    .status-info { background: #3B82F6; color: white; }
    
    /* Headers */
    h1 {
        color: #F1F5F9 !important;
        font-weight: 700 !important;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }
    
    h2 {
        color: #E2E8F0 !important;
        font-weight: 600 !important;
        font-size: 1.5rem !important;
        margin-top: 1.5rem !important;
    }
    
    h3 {
        color: #CBD5E1 !important;
        font-weight: 600 !important;
        font-size: 1.2rem !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #06B6D4 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(30, 41, 59, 0.5);
        padding: 8px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #94A3B8;
        font-weight: 600;
        padding: 12px 24px;
        border-radius: 8px;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #06B6D4 0%, #0891B2 100%);
        color: white;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #06B6D4 0%, #0891B2 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        box-shadow: 0 4px 16px rgba(6, 182, 212, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #0891B2 0%, #0E7490 100%);
        box-shadow: 0 6px 20px rgba(6, 182, 212, 0.4);
        transform: translateY(-2px);
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #06B6D4 0%, #10B981 100%);
        border-radius: 10px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1E293B 0%, #0F172A 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.1);
    }
    
    /* Info boxes */
    .stInfo {
        background: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3B82F6;
        border-radius: 8px;
    }
    
    /* Success boxes */
    .stSuccess {
        background: rgba(16, 185, 129, 0.1);
        border-left: 4px solid #10B981;
        border-radius: 8px;
    }
    
    /* Warning boxes */
    .stWarning {
        background: rgba(245, 158, 11, 0.1);
        border-left: 4px solid #F59E0B;
        border-radius: 8px;
    }
    
    /* Error boxes */
    .stError {
        background: rgba(220, 38, 38, 0.1);
        border-left: 4px solid #DC2626;
        border-radius: 8px;
    }
    
    /* DataFrames */
    .dataframe {
        background: rgba(30, 41, 59, 0.5) !important;
        border-radius: 8px !important;
    }
    
    /* Plotly Charts */
    .js-plotly-plot {
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# SESSION STATE
if 'running' not in st.session_state: st.session_state.running = False
if 'sim' not in st.session_state: st.session_state.sim = None
if 'state' not in st.session_state: st.session_state.state = None
if 'speed' not in st.session_state: st.session_state.speed = 2.0
if 'current_seed' not in st.session_state: st.session_state.current_seed = Config.SEEDS[0]
if 'logs' not in st.session_state: st.session_state.logs = deque(maxlen=200)
if 'alerts' not in st.session_state: st.session_state.alerts = deque(maxlen=50)
if 'ai_hub' not in st.session_state: st.session_state.ai_hub = None
if 'ai_metrics' not in st.session_state: st.session_state.ai_metrics = {}
if 'train_trigger' not in st.session_state: st.session_state.train_trigger = 600
if 'mode' not in st.session_state: st.session_state.mode = "realtime"
if 'selected_soldier' not in st.session_state: st.session_state.selected_soldier = 1
if 'replay_data' not in st.session_state: st.session_state.replay_data = None
if 'replay_time' not in st.session_state: st.session_state.replay_time = 0
if 'replay_playing' not in st.session_state: st.session_state.replay_playing = False
if 'replay_speed' not in st.session_state: st.session_state.replay_speed = 1.0

# SIDEBAR
with st.sidebar:
    st.markdown("# ⚡ COMMAND CENTER")
    st.markdown("---")
    
    mode = st.radio("", ["🔴 LIVE OPERATION", "📹 REPLAY ANALYSIS"], label_visibility="collapsed")
    st.session_state.mode = "realtime" if mode == "🔴 LIVE OPERATION" else "replay"
    
    st.markdown("---")
    
    if st.session_state.mode == "realtime":
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶ START", use_container_width=True, disabled=st.session_state.running, type="primary"):
                if st.session_state.sim is None:
                    st.session_state.sim = simulation_generator(st.session_state.current_seed)
                    st.session_state.ai_hub = None; st.session_state.ai_metrics = {}
                st.session_state.running = True
                st.session_state.logs.append(f"✓ Mission initiated")
        with col2:
            if st.button("⏸ PAUSE", use_container_width=True):
                st.session_state.running = False
        
        if st.button("🔄 RESET", use_container_width=True):
            st.session_state.sim = None; st.session_state.state = None; st.session_state.running = False
            st.session_state.ai_hub = None; st.session_state.ai_metrics = {}
            st.session_state.logs.clear(); st.session_state.alerts.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### ⚙️ CONFIGURATION")
        st.session_state.speed = st.slider("Speed", 0.5, 5.0, st.session_state.speed, 0.5)
        st.session_state.current_seed = st.selectbox("Scenario", Config.SEEDS, format_func=lambda x: f"Scenario {Config.SEEDS.index(x)+1}")
        Config.CURRENT_ENV = st.selectbox("Terrain", ["RURAL", "URBAN"])
        st.session_state.train_trigger = st.slider("AI Training Step", 300, 1500, st.session_state.train_trigger, 100)
    else:
        # REPLAY MODE CONTROLS
        st.markdown("### 📹 REPLAY CONTROLS")
        
        if st.session_state.state and st.session_state.state.get('all_data'):
            if st.button("💾 SAVE FOR REPLAY", use_container_width=True, type="primary"):
                st.session_state.replay_data = pd.DataFrame(st.session_state.state['all_data'])
                st.success("✓ Mission data saved for replay!")
        
        if st.session_state.replay_data is not None:
            max_time = int(st.session_state.replay_data['time'].max())
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("▶ PLAY" if not st.session_state.replay_playing else "⏸ PAUSE", use_container_width=True):
                    st.session_state.replay_playing = not st.session_state.replay_playing
            with col2:
                if st.button("⏮ RESTART", use_container_width=True):
                    st.session_state.replay_time = 0
                    st.session_state.replay_playing = False
            
            st.session_state.replay_speed = st.slider("Replay Speed", 0.5, 5.0, st.session_state.replay_speed, 0.5)
            st.session_state.replay_time = st.slider("Timeline", 0, max_time, st.session_state.replay_time, 1)
            
            st.markdown("### 📊 MISSION SUMMARY")
            total_falls = st.session_state.replay_data['label_fall'].sum()
            total_emergencies = st.session_state.replay_data['label_emerg'].sum()
            st.markdown(f"""
            <div style='background: rgba(30, 41, 59, 0.5); padding: 16px; border-radius: 8px; font-size: 13px;'>
            <strong>Total Falls:</strong> {total_falls}<br>
            <strong>Total Emergencies:</strong> {total_emergencies}<br>
            <strong>Duration:</strong> {max_time}s<br>
            <strong>Personnel:</strong> {len(st.session_state.replay_data['id'].unique())}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("ℹ️ No replay data available. Run a live mission first and save it for replay.")
    
    st.markdown("---")
    st.markdown("### 📊 MISSION PARAMETERS")
    st.markdown(f"""
    <div style='background: rgba(30, 41, 59, 0.5); padding: 16px; border-radius: 8px; font-size: 13px;'>
    <strong>Environment:</strong> {Config.CURRENT_ENV}<br>
    <strong>Duration:</strong> {Config.DURATION_MIN} minutes<br>
    <strong>Units:</strong> {Config.NUM_TROOPS} troops<br>
    <strong>Personnel:</strong> {Config.NUM_TROOPS * Config.SOLDIERS_PER_TROOP} soldiers<br>
    <strong>HR Critical:</strong> > {Config.HR_CRITICAL} BPM<br>
    <strong>HR Normal:</strong> 60-90 BPM
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    if st.session_state.mode == "realtime" and st.session_state.state and len(st.session_state.state['all_data']) > 0:
        df_export = pd.DataFrame(st.session_state.state['all_data'])
        csv = df_export.to_csv(index=False)
        st.download_button("💾 EXPORT DATA", csv, "mission_data.csv", "text/csv", use_container_width=True)
    elif st.session_state.mode == "replay" and st.session_state.replay_data is not None:
        csv = st.session_state.replay_data.to_csv(index=False)
        st.download_button("💾 EXPORT REPLAY DATA", csv, "replay_data.csv", "text/csv", use_container_width=True)

# HEADER
st.markdown("""
<div style='text-align: center; padding: 20px 0;'>
    <h1 style='margin: 0; background: linear-gradient(90deg, #06B6D4 0%, #10B981 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
        🛡️ WATCHGUARD COMMAND CENTER
    </h1>
    <p style='color: #94A3B8; font-size: 1.1rem; margin-top: 0.5rem; font-weight: 500;'>
        AI-Powered Battlefield Intelligence & Real-Time Health Monitoring System
    </p>
</div>
""", unsafe_allow_html=True)

# PROGRESS BAR
if st.session_state.mode == "realtime" and st.session_state.state:
    progress = st.session_state.state['step'] / Config.TOTAL_STEPS
    st.progress(progress, text=f"⏱ MISSION TIME: {st.session_state.state['step']}s / {Config.TOTAL_STEPS}s  •  {progress*100:.1f}% COMPLETE")
elif st.session_state.mode == "replay" and st.session_state.replay_data is not None:
    max_time = int(st.session_state.replay_data['time'].max())
    progress = st.session_state.replay_time / max_time if max_time > 0 else 0
    st.progress(progress, text=f"📹 REPLAY TIME: {st.session_state.replay_time}s / {max_time}s  •  {progress*100:.1f}% COMPLETE")

st.markdown("---")

# Determine which data to display
display_data = None
current_step = 0

if st.session_state.mode == "realtime" and st.session_state.state and len(st.session_state.state['all_data']) > 0:
    display_data = st.session_state.state['all_data']
    current_step = st.session_state.state['step']
elif st.session_state.mode == "replay" and st.session_state.replay_data is not None:
    display_data = st.session_state.replay_data.to_dict('records')
    current_step = st.session_state.replay_time

# MAIN DASHBOARD
if display_data and len(display_data) > 0:
    df = pd.DataFrame(display_data)
    current_positions = df[df['time'] == current_step]
    
    # METRICS
    tx_attempts = df[df['udp_loss'] != -1]
    total_packets = len(tx_attempts)
    udp_success = len(tx_attempts[tx_attempts['udp_loss'] == 0])
    pdr = (udp_success / total_packets * 100) if total_packets > 0 else 0
    falls = df[df['label_fall'] == 1]['id'].nunique()
    health_alerts = df[df['label_emerg'] == 1]['id'].nunique()
    successful = tx_attempts[tx_attempts['udp_loss'] == 0]
    avg_latency = successful['udp_latency'].mean() if not successful.empty else 0
    avg_battery = current_positions['batt'].mean() if not current_positions.empty else 0
    loss_rate = tx_attempts['udp_loss'].mean() if not tx_attempts.empty else 0
    avg_rssi = tx_attempts['udp_rssi'].mean() if not tx_attempts.empty else -120
    
    # Handle base node buffer for replay mode
    if st.session_state.mode == "realtime" and st.session_state.state:
        max_buffer = max([st.session_state.state['base_nodes'][tid]['buffer'] for tid in range(1, Config.NUM_TROOPS + 1)])
    else:
        max_buffer = df['base_buffer_bytes'].max() if 'base_buffer_bytes' in df.columns else 0
    
    buffer_pct = (max_buffer / 500000) * 100
    net_status, net_color = get_network_health(loss_rate, avg_rssi, buffer_pct)
    sat_active = not (Config.SAT_OUTAGE_START <= current_step <= Config.SAT_OUTAGE_END)
    
    # KPI DASHBOARD
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='color: #94A3B8; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;'>NETWORK STATUS</div>
            <div style='font-size: 24px; font-weight: 700; color: {net_color}; margin-bottom: 4px;'>{net_status}</div>
            <div style='color: #CBD5E1; font-size: 13px;'>PDR: {pdr:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        color = "#DC2626" if falls > 0 else "#10B981"
        st.markdown(f"""
        <div class='metric-card'>
            <div style='color: #94A3B8; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;'>FALL EVENTS</div>
            <div style='font-size: 32px; font-weight: 700; color: {color}; margin-bottom: 4px;'>{falls}</div>
            <div style='color: #CBD5E1; font-size: 13px;'>Personnel</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        color = "#DC2626" if health_alerts > 0 else "#10B981"
        st.markdown(f"""
        <div class='metric-card'>
            <div style='color: #94A3B8; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;'>HEALTH ALERTS</div>
            <div style='font-size: 32px; font-weight: 700; color: {color}; margin-bottom: 4px;'>{health_alerts}</div>
            <div style='color: #CBD5E1; font-size: 13px;'>Critical Vitals</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        color = "#10B981" if avg_latency < 50 else "#F59E0B" if avg_latency < 100 else "#DC2626"
        st.markdown(f"""
        <div class='metric-card'>
            <div style='color: #94A3B8; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;'>AVG LATENCY</div>
            <div style='font-size: 28px; font-weight: 700; color: {color}; margin-bottom: 4px;'>{avg_latency:.0f}<span style='font-size: 16px;'>ms</span></div>
            <div style='color: #CBD5E1; font-size: 13px;'>Response Time</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        color = "#10B981" if avg_battery > 50 else "#F59E0B" if avg_battery > 20 else "#DC2626"
        st.markdown(f"""
        <div class='metric-card'>
            <div style='color: #94A3B8; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;'>AVG BATTERY</div>
            <div style='font-size: 28px; font-weight: 700; color: {color}; margin-bottom: 4px;'>{avg_battery:.0f}<span style='font-size: 16px;'>%</span></div>
            <div style='color: #CBD5E1; font-size: 13px;'>Fleet Power</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        sat_color = "#10B981" if sat_active else "#DC2626"
        sat_text = "ONLINE" if sat_active else "OFFLINE"
        st.markdown(f"""
        <div class='metric-card'>
            <div style='color: #94A3B8; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;'>SATCOM</div>
            <div style='font-size: 24px; font-weight: 700; color: {sat_color}; margin-bottom: 4px;'>{sat_text}</div>
            <div style='color: #CBD5E1; font-size: 13px;'>Primary Link</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # TABS
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🗺 TACTICAL MAP", "💊 HEALTH MATRIX", "📡 NETWORK INTEL", "👤 PERSONNEL", "🤖 AI ANALYTICS"])
    
    with tab1:
        col1, col2 = st.columns([3, 1])
        with col1:
            fig_map = go.Figure()
            fig_map.add_trace(go.Scatter(x=[0], y=[0], mode='markers', 
                                        marker=dict(size=35, color='#06B6D4', symbol='triangle-up', 
                                                   line=dict(width=3, color='white')),
                                        name='Base Station', showlegend=True))
            
            for tid in range(1, Config.NUM_TROOPS + 1):
                troop_data = current_positions[current_positions['troop_id'] == tid]
                colors = ['#DC2626' if s == 'FALL_EVENT' else '#F59E0B' if s == 'PANIC' else 
                         '#EAB308' if s == 'RUNNING' else '#10B981' for s in troop_data['scenario']]
                sizes = [25 if s in ['FALL_EVENT', 'PANIC'] else 15 for s in troop_data['scenario']]
                
                fig_map.add_trace(go.Scatter(x=troop_data['rel_x'], y=troop_data['rel_y'], mode='markers',
                                            marker=dict(size=sizes, color=colors, 
                                                       line=dict(width=2, color='white'),
                                                       opacity=0.9),
                                            name=f'Unit {tid}',
                                            text=[f"<b>S-{row['id']:02d}</b><br>Status: {row['scenario']}<br>HR: {row['hr']:.0f} bpm<br>SpO2: {row['spo2']:.0f}%" 
                                                 for _, row in troop_data.iterrows()],
                                            hovertemplate='%{text}<extra></extra>'))
            
            mode_text = "REPLAY MODE" if st.session_state.mode == "replay" else "LIVE OPERATION"
            fig_map.update_layout(
                title=dict(text=f"REAL-TIME BATTLEFIELD POSITIONING ({mode_text})", font=dict(size=18, color='#E2E8F0', family='Arial Black')),
                xaxis_title="X Coordinate (meters)", yaxis_title="Y Coordinate (meters)",
                plot_bgcolor='#0F172A', paper_bgcolor='#1E293B',
                font=dict(color='#CBD5E1'), height=550,
                xaxis=dict(gridcolor='#334155', zerolinecolor='#475569'),
                yaxis=dict(gridcolor='#334155', zerolinecolor='#475569'),
                legend=dict(bgcolor='rgba(30, 41, 59, 0.8)', bordercolor='#475569', borderwidth=1)
            )
            st.plotly_chart(fig_map, use_container_width=True, key="tac_map")
        
        with col2:
            st.markdown("### 🎯 SITREP")
            critical = current_positions[current_positions['scenario'].isin(['FALL_EVENT', 'PANIC', 'INJURED'])]
            if not critical.empty:
                st.error(f"⚠️ {len(critical)} REQUIRE ATTENTION")
                for _, row in critical.head(8).iterrows():
                    status, level, color = classify_health_status(row['hr'], row['spo2'], row['scenario'])
                    st.markdown(f"<span style='color: {color}; font-weight: 600;'>S-{row['id']:02d}</span> (U{row['troop_id']}): {status}", unsafe_allow_html=True)
            else:
                st.success("✓ ALL CLEAR")
            
            st.markdown("### 📊 DISTRIBUTION")
            scenario_counts = current_positions['scenario'].value_counts()
            for scen, count in scenario_counts.items():
                st.text(f"{scen}: {count} personnel")
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig_hr = go.Figure()
            fig_hr.add_trace(go.Histogram(x=current_positions['hr'], nbinsx=25, 
                                         marker_color='#EF4444', opacity=0.8))
            fig_hr.add_vline(x=Config.HR_CRITICAL, line_dash="dash", line_color="#DC2626", 
                            line_width=3, annotation_text="CRITICAL (>120)", annotation_position="top")
            fig_hr.add_vline(x=60, line_dash="dash", line_color="#F59E0B", 
                            line_width=2, annotation_text="LOW (<60)", annotation_position="bottom left")
            fig_hr.add_vline(x=50, line_dash="dash", line_color="#DC2626", 
                            line_width=3, annotation_text="EMERGENCY (<50)", annotation_position="bottom left")
            fig_hr.update_layout(
                title="HEART RATE DISTRIBUTION", plot_bgcolor='#0F172A', paper_bgcolor='#1E293B',
                font=dict(color='#CBD5E1'), height=320, xaxis_title="BPM", yaxis_title="Count",
                xaxis=dict(gridcolor='#334155'), yaxis=dict(gridcolor='#334155')
            )
            st.plotly_chart(fig_hr, use_container_width=True, key="hr_chart")
        
        with col2:
            fig_spo2 = go.Figure()
            fig_spo2.add_trace(go.Histogram(x=current_positions['spo2'], nbinsx=25,
                                           marker_color='#3B82F6', opacity=0.8))
            fig_spo2.add_vline(x=Config.SPO2_CRITICAL, line_dash="dash", line_color="#DC2626",
                              line_width=3, annotation_text="CRITICAL", annotation_position="bottom")
            fig_spo2.update_layout(
                title="OXYGEN SATURATION DISTRIBUTION", plot_bgcolor='#0F172A', paper_bgcolor='#1E293B',
                font=dict(color='#CBD5E1'), height=320, xaxis_title="SpO2 %", yaxis_title="Count",
                xaxis=dict(gridcolor='#334155'), yaxis=dict(gridcolor='#334155')
            )
            st.plotly_chart(fig_spo2, use_container_width=True, key="spo2_chart")
        
        st.markdown("### 🏥 PERSONNEL HEALTH MATRIX")
        health_summary = []
        for _, row in current_positions.iterrows():
            status, level, color = classify_health_status(row['hr'], row['spo2'], row['scenario'])
            health_summary.append({
                'ID': f"S-{row['id']:02d}", 'Unit': f"U{row['troop_id']}", 'Status': status,
                'HR': f"{row['hr']:.0f}", 'SpO2': f"{row['spo2']:.0f}", 'Battery': f"{row['batt']:.0f}%"
            })
        health_df = pd.DataFrame(health_summary)
        critical_health = health_df[health_df['Status'].str.contains('EMERGENCY|FALL|INJURED|LOW HR')]
        if not critical_health.empty:
            st.error(f"🚨 {len(critical_health)} CRITICAL/WARNING STATUS")
            st.dataframe(critical_health, use_container_width=True, hide_index=True, height=200)
        st.dataframe(health_df, use_container_width=True, hide_index=True, height=300)
    
    with tab3:
        st.markdown(f"### 🌐 NETWORK STATUS: <span style='color: {net_color};'>{net_status}</span>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Packet Delivery", f"{pdr:.1f}%", delta=None)
        with col2: st.metric("Signal Strength", f"{avg_rssi:.1f} dBm", delta=None)
        with col3: st.metric("Loss Rate", f"{loss_rate*100:.1f}%", delta=None)
        with col4: st.metric("Buffer Usage", f"{buffer_pct:.1f}%", delta=None)
        
        col1, col2 = st.columns(2)
        with col1:
            recent_tx = df[df['udp_loss'] != -1].tail(500)
            fig_rssi = go.Figure()
            fig_rssi.add_trace(go.Scatter(x=recent_tx['dist_to_base'], y=recent_tx['udp_rssi'], mode='markers',
                                         marker=dict(size=6, color=recent_tx['udp_loss'], 
                                                    colorscale=[[0, '#10B981'], [1, '#DC2626']], 
                                                    showscale=True, colorbar=dict(title="Loss"))))
            fig_rssi.add_hline(y=Config.UDP_SENSITIVITY_DBM, line_dash="dash", line_color="#DC2626", line_width=2)
            fig_rssi.update_layout(title="LINK QUALITY ANALYSIS", plot_bgcolor='#0F172A', paper_bgcolor='#1E293B',
                                  font=dict(color='#CBD5E1'), height=340, xaxis_title="Distance (m)", yaxis_title="RSSI (dBm)",
                                  xaxis=dict(gridcolor='#334155'), yaxis=dict(gridcolor='#334155'))
            st.plotly_chart(fig_rssi, use_container_width=True, key="rssi_plot")
        
        with col2:
            if len(df) > 100:
                time_data = df.groupby('time').agg({'udp_loss': lambda x: (x == 0).sum() / len(x[x != -1]) if len(x[x != -1]) > 0 else 0}).tail(100)
                fig_pdr = go.Figure()
                fig_pdr.add_trace(go.Scatter(x=time_data.index, y=time_data['udp_loss'], mode='lines',
                                            fill='tozeroy', line=dict(color='#10B981', width=2)))
                fig_pdr.add_vrect(x0=Config.SAT_OUTAGE_START, x1=Config.SAT_OUTAGE_END, 
                                 fillcolor="#DC2626", opacity=0.2, layer="below", line_width=0,
                                 annotation_text="SATCOM OUTAGE", annotation_position="top")
                fig_pdr.update_layout(title="PACKET DELIVERY TIMELINE", plot_bgcolor='#0F172A', paper_bgcolor='#1E293B',
                                     font=dict(color='#CBD5E1'), height=340, xaxis_title="Time (s)", yaxis_title="PDR",
                                     xaxis=dict(gridcolor='#334155'), yaxis=dict(gridcolor='#334155'))
                st.plotly_chart(fig_pdr, use_container_width=True, key="pdr_plot")
    
    with tab4:
        st.markdown("### 👤 INDIVIDUAL PERSONNEL ANALYSIS")
        soldier_ids = sorted(current_positions['id'].unique())
        st.session_state.selected_soldier = st.selectbox("Select Personnel ID", soldier_ids, 
                                                         format_func=lambda x: f"S-{x:02d}",
                                                         index=soldier_ids.index(st.session_state.selected_soldier) if st.session_state.selected_soldier in soldier_ids else 0)
        
        soldier_data = df[df['id'] == st.session_state.selected_soldier]
        if not soldier_data.empty:
            latest = soldier_data.iloc[-1]
            status, level, color = classify_health_status(latest['hr'], latest['spo2'], latest['scenario'])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.markdown(f"<div class='metric-card'><div style='color: #94A3B8; font-size: 11px;'>STATUS</div><div style='font-size: 20px; font-weight: 700; color: {color};'>{status}</div></div>", unsafe_allow_html=True)
            with col2: st.markdown(f"<div class='metric-card'><div style='color: #94A3B8; font-size: 11px;'>HEART RATE</div><div style='font-size: 24px; font-weight: 700; color: #EF4444;'>{latest['hr']:.0f}<span style='font-size: 14px;'> BPM</span></div></div>", unsafe_allow_html=True)
            with col3: st.markdown(f"<div class='metric-card'><div style='color: #94A3B8; font-size: 11px;'>OXYGEN</div><div style='font-size: 24px; font-weight: 700; color: #3B82F6;'>{latest['spo2']:.0f}<span style='font-size: 14px;'> %</span></div></div>", unsafe_allow_html=True)
            with col4: st.markdown(f"<div class='metric-card'><div style='color: #94A3B8; font-size: 11px;'>BATTERY</div><div style='font-size: 24px; font-weight: 700; color: #10B981;'>{latest['batt']:.0f}<span style='font-size: 14px;'> %</span></div></div>", unsafe_allow_html=True)
            
            fig_vitals = make_subplots(rows=2, cols=2, 
                                       subplot_titles=("Heart Rate Timeline", "Blood Oxygen Timeline", 
                                                      "Accelerometer Data", "Battery Level"))
            fig_vitals.add_trace(go.Scatter(x=soldier_data['time'], y=soldier_data['hr'], mode='lines',
                                           line=dict(color='#EF4444', width=2)), row=1, col=1)
            fig_vitals.add_hline(y=Config.HR_CRITICAL, line_dash="dash", line_color="#DC2626", line_width=1, row=1, col=1)
            fig_vitals.add_hline(y=60, line_dash="dash", line_color="#F59E0B", line_width=1, row=1, col=1)
            
            fig_vitals.add_trace(go.Scatter(x=soldier_data['time'], y=soldier_data['spo2'], mode='lines',
                                           line=dict(color='#3B82F6', width=2)), row=1, col=2)
            soldier_data_copy = soldier_data.copy()
            soldier_data_copy['acc_mag'] = np.sqrt(soldier_data_copy['acc_x']**2 + soldier_data_copy['acc_y']**2 + soldier_data_copy['acc_z']**2)
            fig_vitals.add_trace(go.Scatter(x=soldier_data_copy['time'], y=soldier_data_copy['acc_mag'], mode='lines',
                                           line=dict(color='#8B5CF6', width=2)), row=2, col=1)
            fig_vitals.add_trace(go.Scatter(x=soldier_data['time'], y=soldier_data['batt'], mode='lines',
                                           line=dict(color='#10B981', width=2)), row=2, col=2)
            fig_vitals.update_layout(height=500, showlegend=False, plot_bgcolor='#0F172A', paper_bgcolor='#1E293B',
                                    font=dict(color='#CBD5E1'))
            fig_vitals.update_xaxes(gridcolor='#334155')
            fig_vitals.update_yaxes(gridcolor='#334155')
            st.plotly_chart(fig_vitals, use_container_width=True, key="vitals_plot")
    
    with tab5:
        st.markdown("### 🤖 ARTIFICIAL INTELLIGENCE ANALYTICS")
        
        # Check if we have AI models (from realtime mode)
        has_ai_models = st.session_state.ai_hub and st.session_state.ai_hub.trained and st.session_state.ai_hub.models
        
        # In replay mode, offer to train AI on replay data
        if st.session_state.mode == "replay" and not has_ai_models:
            if len(df) >= 500:
                if st.button("🚀 TRAIN AI MODELS ON REPLAY DATA", type="primary"):
                    with st.spinner("Training models on replay data..."):
                        st.session_state.ai_hub = HybridAIHub(df, seed=42)
                        st.session_state.ai_hub.feature_eng()
                        st.session_state.ai_metrics = st.session_state.ai_hub.train()
                    st.rerun()
            else:
                st.info(f"⏳ Need at least 500 data points for AI training. Currently: {len(df)}")
        
        if has_ai_models:
            col1, col2 = st.columns(2)
            with col1:
                st.success("✓ AI MODELS OPERATIONAL")
                for name, metrics in st.session_state.ai_metrics.items():
                    if metrics['auc'] > 0:
                        st.markdown(f"<div class='metric-card'><div style='color: #94A3B8; font-size: 11px;'>{name.upper()}</div><div style='font-size: 24px; font-weight: 700; color: #06B6D4;'>AUC: {metrics['auc']:.3f}</div><div style='color: #CBD5E1; font-size: 13px;'>F1: {metrics['f1']:.3f}</div></div>", unsafe_allow_html=True)
            
            for name, model_data in st.session_state.ai_hub.models.items():
                st.markdown(f"#### {name.replace('_', ' ').upper()}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    fpr, tpr, _ = roc_curve(model_data['y_test'], model_data['y_probs'])
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', line=dict(color='#F59E0B', width=3), name='Model'))
                    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='#94A3B8', width=2), name='Random'))
                    fig_roc.update_layout(title="ROC Curve", plot_bgcolor='#0F172A', paper_bgcolor='#1E293B',
                                         font=dict(color='#CBD5E1'), height=320, xaxis_title="FPR", yaxis_title="TPR",
                                         xaxis=dict(gridcolor='#334155'), yaxis=dict(gridcolor='#334155'))
                    st.plotly_chart(fig_roc, use_container_width=True, key=f"roc_{name}")
                with col2:
                    cm = confusion_matrix(model_data['y_test'], model_data['y_pred'])
                    fig_cm = go.Figure(data=go.Heatmap(z=cm, x=['Pred 0', 'Pred 1'], y=['True 0', 'True 1'],
                                                       colorscale='Blues', text=cm, texttemplate='%{text}', textfont=dict(size=16)))
                    fig_cm.update_layout(title="Confusion Matrix", plot_bgcolor='#0F172A', paper_bgcolor='#1E293B',
                                        font=dict(color='#CBD5E1'), height=320)
                    st.plotly_chart(fig_cm, use_container_width=True, key=f"cm_{name}")
                with col3:
                    imp = model_data['model'].feature_importances_
                    names = np.array(model_data['feature_names'])
                    indices = np.argsort(imp)[::-1][:10]
                    fig_imp = go.Figure()
                    fig_imp.add_trace(go.Bar(x=imp[indices], y=names[indices], orientation='h', 
                                            marker_color='#06B6D4'))
                    fig_imp.update_layout(title="Feature Importance", plot_bgcolor='#0F172A', paper_bgcolor='#1E293B',
                                         font=dict(color='#CBD5E1'), height=320, xaxis_title="Importance",
                                         xaxis=dict(gridcolor='#334155'), yaxis=dict(gridcolor='#334155'))
                    st.plotly_chart(fig_imp, use_container_width=True, key=f"imp_{name}")
        else:
            if st.session_state.mode == "realtime":
                st.info(f"⏳ DATA COLLECTION IN PROGRESS... AI training scheduled at step {st.session_state.train_trigger}")
                if len(df) >= st.session_state.train_trigger:
                    if st.button("🚀 TRAIN AI MODELS", type="primary"):
                        with st.spinner("Training models..."):
                            st.session_state.ai_hub = HybridAIHub(df, seed=st.session_state.current_seed)
                            st.session_state.ai_hub.feature_eng()
                            st.session_state.ai_metrics = st.session_state.ai_hub.train()
                        st.rerun()
else:
    st.info("👈 Initialize mission from command center or load replay data")
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Total Personnel", Config.NUM_TROOPS * Config.SOLDIERS_PER_TROOP)
    with col2: st.metric("Mission Duration", f"{Config.DURATION_MIN} minutes")
    with col3: st.metric("Combat Units", Config.NUM_TROOPS)

# SIMULATION LOOP - REALTIME MODE
if st.session_state.running and st.session_state.sim and st.session_state.mode == "realtime":
    try:
        state = next(st.session_state.sim)
        st.session_state.state = state
        current_step = state['step']
        
        for pkt in state['current_data']:
            if pkt['scenario'] == 'FALL_EVENT' and random.random() < 0.1:
                st.session_state.alerts.append(f"Soldier {pkt['id']} (Unit {pkt['troop_id']}) - FALL DETECTED at T+{current_step}s")
            elif pkt['hr'] > Config.HR_CRITICAL and random.random() < 0.05:
                st.session_state.alerts.append(f"Soldier {pkt['id']} - EMERGENCY: Heart rate {pkt['hr']:.0f} BPM")
            elif pkt['hr'] < 50 and random.random() < 0.05:
                st.session_state.alerts.append(f"Soldier {pkt['id']} - EMERGENCY: Heart rate critically low {pkt['hr']:.0f} BPM")
            elif pkt['hr'] < 60 and random.random() < 0.03:
                st.session_state.alerts.append(f"Soldier {pkt['id']} - LOW HR WARNING: {pkt['hr']:.0f} BPM")
            elif pkt['spo2'] < Config.SPO2_CRITICAL and random.random() < 0.05:
                st.session_state.alerts.append(f"Soldier {pkt['id']} - EMERGENCY: Oxygen {pkt['spo2']:.0f}%")
            elif pkt['batt'] < 15 and random.random() < 0.02:
                st.session_state.alerts.append(f"Soldier {pkt['id']} - BATTERY CRITICAL: {pkt['batt']:.0f}%")
        
        if current_step % 120 == 0:
            st.session_state.logs.append(f"Mission time: {current_step}s")
        if current_step == Config.SAT_OUTAGE_START:
            st.session_state.alerts.append("SATELLITE LINK DOWN - Backup systems active")
        elif current_step == Config.SAT_OUTAGE_END:
            st.session_state.alerts.append("SATELLITE LINK RESTORED - Primary comms online")
        
        if current_step == st.session_state.train_trigger and st.session_state.ai_hub is None:
            df = pd.DataFrame(state['all_data'])
            if len(df) >= 500:
                st.session_state.ai_hub = HybridAIHub(df, seed=st.session_state.current_seed)
                st.session_state.ai_hub.feature_eng()
                st.session_state.ai_metrics = st.session_state.ai_hub.train()
                if st.session_state.ai_metrics:
                    st.session_state.alerts.append("AI SYSTEMS ONLINE - Predictive analytics active")
        
        time.sleep(1.0 / st.session_state.speed)
        st.rerun()
    except StopIteration:
        st.session_state.running = False
        st.session_state.alerts.append("MISSION COMPLETE - All objectives achieved")
        st.success("✓ MISSION COMPLETE")
        st.balloons()

# REPLAY LOOP - REPLAY MODE
if st.session_state.replay_playing and st.session_state.mode == "replay" and st.session_state.replay_data is not None:
    max_time = int(st.session_state.replay_data['time'].max())
    if st.session_state.replay_time < max_time:
        time.sleep(1.0 / st.session_state.replay_speed)
        st.session_state.replay_time += 1
        st.rerun()
    else:
        st.session_state.replay_playing = False
        st.success("✓ REPLAY COMPLETE")