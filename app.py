from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

app = Flask(__name__, template_folder='.')

# FUZZY ENGINE SETUP (Global)

# 1. Variables & Universe (4 Inputs, 1 Output)
stok_ctrl = ctrl.Antecedent(np.arange(0, 101, 1), 'stok')             # Sisa Stok (0-100%)
permintaan_ctrl = ctrl.Antecedent(np.arange(0, 101, 1), 'permintaan') # Permintaan Harian (0-100 unit/hari)
pengiriman_ctrl = ctrl.Antecedent(np.arange(1, 15, 1), 'pengiriman')  # Lama Pengiriman Supplier (1-14 hari)
kedaluwarsa_ctrl = ctrl.Antecedent(np.arange(1, 61, 1), 'kedaluwarsa')# Sisa Masa Kedaluwarsa (1-60 hari)

urgensi_ctrl = ctrl.Consequent(np.arange(0, 101, 1), 'urgensi')       # Tingkat Urgensi (0-100%)

# 2. Membership Functions
# Input 1: Sisa Stok
stok_ctrl['sedikit'] = fuzz.trapmf(stok_ctrl.universe, [0, 0, 20, 40])
stok_ctrl['sedang']  = fuzz.trimf(stok_ctrl.universe, [30, 50, 70])
stok_ctrl['banyak']  = fuzz.trapmf(stok_ctrl.universe, [60, 80, 100, 100])

# Input 2: Permintaan Harian
permintaan_ctrl['rendah'] = fuzz.trapmf(permintaan_ctrl.universe, [0, 0, 20, 40])
permintaan_ctrl['sedang'] = fuzz.trimf(permintaan_ctrl.universe, [30, 50, 70])
permintaan_ctrl['tinggi'] = fuzz.trapmf(permintaan_ctrl.universe, [60, 80, 100, 100])

# Input 3: Lama Pengiriman (Lead Time)
pengiriman_ctrl['cepat']  = fuzz.trimf(pengiriman_ctrl.universe, [1, 1, 5])
pengiriman_ctrl['sedang'] = fuzz.trimf(pengiriman_ctrl.universe, [3, 7, 10])
pengiriman_ctrl['lama']   = fuzz.trapmf(pengiriman_ctrl.universe, [8, 12, 14, 14])

# Input 4: Sisa Umur Simpan (Kedaluwarsa)
kedaluwarsa_ctrl['kritis'] = fuzz.trapmf(kedaluwarsa_ctrl.universe, [1, 1, 5, 10])
kedaluwarsa_ctrl['dekat']  = fuzz.trimf(kedaluwarsa_ctrl.universe, [7, 15, 25])
kedaluwarsa_ctrl['jauh']   = fuzz.trapmf(kedaluwarsa_ctrl.universe, [20, 35, 60, 60])

# Output: Urgensi Restock
urgensi_ctrl['rendah'] = fuzz.trapmf(urgensi_ctrl.universe, [0, 0, 20, 40])
urgensi_ctrl['sedang'] = fuzz.trimf(urgensi_ctrl.universe, [30, 50, 70])
urgensi_ctrl['tinggi'] = fuzz.trimf(urgensi_ctrl.universe, [60, 75, 90])
urgensi_ctrl['kritis'] = fuzz.trapmf(urgensi_ctrl.universe, [80, 90, 100, 100])

# 3. Fuzzy Rules
rules = [
    # PRIORITAS UTAMA: Stok vs Permintaan (9 Kombinasi)
    ctrl.Rule(stok_ctrl['sedikit'] & permintaan_ctrl['tinggi'], urgensi_ctrl['kritis']),
    ctrl.Rule(stok_ctrl['sedikit'] & permintaan_ctrl['sedang'], urgensi_ctrl['kritis']),
    ctrl.Rule(stok_ctrl['sedikit'] & permintaan_ctrl['rendah'], urgensi_ctrl['tinggi']),
    
    ctrl.Rule(stok_ctrl['sedang'] & permintaan_ctrl['tinggi'], urgensi_ctrl['tinggi']),
    ctrl.Rule(stok_ctrl['sedang'] & permintaan_ctrl['sedang'], urgensi_ctrl['sedang']),
    ctrl.Rule(stok_ctrl['sedang'] & permintaan_ctrl['rendah'], urgensi_ctrl['rendah']),

    ctrl.Rule(stok_ctrl['banyak'] & permintaan_ctrl['tinggi'], urgensi_ctrl['sedang']),
    ctrl.Rule(stok_ctrl['banyak'] & permintaan_ctrl['sedang'], urgensi_ctrl['rendah']),
    ctrl.Rule(stok_ctrl['banyak'] & permintaan_ctrl['rendah'], urgensi_ctrl['rendah']),

    # Faktor Tambahan: Pengiriman (Lead Time)
    ctrl.Rule(pengiriman_ctrl['lama'] & stok_ctrl['sedikit'], urgensi_ctrl['kritis']),
    ctrl.Rule(pengiriman_ctrl['lama'] & stok_ctrl['sedang'], urgensi_ctrl['tinggi']),
    
    # Faktor Tambahan: Kedaluwarsa (Expired)
    ctrl.Rule(kedaluwarsa_ctrl['kritis'] & stok_ctrl['banyak'], urgensi_ctrl['rendah']),
    ctrl.Rule(kedaluwarsa_ctrl['dekat'] & stok_ctrl['sedikit'], urgensi_ctrl['tinggi']),
    ctrl.Rule(kedaluwarsa_ctrl['kritis'] & stok_ctrl['sedikit'], urgensi_ctrl['kritis']),
]

# 4. Setup Control System
urgensi_cs = ctrl.ControlSystem(rules)
urgensi_sim = ctrl.ControlSystemSimulation(urgensi_cs)

def hitung_urgensi(stok: int, permintaan: int, pengiriman: int, kedaluwarsa: int) -> dict:
    # Set inputs
    urgensi_sim.input['stok'] = stok
    urgensi_sim.input['permintaan'] = permintaan
    urgensi_sim.input['pengiriman'] = pengiriman
    urgensi_sim.input['kedaluwarsa'] = kedaluwarsa
    
    # Compute
    try:
        urgensi_sim.compute()
        hasil = float(urgensi_sim.output['urgensi'])
    except Exception as e:
        print(f"Error fuzzy compute: {e}")
        hasil = 0.0

    # Tentukan keterangan label
    keterangan = "Normal"
    if hasil >= 80: keterangan = "Kritis (Segera Pesan!)"
    elif hasil >= 60: keterangan = "Tinggi (Prioritaskan)"
    elif hasil >= 30: keterangan = "Sedang (Pantau)"
    else: keterangan = "Rendah (Belum Perlu)"

    # Membership degrees for UI
    degrees = {
        "input": {
            "stok": {
                "sedikit": float(fuzz.interp_membership(stok_ctrl.universe, stok_ctrl['sedikit'].mf, stok)),
                "sedang": float(fuzz.interp_membership(stok_ctrl.universe, stok_ctrl['sedang'].mf, stok)),
                "banyak": float(fuzz.interp_membership(stok_ctrl.universe, stok_ctrl['banyak'].mf, stok)),
            },
            "permintaan": {
                "rendah": float(fuzz.interp_membership(permintaan_ctrl.universe, permintaan_ctrl['rendah'].mf, permintaan)),
                "sedang": float(fuzz.interp_membership(permintaan_ctrl.universe, permintaan_ctrl['sedang'].mf, permintaan)),
                "tinggi": float(fuzz.interp_membership(permintaan_ctrl.universe, permintaan_ctrl['tinggi'].mf, permintaan)),
            },
            "pengiriman": {
                "cepat": float(fuzz.interp_membership(pengiriman_ctrl.universe, pengiriman_ctrl['cepat'].mf, pengiriman)),
                "sedang": float(fuzz.interp_membership(pengiriman_ctrl.universe, pengiriman_ctrl['sedang'].mf, pengiriman)),
                "lama": float(fuzz.interp_membership(pengiriman_ctrl.universe, pengiriman_ctrl['lama'].mf, pengiriman)),
            },
            "kedaluwarsa": {
                "kritis": float(fuzz.interp_membership(kedaluwarsa_ctrl.universe, kedaluwarsa_ctrl['kritis'].mf, kedaluwarsa)),
                "dekat": float(fuzz.interp_membership(kedaluwarsa_ctrl.universe, kedaluwarsa_ctrl['dekat'].mf, kedaluwarsa)),
                "jauh": float(fuzz.interp_membership(kedaluwarsa_ctrl.universe, kedaluwarsa_ctrl['jauh'].mf, kedaluwarsa)),
            }
        },
        "output": {
            "rendah": float(fuzz.interp_membership(urgensi_ctrl.universe, urgensi_ctrl['rendah'].mf, hasil)),
            "sedang": float(fuzz.interp_membership(urgensi_ctrl.universe, urgensi_ctrl['sedang'].mf, hasil)),
            "tinggi": float(fuzz.interp_membership(urgensi_ctrl.universe, urgensi_ctrl['tinggi'].mf, hasil)),
            "kritis": float(fuzz.interp_membership(urgensi_ctrl.universe, urgensi_ctrl['kritis'].mf, hasil)),
        }
    }

    return {
        "persentase": round(hasil, 2),
        "keterangan": keterangan,
        "degrees": degrees
    }


# FLASK ROUTES

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/fuzzy", methods=["POST"])
def api_fuzzy():
    data = request.get_json(force=True)
    stok = int(data.get("stok", 50))
    permintaan = int(data.get("permintaan", 50))
    pengiriman = int(data.get("pengiriman", 7))
    kedaluwarsa = int(data.get("kedaluwarsa", 30))

    result = hitung_urgensi(stok, permintaan, pengiriman, kedaluwarsa)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=5002)
