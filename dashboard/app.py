from flask import Flask, request, jsonify, Response
import pickle
import pandas as pd
import sys
import os
import logging
import csv
import io

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
TOP_SHAP_FEATURES = 6
CHURN_HIGH_THRESHOLD = 0.7
CHURN_MEDIUM_THRESHOLD = 0.4

# ── Model loading ─────────────────────────────────────────────────────────────
def load_model():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(base, "models/xgb_churn.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(base, "models/feature_cols.pkl"), "rb") as f:
        feature_cols = pickle.load(f)
    return model, feature_cols

model, feature_cols = load_model()

# ── SHAP explainer cache (created once at startup, not on every request) ──────
shap_explainer = None
try:
    import shap
    shap_explainer = shap.TreeExplainer(model)
    logger.info("SHAP explainer loaded successfully.")
except Exception as e:
    logger.warning(f"SHAP not available: {e}")

# ── Feature mappings ──────────────────────────────────────────────────────────
MAPS = {
    "gender":           {"Female": 0, "Male": 1},
    "partner":          {"No": 0, "Yes": 1},
    "dependents":       {"No": 0, "Yes": 1},
    "phoneservice":     {"No": 0, "Yes": 1},
    "multiplelines":    {"No": 0, "No phone service": 1, "Yes": 2},
    "internetservice":  {"DSL": 0, "Fiber optic": 1, "No": 2},
    "onlinesecurity":   {"No": 0, "No internet service": 1, "Yes": 2},
    "onlinebackup":     {"No": 0, "No internet service": 1, "Yes": 2},
    "deviceprotection": {"No": 0, "No internet service": 1, "Yes": 2},
    "techsupport":      {"No": 0, "No internet service": 1, "Yes": 2},
    "streamingtv":      {"No": 0, "No internet service": 1, "Yes": 2},
    "streamingmovies":  {"No": 0, "No internet service": 1, "Yes": 2},
    "contract":         {"Month-to-month": 0, "One year": 1, "Two year": 2},
    "paperlessbilling": {"No": 0, "Yes": 1},
    "paymentmethod":    {
        "Bank transfer (automatic)": 0,
        "Credit card (automatic)": 1,
        "Electronic check": 2,
        "Mailed check": 3,
    },
}

REQUIRED_NUMERIC = {"tenure", "monthlycharges", "totalcharges", "seniorcitizen"}


def validate_input(raw: dict) -> list[str]:
    """Return a list of validation error messages (empty = valid)."""
    errors = []
    for field in REQUIRED_NUMERIC:
        value = raw.get(field)
        if value is None:
            errors.append(f"Missing required field: '{field}'")
            continue
        try:
            float(value)
        except (TypeError, ValueError):
            errors.append(f"Field '{field}' must be a number, got: {value!r}")
    for field in MAPS:
        value = raw.get(field)
        if value is not None and str(value) not in MAPS[field]:
            errors.append(
                f"Invalid value for '{field}': {value!r}. "
                f"Allowed: {list(MAPS[field].keys())}"
            )
    return errors


def preprocess(raw: dict) -> pd.DataFrame:
    processed = {}
    for col in feature_cols:
        if col in MAPS:
            processed[col] = MAPS[col].get(raw.get(col, list(MAPS[col].keys())[0]), 0)
        else:
            processed[col] = float(raw.get(col, 0))
    return pd.DataFrame([processed])


def get_risk_level(probability: float) -> str:
    if probability > CHURN_HIGH_THRESHOLD:
        return "High"
    if probability > CHURN_MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"


# ── HTML (unchanged) ──────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>ChurnAI Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono&display=swap" rel="stylesheet"/>
<style>
:root{--bg:#f4f6fb;--s:#ffffff;--s2:#f0f2f8;--b:#e2e6f0;--t:#1a1d2e;--m:#7b82a0;--p:#4f46e5;--pk:#e0478a;--g:#16a34a;--y:#d97706;--r:#dc2626;--shadow:0 2px 12px rgba(0,0,0,.07)}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--t);min-height:100vh}
header{background:var(--s);border-bottom:1px solid var(--b);padding:1rem 2rem;display:flex;align-items:center;gap:1rem;box-shadow:var(--shadow)}
.logo{font-size:1.5rem;font-weight:700;background:linear-gradient(90deg,var(--p),var(--pk));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.sub{color:var(--m);font-size:.83rem}
.layout{display:grid;grid-template-columns:290px 1fr;min-height:calc(100vh - 62px)}
.sidebar{background:var(--s);border-right:1px solid var(--b);padding:1.2rem;overflow-y:auto;height:calc(100vh - 62px);position:sticky;top:0}
.sidebar-title{font-size:.7rem;font-weight:600;color:var(--m);text-transform:uppercase;letter-spacing:.1em;margin-bottom:1rem}
.fg{margin-bottom:.7rem}
.fg label{display:block;font-size:.7rem;font-weight:500;color:var(--m);margin-bottom:.22rem;text-transform:uppercase;letter-spacing:.04em}
.fg select,.fg input[type=number]{width:100%;background:var(--s2);border:1px solid var(--b);border-radius:7px;color:var(--t);padding:.4rem .6rem;font-family:inherit;font-size:.82rem;outline:none;transition:border .2s}
.fg select:focus,.fg input:focus{border-color:var(--p);background:#fff}
.fg .rrow{display:flex;justify-content:space-between;align-items:center}
.fg .rv{font-size:.8rem;color:var(--p);font-family:'JetBrains Mono',monospace;font-weight:600}
.fg input[type=range]{width:100%;accent-color:var(--p);margin-top:.2rem}
.btn{width:100%;padding:.78rem;background:linear-gradient(135deg,var(--p),var(--pk));border:none;border-radius:10px;color:#fff;font-size:.88rem;font-weight:600;cursor:pointer;font-family:inherit;margin-top:.5rem;transition:opacity .2s;box-shadow:0 4px 14px rgba(79,70,229,.25)}
.btn:hover{opacity:.88}
.btn-export{width:100%;padding:.6rem;background:var(--s2);border:1px solid var(--b);border-radius:10px;color:var(--t);font-size:.82rem;font-weight:500;cursor:pointer;font-family:inherit;margin-top:.4rem;transition:background .2s}
.btn-export:hover{background:var(--b)}
.sep{border:none;border-top:1px solid var(--b);margin:.8rem 0}
.main{padding:1.4rem;overflow-y:auto}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:1.4rem}
.kpi{background:var(--s);border:1px solid var(--b);border-radius:13px;padding:1rem;box-shadow:var(--shadow);transition:transform .2s}
.kpi:hover{transform:translateY(-2px)}
.kpi .lb{font-size:.67rem;font-weight:600;color:var(--m);text-transform:uppercase;letter-spacing:.1em;margin-bottom:.28rem}
.kpi .vl{font-size:1.75rem;font-weight:700}
.rh{border-top:3px solid var(--r)}.rm{border-top:3px solid var(--y)}.rl{border-top:3px solid var(--g)}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1.4rem}
.card{background:var(--s);border:1px solid var(--b);border-radius:13px;padding:1.2rem;box-shadow:var(--shadow)}
.ct{font-size:.7rem;font-weight:600;color:var(--m);text-transform:uppercase;letter-spacing:.08em;margin-bottom:.9rem}
.gauge-wrap{display:flex;flex-direction:column;align-items:center;justify-content:center;height:210px}
.gauge-ring{width:148px;height:148px;border-radius:50%;display:flex;align-items:center;justify-content:center;transition:background .5s}
.gauge-inner{width:110px;height:110px;border-radius:50%;background:var(--s);display:flex;flex-direction:column;align-items:center;justify-content:center;box-shadow:inset 0 2px 8px rgba(0,0,0,.06)}
.gauge-pct{font-size:1.75rem;font-weight:700}
.gauge-lbl{font-size:.6rem;color:var(--m);font-weight:500;text-transform:uppercase;letter-spacing:.05em}
.risk-badge{margin-top:.65rem;font-size:.78rem;font-weight:600;padding:.22rem .85rem;border-radius:20px}
.shap-list{display:flex;flex-direction:column;gap:.5rem;height:210px;overflow-y:auto}
.shap-item{display:flex;align-items:center;gap:.55rem}
.shap-name{width:125px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:var(--t);font-size:.73rem}
.shap-bar-bg{flex:1;background:var(--s2);border-radius:4px;height:15px;overflow:hidden;border:1px solid var(--b)}
.shap-bar{height:100%;border-radius:4px;transition:width .5s;min-width:2px}
.shap-val{font-family:'JetBrains Mono',monospace;font-size:.68rem;width:50px;text-align:right}
.divider{border:none;border-top:1px solid var(--b);margin:1.4rem 0}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem}
.empty{color:var(--m);font-size:.82rem;padding:2rem;text-align:center;background:var(--s2);border-radius:9px}
.error-toast{position:fixed;bottom:1.5rem;right:1.5rem;background:#fee2e2;border:1px solid #fca5a5;color:#dc2626;padding:.8rem 1.2rem;border-radius:10px;font-size:.82rem;font-weight:500;box-shadow:var(--shadow);display:none;z-index:999}
</style>
</head>
<body>
<div class="error-toast" id="error-toast"></div>
<header>
  <div class="logo">🧠 ChurnAI</div>
  <div class="sub">Customer Churn Prediction Dashboard &nbsp;·&nbsp; IBM Telco Dataset (7,043 customers)</div>
</header>
<div class="layout">
<aside class="sidebar">
  <div class="sidebar-title">👤 Customer Profile</div>
  <div class="fg"><label>Gender</label><select id="gender"><option>Male</option><option>Female</option></select></div>
  <div class="fg"><label>Senior Citizen</label><select id="seniorcitizen"><option value="0">No</option><option value="1">Yes</option></select></div>
  <div class="fg"><label>Partner</label><select id="partner"><option>No</option><option>Yes</option></select></div>
  <div class="fg"><label>Dependents</label><select id="dependents"><option>No</option><option>Yes</option></select></div>
  <div class="fg">
    <div class="rrow"><label>Tenure (months)</label><span class="rv" id="tenure-val">12</span></div>
    <input type="range" id="tenure" min="0" max="72" value="12" oninput="document.getElementById('tenure-val').textContent=this.value;el('kpi-tenure').textContent=this.value+' mo'">
  </div>
  <hr class="sep"/>
  <div class="fg"><label>Phone Service</label><select id="phoneservice"><option>Yes</option><option>No</option></select></div>
  <div class="fg"><label>Multiple Lines</label><select id="multiplelines"><option>No</option><option>Yes</option><option>No phone service</option></select></div>
  <div class="fg"><label>Internet Service</label><select id="internetservice"><option>Fiber optic</option><option>DSL</option><option>No</option></select></div>
  <div class="fg"><label>Online Security</label><select id="onlinesecurity"><option>No</option><option>Yes</option><option>No internet service</option></select></div>
  <div class="fg"><label>Online Backup</label><select id="onlinebackup"><option>No</option><option>Yes</option><option>No internet service</option></select></div>
  <div class="fg"><label>Device Protection</label><select id="deviceprotection"><option>No</option><option>Yes</option><option>No internet service</option></select></div>
  <div class="fg"><label>Tech Support</label><select id="techsupport"><option>No</option><option>Yes</option><option>No internet service</option></select></div>
  <div class="fg"><label>Streaming TV</label><select id="streamingtv"><option>No</option><option>Yes</option><option>No internet service</option></select></div>
  <div class="fg"><label>Streaming Movies</label><select id="streamingmovies"><option>No</option><option>Yes</option><option>No internet service</option></select></div>
  <hr class="sep"/>
  <div class="fg"><label>Contract</label><select id="contract" onchange="el('kpi-contract').textContent=this.value"><option>Month-to-month</option><option>One year</option><option>Two year</option></select></div>
  <div class="fg"><label>Paperless Billing</label><select id="paperlessbilling"><option>Yes</option><option>No</option></select></div>
  <div class="fg"><label>Payment Method</label><select id="paymentmethod"><option>Electronic check</option><option>Mailed check</option><option>Bank transfer (automatic)</option><option>Credit card (automatic)</option></select></div>
  <div class="fg"><label>Monthly Charges ($)</label><input type="number" id="monthlycharges" value="65" min="18" max="120" step="1"></div>
  <div class="fg"><label>Total Charges ($)</label><input type="number" id="totalcharges" value="780" min="0" max="9000" step="50"></div>
  <button class="btn" onclick="predict()">⚡ Analyze Customer</button>
  <button class="btn-export" onclick="exportResult()">📥 Export Result as CSV</button>
</aside>
<main class="main">
  <div class="kpis">
    <div class="kpi rl" id="kpi-prob"><div class="lb">Churn Probability</div><div class="vl" id="prob-val" style="color:var(--g)">—</div></div>
    <div class="kpi rl" id="kpi-risk"><div class="lb">Risk Level</div><div class="vl" id="risk-val" style="color:var(--g)">—</div></div>
    <div class="kpi"><div class="lb">Tenure</div><div class="vl" style="color:var(--p)" id="kpi-tenure">12 mo</div></div>
    <div class="kpi"><div class="lb">Contract</div><div class="vl" style="color:var(--p);font-size:1rem;padding-top:.4rem" id="kpi-contract">Month-to-month</div></div>
  </div>
  <div class="charts">
    <div class="card">
      <div class="ct">📊 Churn Risk Gauge</div>
      <div class="gauge-wrap">
        <div class="gauge-ring" id="gauge-ring" style="background:conic-gradient(#e2e6f0 360deg,#e2e6f0 0deg)">
          <div class="gauge-inner">
            <div class="gauge-pct" id="gauge-pct" style="color:var(--m)">—</div>
            <div class="gauge-lbl">Churn Risk</div>
          </div>
        </div>
        <div class="risk-badge" id="risk-badge" style="background:#f0f2f8;color:var(--m)">Awaiting analysis</div>
      </div>
    </div>
    <div class="card">
      <div class="ct">🔍 Top Risk Factors (SHAP)</div>
      <div class="shap-list" id="shap-list"><div class="empty">Click "Analyze Customer" to see SHAP factors</div></div>
    </div>
  </div>
  <hr class="divider"/>
  <div class="ct" style="margin-bottom:1rem">📈 Dataset Overview — IBM Telco</div>
  <div class="stats">
    <div class="card"><div class="ct">Churn Distribution</div><canvas id="pieChart" height="180"></canvas></div>
    <div class="card"><div class="ct">Churn Rate by Contract</div><canvas id="contractChart" height="180"></canvas></div>
    <div class="card"><div class="ct">Churn Rate by Tenure Group</div><canvas id="tenureChart" height="180"></canvas></div>
  </div>
</main>
</div>
<script>
const C={red:'#dc2626',yellow:'#d97706',green:'#16a34a',purple:'#4f46e5',muted:'#7b82a0'};
function el(id){return document.getElementById(id)}
function val(id){return el(id).value}

let lastResult = null;

function showError(msg) {
  const toast = el('error-toast');
  toast.textContent = '⚠️ ' + msg;
  toast.style.display = 'block';
  setTimeout(() => { toast.style.display = 'none'; }, 4000);
}

function getFormData() {
  return {
    gender:val('gender'),seniorcitizen:parseInt(val('seniorcitizen')),
    partner:val('partner'),dependents:val('dependents'),tenure:parseInt(val('tenure')),
    phoneservice:val('phoneservice'),multiplelines:val('multiplelines'),
    internetservice:val('internetservice'),onlinesecurity:val('onlinesecurity'),
    onlinebackup:val('onlinebackup'),deviceprotection:val('deviceprotection'),
    techsupport:val('techsupport'),streamingtv:val('streamingtv'),
    streamingmovies:val('streamingmovies'),contract:val('contract'),
    paperlessbilling:val('paperlessbilling'),paymentmethod:val('paymentmethod'),
    monthlycharges:parseFloat(val('monthlycharges')),totalcharges:parseFloat(val('totalcharges'))
  };
}

async function predict(){
  const data = getFormData();
  try {
    const res = await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
    const r = await res.json();
    if (!res.ok) { showError(r.error || 'Prediction failed'); return; }
    lastResult = { ...data, ...r };
    const color=r.risk==='High'?C.red:r.risk==='Medium'?C.yellow:C.green;
    const emoji=r.risk==='High'?'🔴':r.risk==='Medium'?'🟡':'🟢';
    const rc=r.risk==='High'?'rh':r.risk==='Medium'?'rm':'rl';
    const bg=r.risk==='High'?'#fee2e2':r.risk==='Medium'?'#fef3c7':'#dcfce7';
    el('prob-val').textContent=r.percentage+'%';el('prob-val').style.color=color;
    el('risk-val').textContent=r.risk+' '+emoji;el('risk-val').style.color=color;
    el('kpi-prob').className='kpi '+rc;el('kpi-risk').className='kpi '+rc;
    const deg=Math.round(r.probability*360);
    el('gauge-ring').style.background=`conic-gradient(${color} ${deg}deg,#e2e6f0 ${deg}deg)`;
    el('gauge-pct').textContent=r.percentage+'%';el('gauge-pct').style.color=color;
    el('risk-badge').textContent=r.risk.toUpperCase()+' RISK '+emoji;
    el('risk-badge').style.background=bg;el('risk-badge').style.color=color;
    if(r.shap_factors&&r.shap_factors.length){
      const mx=Math.max(...r.shap_factors.map(f=>Math.abs(f.shap_value)));
      el('shap-list').innerHTML=r.shap_factors.map(f=>{
        const w=Math.round((Math.abs(f.shap_value)/mx)*100);
        const c=f.impact==='increases'?C.red:C.green;
        return `<div class="shap-item"><div class="shap-name">${f.feature}</div><div class="shap-bar-bg"><div class="shap-bar" style="width:${w}%;background:${c}"></div></div><div class="shap-val" style="color:${c}">${f.shap_value>0?'+':''}${f.shap_value.toFixed(3)}</div></div>`;
      }).join('');
    }
  } catch(e) {
    showError('Network error — is the server running?');
  }
}

function exportResult() {
  if (!lastResult) { showError('Run an analysis first before exporting.'); return; }
  const res = await fetch('/export', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(lastResult)
  });
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'churn_result.csv'; a.click();
  URL.revokeObjectURL(url);
}

async function loadStats(){
  try {
    const res=await fetch('/dataset-stats');const d=await res.json();if(d.error)return;
    const base={responsive:true,plugins:{legend:{labels:{color:'#1a1d2e',font:{family:'Inter',size:11}}}},scales:{x:{ticks:{color:C.muted},grid:{color:'#e2e6f0'}},y:{ticks:{color:C.muted},grid:{color:'#e2e6f0'}}}};
    new Chart(el('pieChart'),{type:'doughnut',data:{labels:['Retained','Churned'],datasets:[{data:[100-d.churn_rate,d.churn_rate],backgroundColor:[C.green,C.red],borderWidth:2,borderColor:'#fff'}]},options:{responsive:true,cutout:'60%',plugins:{legend:{labels:{color:'#1a1d2e',font:{family:'Inter',size:11}}}}}});
    new Chart(el('contractChart'),{type:'bar',data:{labels:Object.keys(d.contract_churn),datasets:[{label:'Churn %',data:Object.values(d.contract_churn).map(v=>Math.round(v*100)),backgroundColor:[C.red,C.yellow,C.green],borderRadius:6,borderSkipped:false}]},options:{...base,plugins:{legend:{display:false}}}});
    new Chart(el('tenureChart'),{type:'line',data:{labels:Object.keys(d.tenure_churn),datasets:[{label:'Churn %',data:Object.values(d.tenure_churn).map(v=>Math.round(v*100)),borderColor:C.purple,backgroundColor:'rgba(79,70,229,0.08)',fill:true,tension:0.4,pointBackgroundColor:C.purple,pointRadius:4}]},options:base});
  } catch(e) { console.warn('Could not load dataset stats:', e); }
}
loadStats();predict();
</script>
</body>
</html>"""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return HTML


@app.route("/health")
def health():
    """Health check endpoint — useful for deployment monitoring."""
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    errors = validate_input(data)
    if errors:
        logger.warning(f"Invalid predict input: {errors}")
        return jsonify({"error": "; ".join(errors)}), 422

    try:
        df_input = preprocess(data)
        prob = float(model.predict_proba(df_input)[0][1])
        risk = get_risk_level(prob)

        shap_factors = []
        if shap_explainer is not None:
            shap_values = shap_explainer.shap_values(df_input)
            for f, v in zip(feature_cols, shap_values[0]):
                shap_factors.append({
                    "feature": f.replace("_", " ").title(),
                    "shap_value": round(float(v), 4),
                    "impact": "increases" if v > 0 else "decreases",
                })
            shap_factors.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
            shap_factors = shap_factors[:TOP_SHAP_FEATURES]

        logger.info(f"Prediction: prob={prob:.3f}, risk={risk}")
        return jsonify({
            "probability": round(prob, 4),
            "percentage": round(prob * 100, 1),
            "risk": risk,
            "shap_factors": shap_factors,
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({"error": "Internal prediction error."}), 500


@app.route("/export", methods=["POST"])
def export():
    """Export a single prediction result as a downloadable CSV."""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided."}), 400

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=data.keys())
    writer.writeheader()
    writer.writerow(data)

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=churn_result.csv"},
    )


@app.route("/dataset-stats")
def dataset_stats():
    try:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        df = pd.read_csv(os.path.join(base, "data/churn_dataset.csv"))
        if df["churn"].dtype == object:
            df["churn"] = (df["churn"] == "Yes").astype(int)
        churn_rate = round(float(df["churn"].mean()) * 100, 1)
        tenure_bins = pd.cut(
            df["tenure"],
            bins=[0, 12, 24, 36, 48, 60, 72],
            labels=["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"],
        )
        tenure_churn = df.groupby(tenure_bins, observed=True)["churn"].mean().round(3).to_dict()
        contract_churn = df.groupby("contract")["churn"].mean().round(3).to_dict()
        return jsonify({
            "total": len(df),
            "churned": int(df["churn"].sum()),
            "churn_rate": churn_rate,
            "tenure_churn": {str(k): float(v) for k, v in tenure_churn.items()},
            "contract_churn": {str(k): float(v) for k, v in contract_churn.items()},
        })
    except Exception as e:
        logger.error(f"Dataset stats error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
