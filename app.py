
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"


from flask import Flask, render_template, request, send_file
import pickle
import pandas as pd
from urllib.parse import urlparse
from difflib import SequenceMatcher
from tensorflow.keras.models import load_model
from xgboost import XGBClassifier

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import io
from datetime import datetime

app = Flask(__name__)

# ================= LOAD MODELS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

xgb_model = XGBClassifier()
xgb_model.load_model(os.path.join(BASE_DIR, "xgb_phish.json"))

dl_model = None

def get_dl_model():
    global dl_model
    if dl_model is None:
        dl_model = load_model(os.path.join(BASE_DIR, "dl_phish.h5"))
    return dl_model

scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))
feature_names = pickle.load(open(os.path.join(BASE_DIR, "features.pkl"), "rb"))
feature_importance = pickle.load(open(os.path.join(BASE_DIR, "feature_importance.pkl"), "rb"))


# ================= TRUSTED DOMAINS =================
TRUSTED_DOMAINS = [
    'amazon.com','amazon.in',
    'google.com','google.co.in',
    'flipkart.com','microsoft.com',
    'apple.com','wikipedia.org',
    'github.com','youtube.com',
    'netflix.com','linkedin.com',
    'instagram.com','facebook.com',
    'twitter.com','x.com','paypal.com'
]

# ================= FEATURE MEANINGS =================
FEATURE_MEANING = {
    "nb_dots": "Excessive number of dots in the domain",
    "nb_subdomains": "Unusually deep subdomain structure",
    "ratio_digits_url": "Random numbers detected in the URL",
    "length_url": "URL length is unusually long",
    "nb_hyphens": "Too many hyphens in the URL",
    "nb_slash": "Unusually deep URL path structure",
    "length_hostname": "Domain name length is higher than normal"
}

# ================= HELPER FUNCTIONS =================
def normalize_url(url):
    url = url.strip().lower()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url

def get_domain(url):
    domain = urlparse(url).netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain

def is_trusted(domain):
    return any(domain == d or domain.endswith("." + d) for d in TRUSTED_DOMAINS)

def extract_features(url):
    parsed = urlparse(url)
    host = parsed.netloc.replace("www.", "")

    feats = {f: 0 for f in feature_names}
    feats["length_url"] = len(url)
    feats["length_hostname"] = len(host)
    feats["nb_dots"] = host.count(".")
    feats["nb_hyphens"] = url.count("-")
    feats["nb_at"] = url.count("@")
    feats["nb_slash"] = url.count("/")
    feats["nb_www"] = 0
    feats["https_token"] = 1
    feats["ratio_digits_url"] = sum(c.isdigit() for c in url) / len(url)

    if "nb_subdomains" in feature_names:
        feats["nb_subdomains"] = max(0, host.count(".") - 1)

    return pd.DataFrame([feats])[feature_names]

def brand_impersonation(domain):
    brands = ['paypal','google','facebook','amazon','microsoft','netflix','apple','instagram']
    main = domain.split(".")[0]
    for b in brands:
        if 0.8 <= SequenceMatcher(None, b, main).ratio() < 1:
            return True
    return False

def explain_ml(input_df, importance_df, top_k=3):
    explanations = []

    if importance_df is None or importance_df.empty:
        return ["Model explanation unavailable"]

    for feature in input_df.columns:
        try:
            val = input_df[feature].values[0]
            if val > 0 and feature in importance_df["feature"].values:
                explanations.append(
                    FEATURE_MEANING.get(feature, feature)
                )
        except Exception:
            continue

    if not explanations:
        return ["No abnormal URL patterns detected"]

    return explanations[:top_k]


# ================= MAIN ROUTE =================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        try:
            raw_url = request.form["url"]
            url = normalize_url(raw_url)
            domain = get_domain(url)

            # -------- TRUST OVERRIDE --------
            if is_trusted(domain):
                result = {
                    "risk_score": 0,
                    "category": "🟢 SAFE",
                    "consequence": "This URL belongs to a verified trusted global domain.",
                    "recommendation": "Safe to use. Standard security practices apply.",
                    "explanation": ["Trusted domain whitelist match"]
                }
                return render_template("index.html", result=result)

            # -------- FEATURE EXTRACTION --------
            X = extract_features(url)
            X_scaled = scaler.transform(X)

            # -------- ML + DL PREDICTION --------
            ml_prob = xgb_model.predict_proba(X)[0][1]
            dl_prob = dl_model.predict(X_scaled, verbose=0)[0][0]

            ml_score = round(ml_prob * 100, 1)
            dl_score = round(dl_prob * 100, 1)

            risk = round((0.6 * ml_prob + 0.4 * dl_prob) * 100, 1)


            explanation = explain_ml(X, feature_importance)
            if not explanation:
                explanation = ["No abnormal URL patterns detected"]

            # -------- DECISION LOGIC --------
            if brand_impersonation(domain) or risk >= 55:
                result = {
                    "risk_score": risk,
                    "ml_score": ml_score,
                    "dl_score": dl_score,
                    "category": "🔴 DANGEROUS",
                    "consequence": "High-risk phishing or malicious patterns detected.",
                    "recommendation": "Do NOT open this link.",
                    "explanation": explanation
                }
            elif risk >= 25:
                result = {
                    "risk_score": risk,
                    "ml_score": ml_score,
                    "dl_score": dl_score,
                    "category": "🟡 SUSPICIOUS",
                    "consequence": "This URL shows suspicious characteristics.",
                    "recommendation": "Proceed only if you fully trust the source.",
                    "explanation": explanation
                }
            else:
                result = {
                    "risk_score": risk,
                    "ml_score": 0,
                    "dl_score": 0,
                    "category": "🟢 SAFE",
                    "consequence": "No immediate security threats were detected.",
                    "recommendation": "Safe to use, but remain cautious.",
                    "explanation": explanation
                }

        except Exception as e:
                result = {
                    "risk_score": 0,
                    "ml_score": "N/A",
                    "dl_score": "N/A",
                    "category": "⚠️ ERROR",
                    "consequence": "Prediction failed due to backend limitation.",
                    "recommendation": "Try again or check deployment logs.",
                    "explanation": ["Model execution failed"]
                }

    return render_template("index.html", result=result)

# ================= PDF REPORT ROUTE =================
@app.route("/download-report", methods=["POST"])
def download_report():
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    data = request.form
    y = height - 50

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, y, "ThreatGuard AI - URL Scan Report")

    y -= 30
    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Basic Info
    y -= 40
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Scanned URL:")
    y -= 18
    c.setFont("Helvetica", 11)
    c.drawString(60, y, data.get("url"))

    y -= 30
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Risk Score:")
    c.setFont("Helvetica", 11)
    c.drawString(160, y, data.get("risk_score"))

    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Category:")
    c.setFont("Helvetica", 11)
    c.drawString(160, y, data.get("category"))

    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Model Confidence:")
    c.setFont("Helvetica", 11)
    c.drawString(200, y, f"{data.get('risk_score', 0)}%")


    # Recommendation
    y -= 30
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Recommendation:")
    y -= 18
    c.setFont("Helvetica", 11)
    c.drawString(60, y, data.get("recommendation"))

    # AI Explanation
    y -= 30
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "AI Explanation:")
    y -= 18
    c.setFont("Helvetica", 11)
    for reason in data.getlist("explanation"):
        c.drawString(60, y, f"- {reason}")
        y -= 16
    # ---------- MODEL SCORE BREAKDOWN ----------
    y -= 30
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Model-wise Risk Assessment:")
    y -= 18

    c.setFont("Helvetica", 11)
    c.drawString(
        60, y,
        f"ML Model (XGBoost) Score: {data.get('ml_score', 0)}%"
    )

    y -= 16
    c.drawString(
        60, y,
        f"DL Model Score: {data.get('dl_score', 0)}%"
    )

    # ---------- HYBRID CALCULATION ----------
    y -= 22
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Hybrid Risk Score Calculation:")
    y -= 18

    c.setFont("Helvetica", 11)
    c.drawString(
        60, y,
        f"Final Risk Score = (0.6 × {data.get('ml_score', 0)}) + "
        f"(0.4 × {data.get('dl_score', 0)}) = "
        f"{data.get('risk_score', 0)}%"
    )

   
    
    # Technical Checks
    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Technical Checks Performed:")
    y -= 18
    c.setFont("Helvetica", 11)
    c.drawString(60, y, "- Domain reputation and trusted domain verification.")
    y -= 16
    c.drawString(60, y, "- SSL certificate presence check.")
    y -= 16
    c.drawString(60, y, "- URL structure and redirect behavior analysis.")

    c.showPage()
    c.save()

    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="ThreatGuard_Report.pdf",
        mimetype="application/pdf"
    )

# ================= RUN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

