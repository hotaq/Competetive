"""Build the canonical portable HTML report artifact for the EDA."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
generated_at = datetime.now(timezone.utc).isoformat()

summary = json.loads((REPORT_DIR / "eda_summary.json").read_text(encoding="utf-8"))["quality"]
profile = pd.read_csv(REPORT_DIR / "profile.csv")
signal = pd.read_csv(REPORT_DIR / "numeric_signal.csv")
bins = pd.read_csv(REPORT_DIR / "binned_target.csv")
drift = pd.read_csv(REPORT_DIR / "drift.csv")
importance = pd.read_csv(REPORT_DIR / "permutation_importance.csv")

top_features = ["daily_screen_time_hours", "weekend_screen_time", "social_media_hours"]
feature_labels = {
    "daily_screen_time_hours": "เวลาหน้าจอรายวัน",
    "weekend_screen_time": "เวลาหน้าจอวันหยุด",
    "social_media_hours": "เวลาโซเชียลมีเดีย",
}

decile_rows = []
for decile in range(1, 11):
    row = {"decile": decile}
    for feature in top_features:
        match = bins[(bins["feature"] == feature) & (bins["decile"] == decile)].iloc[0]
        row[feature] = round(float(match["target_rate"]), 6)
        row[f"{feature}_count"] = int(match["count"])
        row[f"{feature}_range"] = match["range"]
    decile_rows.append(row)

importance_rows = []
for rank, (_, row) in enumerate(importance.iterrows(), start=1):
    importance_rows.append({
        "feature": row["feature"],
        "importance": round(float(row["importance_mean"]), 6),
        "uncertainty": round(float(row["importance_std"]), 6),
        "rank": rank,
    })

profile_rows = []
for _, row in profile.sort_values("train_missing_pct", ascending=False).iterrows():
    profile_rows.append({
        "feature": row["feature"],
        "type": row["dtype"],
        "train_missing_pct": round(float(row["train_missing_pct"]), 2),
        "test_missing_pct": round(float(row["test_missing_pct"]), 2),
        "missing_diff_pp": round(float(row["test_missing_pct"] - row["train_missing_pct"]), 2),
        "train_unique": int(row["train_unique"]),
        "test_unique": int(row["test_unique"]),
    })

drift_rows = []
for _, row in drift.iterrows():
    drift_rows.append({
        "feature": row["feature"],
        "kind": row["kind"],
        "drift_score": round(float(row["drift_score"]), 6),
        "missing_diff_pp": round(float(row["missing_diff_pp"]), 2),
    })

headline = [{
    "train_rows": summary["train_rows"],
    "features": summary["feature_count"],
    "target_rate": summary["target_rate"],
    "holdout_auc": summary["holdout_auc_histgb"],
}]

sources = [
    {
        "id": "competition-data",
        "label": "Kaggle Playground Series S6E8 — train.csv และ test.csv",
        "query": {
            "language": "sql",
            "engine": "duckdb",
            "sql": "SELECT * FROM read_csv_auto('data/raw/train.csv') UNION ALL BY NAME SELECT *, NULL::INTEGER AS addicted_label FROM read_csv_auto('data/raw/test.csv')",
            "description": "การตรวจคุณภาพ การกระจาย target และ train–test drift จากไฟล์การแข่งขัน",
            "tables_used": ["data/raw/train.csv", "data/raw/test.csv"],
            "filters": ["ใช้ทุกแถว", "ตัด id และ addicted_label ออกจากชุดฟีเจอร์เมื่อสร้างโมเดล"],
            "metric_definitions": [
                "target rate = ค่าเฉลี่ยของ addicted_label",
                "missing difference = อัตรา missing ใน test ลบ train หน่วย percentage point",
                "PSI ใช้ quantile bins 10 กลุ่มจาก train",
            ],
        },
    },
    {
        "id": "analysis-code",
        "label": "Reproducible EDA และ holdout baseline",
        "path": "src/deep_dive.py",
        "query": {
            "language": "sql",
            "engine": "duckdb",
            "sql": "SELECT * FROM read_csv_auto('reports/permutation_importance.csv')",
            "description": "สคริปต์สร้างผล EDA ตาราง drift และโมเดล HistGradientBoosting holdout",
            "tables_used": ["data/raw/train.csv", "data/raw/test.csv"],
            "filters": ["สุ่มตัวอย่าง 350,000 แถวด้วย random_state=42 สำหรับ holdout baseline", "validation 25% แบบ stratified"],
            "metric_definitions": [
                "holdout AUC = ROC AUC บน validation split ที่ไม่ถูกใช้ฝึก",
                "permutation importance = ค่า ROC AUC ที่ลดลงเมื่อสลับค่าฟีเจอร์บนตัวอย่าง validation 30,000 แถว",
            ],
        },
    },
]

manifest = {
    "version": 1,
    "surface": "report",
    "title": "เจาะลึกข้อมูล Smartphone Addiction",
    "description": "ภาพรวมคุณภาพข้อมูล สัญญาณของ target และแนวทางสร้างโมเดลสำหรับ Kaggle Playground Series S6E8",
    "generatedAt": generated_at,
    "sources": sources,
    "cards": [
        {"id": "rows", "dataset": "headline", "sourceId": "competition-data", "description": "จำนวนแถวฝึกทั้งหมด", "metrics": [{"label": "Train rows", "field": "train_rows", "format": "compact"}]},
        {"id": "features", "dataset": "headline", "sourceId": "competition-data", "description": "ฟีเจอร์ที่ใช้ได้ ไม่รวม id และ target", "metrics": [{"label": "Features", "field": "features", "format": "number"}]},
        {"id": "target", "dataset": "headline", "sourceId": "competition-data", "description": "สัดส่วน addicted_label = 1", "metrics": [{"label": "Positive class", "field": "target_rate", "format": "percent"}]},
        {"id": "auc", "dataset": "headline", "sourceId": "analysis-code", "description": "Diagnostic holdout จาก HistGradientBoosting", "metrics": [{"label": "Holdout ROC AUC", "field": "holdout_auc", "format": "number"}]},
    ],
    "charts": [
        {
            "id": "target-deciles",
            "title": "อัตรา addicted_label ตาม decile ของฟีเจอร์หลัก",
            "subtitle": "10 กลุ่มจากค่าต่ำไปสูง; เส้นแสดงสัดส่วน target=1 ใน train",
            "type": "line",
            "intent": "trend",
            "question": "ความเสี่ยงเพิ่มขึ้นตามระดับการใช้หน้าจออย่างไร",
            "rationale": "Decile line แสดงความชันและ threshold ของตัวแปรต่อเนื่องได้ตรงไปตรงมา",
            "dataset": "target_deciles",
            "sourceId": "competition-data",
            "encodings": {
                "x": {"field": "decile", "type": "ordinal", "label": "Decile (ค่าต่ำ → สูง)"},
                "y": {"fields": ["daily_screen_time_hours", "weekend_screen_time", "social_media_hours"], "type": "quantitative", "format": "percent", "label": "Target rate"},
            },
            "xAxisTitle": "Decile (ค่าต่ำ → สูง)",
            "yAxisTitle": "Target rate",
            "valueFormat": "percent",
            "layout": "full",
            "maxRows": 10,
        },
        {
            "id": "feature-importance",
            "title": "Permutation importance ของ baseline",
            "subtitle": "AUC ที่ลดลงเมื่อสลับฟีเจอร์; validation sample 30,000 แถว",
            "type": "horizontalBar",
            "intent": "comparison",
            "question": "โมเดลพึ่งพาฟีเจอร์ใดมากที่สุด",
            "rationale": "แถบแนวนอนเหมาะกับการจัดอันดับชื่อฟีเจอร์ที่ยาว",
            "dataset": "importance",
            "sourceId": "analysis-code",
            "encodings": {
                "x": {"field": "feature", "type": "nominal", "label": "Feature"},
                "y": {"field": "importance", "type": "quantitative", "label": "AUC decrease"},
                "tooltip": [
                    {"field": "rank", "type": "quantitative", "label": "Rank"},
                    {"field": "uncertainty", "type": "quantitative", "label": "Std"},
                ],
            },
            "layout": "full",
            "maxRows": 12,
        },
    ],
    "tables": [
        {
            "id": "missingness",
            "title": "Missingness และ cardinality รายฟีเจอร์",
            "subtitle": "ทุกแถวของ train และ test; เรียงตาม missing rate ใน train",
            "dataset": "profile",
            "sourceId": "competition-data",
            "defaultSort": {"field": "train_missing_pct", "direction": "desc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "feature", "label": "Feature", "type": "text"},
                {"field": "type", "label": "Type", "type": "text"},
                {"field": "train_missing_pct", "label": "Train missing (%)", "format": "number"},
                {"field": "test_missing_pct", "label": "Test missing (%)", "format": "number"},
                {"field": "missing_diff_pp", "label": "Diff (pp)", "format": "number", "movement": True},
                {"field": "train_unique", "label": "Train unique", "format": "number"},
            ],
        },
        {
            "id": "drift",
            "title": "Train–test drift รายฟีเจอร์",
            "subtitle": "PSI สำหรับตัวเลข และ total variation distance สำหรับ categorical",
            "dataset": "drift",
            "sourceId": "competition-data",
            "defaultSort": {"field": "drift_score", "direction": "desc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "feature", "label": "Feature", "type": "text"},
                {"field": "kind", "label": "Type", "type": "text"},
                {"field": "drift_score", "label": "Drift score", "format": "number"},
                {"field": "missing_diff_pp", "label": "Missing diff (pp)", "format": "number", "movement": True},
            ],
        },
    ],
    "blocks": [
        {"id": "title", "type": "markdown", "body": "# เจาะลึกข้อมูล Smartphone Addiction", "layout": "full"},
        {"id": "summary", "type": "markdown", "sourceId": "analysis-code", "body": "## Executive Summary\n\n- **โจทย์มีสัญญาณแรงและเหมาะกับ tree boosting:** baseline แบบ HistGradientBoosting บน holdout ได้ ROC AUC **0.9565** โดยยังไม่ได้ tune จริงจัง\n- **สามแกนหลักคือเวลาใช้หน้าจอ:** เวลาใช้หน้าจอรายวัน, เวลาใช้หน้าจอวันหยุด และเวลาใช้โซเชียลมีเดียครองอันดับความสำคัญ และ composite ง่าย ๆ ของสามตัวนี้ได้ AUC **0.9108**\n- **ข้อมูลสะอาดในเชิงโครงสร้าง แต่ missing เยอะ:** มี missing ทุกฟีเจอร์ราว **4.18%–19.38%**; อย่างไรก็ดี missingness แทบไม่เปลี่ยน target rate จึงไม่เห็น leakage ชัดเจน\n- **Train/test ใกล้กันมากในค่าตัวเลข:** PSI ต่ำกว่า **0.0001** ทุกฟีเจอร์ตัวเลข แต่ missing rate บางคอลัมน์ต่างกันถึงราว **3.38 percentage points** จึงต้องใช้ validation และ preprocessing ที่รับ missing ได้"},
        {"id": "headline-metrics", "type": "metric-strip", "cardIds": ["rows", "features", "target", "auc"], "layout": "full"},
        {"id": "drivers", "type": "markdown", "sourceId": "competition-data", "body": "## การใช้หน้าจอสร้างเส้นแบ่งที่ชัดมาก\n\n**อัตรา target เพิ่มแบบเกือบ monotonic ตามระดับการใช้หน้าจอ** โดย decile สูงสุดของเวลาใช้หน้าจอรายวันและวันหยุดมี target rate เกือบ 100% ขณะที่ decile ต่ำสุดอยู่ราว 26% นี่เป็นหลักฐานว่าควรใช้โมเดลที่จับ threshold และ interaction ได้ดี ไม่ใช่พึ่ง linear model อย่างเดียว"},
        {"id": "decile-chart", "type": "chart", "chartId": "target-deciles", "layout": "full"},
        {"id": "importance-copy", "type": "markdown", "sourceId": "analysis-code", "body": "## โมเดลยืนยันว่าฟีเจอร์หลักมีลำดับชัด\n\n**การสลับค่าเวลาใช้หน้าจอรายวันทำให้ AUC ลดลงมากที่สุด** ตามด้วยเวลาใช้หน้าจอวันหยุดและเวลาโซเชียล ส่วน categorical ทั้งสามแทบไม่เพิ่มประสิทธิภาพ baseline เลย อย่างไรก็ตามไม่ควรรีบตัดฟีเจอร์รองก่อนทดสอบ cross-validation เพราะจำนวนการแจ้งเตือนและจำนวนครั้งที่เปิดแอปมีประโยชน์แบบ interaction แม้ univariate AUC จะต่ำ"},
        {"id": "importance-chart-block", "type": "chart", "chartId": "feature-importance", "layout": "full"},
        {"id": "quality-copy", "type": "markdown", "sourceId": "competition-data", "body": "## Missingness สูงแต่ยังจัดการได้\n\n**ข้อมูลไม่มี id ซ้ำ ไม่มีแถวซ้ำ และไม่พบค่าหลุดช่วงกายภาพที่ตรวจ** แต่หลายคอลัมน์หายมากกว่า 10% โดยเฉพาะเวลาโซเชียล (19.38%), gaming (18.34%) และเวลาใช้หน้าจอวันหยุด (16.21%) ค่าเฉลี่ยจำนวนฟีเจอร์ที่หายต่อแถวอยู่ที่ 1.26 ใน train และ 1.27 ใน test จึงควรใช้ native missing handling หรือ imputation ภายในแต่ละ fold"},
        {"id": "missing-table", "type": "table", "tableId": "missingness", "layout": "full"},
        {"id": "drift-copy", "type": "markdown", "sourceId": "competition-data", "body": "## Distribution หลักคงที่ แต่รูปแบบ missing ขยับเล็กน้อย\n\n**ค่าตัวเลข train/test แทบไม่ drift** และ target rate ก็ไม่เปลี่ยนตามช่วง `id` อย่างมีนัยสำคัญเชิงปฏิบัติ จึงไม่มีเหตุผลให้ใช้ `id` เป็นฟีเจอร์ จุดที่ควรเฝ้าระวังคือ missing rate: social media ลดลง 3.38 pp ใน test ขณะที่ app opens ลดลง 3.00 pp และ daily screen time ลดลง 2.80 pp"},
        {"id": "drift-table", "type": "table", "tableId": "drift", "layout": "full"},
        {"id": "recommendations", "type": "markdown", "body": "## ขั้นตอนถัดไปที่แนะนำ\n\n1. ใช้ **5-fold StratifiedKFold** และเก็บ OOF prediction เป็นมาตรฐานกลาง\n2. เริ่มจาก **CatBoost หรือ LightGBM/XGBoost** ที่รองรับ nonlinear interaction; เปรียบเทียบกับ HistGradientBoosting baseline\n3. เก็บ missing indicator และทำ preprocessing ภายใน fold เพื่อป้องกัน leakage\n4. ทดลองฟีเจอร์เชิงพฤติกรรม เช่น weekday–weekend gap, social share และผลรวม screen activity แต่ยอมรับเฉพาะฟีเจอร์ที่เพิ่ม OOF AUC\n5. ทำ seed averaging และ blend เฉพาะโมเดลที่ OOF prediction มีความต่างจริง"},
        {"id": "questions", "type": "markdown", "body": "## คำถามที่ควรทดสอบต่อ\n\n- split แบบสุ่มให้คะแนน optimistic เพราะข้อมูล synthetic มี latent groups หรือไม่?\n- CatBoost ให้ผลดีขึ้นจาก categorical/missing pattern จริงหรือฟีเจอร์เหล่านั้นเป็น noise?\n- ฟีเจอร์ composite เพิ่มคะแนนเหนือ raw tree features หรือเพียงซ้ำสัญญาณเดิม?\n- OOF calibration และ rank averaging ช่วย leaderboard มากน้อยเพียงใด?"},
        {"id": "caveats", "type": "markdown", "sourceId": "analysis-code", "body": "## ข้อจำกัดและสมมติฐาน\n\n- Holdout AUC 0.9565 มาจาก split เดียวและ sample 350,000 แถว ใช้เพื่อยืนยันสัญญาณ ไม่ใช่คะแนน CV ขั้นสุดท้าย\n- ข้อมูลการแข่งขันระบุว่าได้รับแรงบันดาลใจจากชุดข้อมูลอื่นและมีลักษณะ synthetic; ความสัมพันธ์นี้จึงไม่ควรตีความเป็นเหตุและผลในโลกจริง\n- การตรวจ validity ใช้ช่วงกว้างตามข้อจำกัดทางกายภาพ ไม่ได้ยืนยันนิยามหน่วยจากเจ้าของข้อมูลต้นทาง"},
    ],
}

artifact = {
    "surface": "report",
    "manifest": manifest,
    "snapshot": {
        "version": 1,
        "generatedAt": generated_at,
        "status": "ready",
        "datasets": {
            "headline": headline,
            "target_deciles": decile_rows,
            "importance": importance_rows,
            "profile": profile_rows,
            "drift": drift_rows,
        },
    },
    "sources": sources,
}

(REPORT_DIR / "artifact.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
print(REPORT_DIR / "artifact.json")
