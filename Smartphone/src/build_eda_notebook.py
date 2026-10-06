"""Build the reader-facing EDA notebook using nbformat."""

from pathlib import Path

import nbformat as nbf


root = Path(__file__).resolve().parents[1]
notebook_path = root / "notebooks" / "01_smartphone_addiction_deep_dive.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb["metadata"]["language_info"] = {"name": "python", "version": "3"}

nb["cells"] = [
    nbf.v4.new_markdown_cell(
        "# เจาะลึกข้อมูล Predicting Smartphone Addiction\n\n"
        "## tl;dr\n\n"
        "Notebook นี้ตรวจคุณภาพข้อมูล, missingness, train–test drift, ความสัมพันธ์กับ target "
        "และสร้างโมเดล holdout ขนาดจำกัดเพื่อยืนยันสัญญาณหลัก ผลสรุปเชิงตัวเลขถูกสร้างจากเซลล์ด้านล่างโดยตรง"
    ),
    nbf.v4.new_markdown_cell(
        "## Context & Methods\n\n"
        "### Key Assumptions\n\n"
        "- หนึ่งแถวแทนหนึ่งตัวอย่างอิสระ และ `id` เป็นเพียงตัวระบุ\n"
        "- ROC AUC เป็น metric หลัก จึงประเมินความสามารถในการจัดอันดับความเสี่ยง\n"
        "- ผล holdout เป็น diagnostic baseline ไม่ใช่คะแนน cross-validation ขั้นสุดท้าย"
    ),
    nbf.v4.new_code_cell(
        "from pathlib import Path\n"
        "import sys\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n\n"
        "ROOT = Path.cwd().resolve().parent if Path.cwd().name == 'notebooks' else Path.cwd().resolve()\n"
        "sys.path.insert(0, str(ROOT))\n"
        "from src.deep_dive import run_analysis\n\n"
        "result = run_analysis()\n"
        "quality = result['summary']['quality']\n"
        "tables = result['tables']\n"
        "quality"
    ),
    nbf.v4.new_markdown_cell("## Data\n\n### โครงสร้างและ missingness"),
    nbf.v4.new_code_cell(
        "display(tables['profile'].round(3))\n"
        "display(tables['validity'].query('invalid_count > 0'))"
    ),
    nbf.v4.new_markdown_cell("## Results\n\n### สัญญาณตัวเลขที่สัมพันธ์กับ target"),
    nbf.v4.new_code_cell(
        "signal = tables['numeric_signal'].copy()\n"
        "display(signal.round(4))\n"
        "ax = signal.sort_values('univariate_auc_strength').plot.barh(\n"
        "    x='feature', y='univariate_auc_strength', legend=False, figsize=(9, 5), color='#2F6BFF')\n"
        "ax.axvline(0.5, color='#3D4652', linestyle='--', linewidth=1)\n"
        "ax.set(title='Univariate ROC AUC strength', xlabel='AUC (ปรับทิศทางให้ ≥ 0.5)', ylabel='')\n"
        "plt.tight_layout(); plt.show()"
    ),
    nbf.v4.new_markdown_cell("### Target rate ตาม decile ของตัวแปรหลัก"),
    nbf.v4.new_code_cell(
        "top = tables['binned_target'].query(\n"
        "    \"feature in ['daily_screen_time_hours', 'weekend_screen_time', 'social_media_hours']\"\n"
        ")\n"
        "pivot = top.pivot(index='decile', columns='feature', values='target_rate')\n"
        "display(pivot.round(4))\n"
        "ax = pivot.plot(figsize=(9, 5), marker='o', color=['#2F6BFF', '#D28E00', '#C45D33'])\n"
        "ax.set(title='Addiction rate by feature decile', xlabel='Decile (ค่าต่ำ → สูง)', ylabel='Target rate', ylim=(0, 1.05))\n"
        "plt.tight_layout(); plt.show()"
    ),
    nbf.v4.new_markdown_cell("### Train–test drift และ missingness"),
    nbf.v4.new_code_cell(
        "display(tables['drift'].round(5))\n"
        "display(tables['missing_pattern'].round(4))"
    ),
    nbf.v4.new_markdown_cell("### ความสำคัญของฟีเจอร์จากโมเดล holdout"),
    nbf.v4.new_code_cell(
        "importance = tables['permutation_importance'].sort_values('importance_mean')\n"
        "display(importance.sort_values('importance_mean', ascending=False).round(5))\n"
        "ax = importance.plot.barh(x='feature', y='importance_mean', legend=False, figsize=(9, 5), color='#2F6BFF')\n"
        "ax.axvline(0, color='#3D4652', linewidth=1)\n"
        "ax.set(title=f\"Permutation importance — holdout AUC {quality['holdout_auc_histgb']:.4f}\", xlabel='AUC decrease', ylabel='')\n"
        "plt.tight_layout(); plt.show()"
    ),
    nbf.v4.new_markdown_cell(
        "## Takeaways\n\n"
        "- ตัวแปรหลักคือเวลาใช้หน้าจอรายวัน, เวลาใช้หน้าจอวันหยุด และเวลา social media\n"
        "- missingness ค่อนข้างสูงแต่แทบไม่เป็นสัญญาณของ target จึงควรเก็บ missing indicator ไว้และปล่อยให้โมเดลจัดการ\n"
        "- distribution ของค่าตัวเลข train/test ใกล้กันมาก แต่ missing rate บางคอลัมน์ต่างกันหลาย percentage points\n"
        "- `id` ต่อเนื่องตามการแบ่ง train/test และ target rate คงที่ตามช่วง id; ไม่ควรใช้ `id` เป็นฟีเจอร์\n"
        "- baseline แบบ tree ให้ holdout AUC สูง จึงควรต่อยอดด้วย stratified CV และโมเดล boosting ที่รองรับ missing/categorical"
    ),
]

nbf.write(nb, notebook_path)
print(notebook_path)
