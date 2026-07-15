"""Translate ap_lnn_sanitized.csv's Chinese column names to English.

Produces:
    data/public_sanitized/ap_lnn_sanitized_en.csv  -- same data, English headers
    docs/lnn_column_translation.md                 -- Chinese -> English mapping table

Translations are standard clinical-chemistry/hematology/coagulation term
equivalents. Clinical review recommended before relying on any translation
for a specific analyte's exact assay/unit semantics -- see
docs/dataset_sources.md for the dataset's provenance and existing caveats
(e.g. the reversed raw target-label direction).

Usage:
    python scripts/translate_lnn_columns.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from penux_ap.utils import setup_logging

log = setup_logging()

# Chinese column name -> (English full name, standard abbreviation or None)
TRANSLATION = {
    "性别": ("Gender", None),
    "年龄": ("Age", None),
    "白蛋白": ("Albumin", "ALB"),
    "白细胞": ("White Blood Cell count", "WBC"),
    "钙": ("Calcium", "Ca"),
    "甘油三脂": ("Triglyceride", "TG"),
    "甘油三酯": ("Triglyceride", "TG"),
    "谷氨酰转肽酶": ("Gamma-glutamyl Transferase", "GGT"),
    "谷丙转氨酶": ("Alanine Aminotransferase", "ALT"),
    "谷草转氨酶": ("Aspartate Aminotransferase", "AST"),
    "红细胞": ("Red Blood Cell count", "RBC"),
    "肌酐": ("Creatinine", "Cr"),
    "碱性磷酸酶": ("Alkaline Phosphatase", "ALP"),
    "尿素": ("Urea", None),
    "尿酸": ("Uric Acid", "UA"),
    "葡萄糖": ("Glucose", "Glu"),
    "乳酸脱氢酶": ("Lactate Dehydrogenase", "LDH"),
    "血红蛋白": ("Hemoglobin", "Hb"),
    "血小板": ("Platelet count", "PLT"),
    "总胆固醇": ("Total Cholesterol", "TC"),
    "总胆红素": ("Total Bilirubin", "TBIL"),
    "总蛋白": ("Total Protein", "TP"),
    "5′核苷酸酶测定": ("5'-Nucleotidase", "5'-NT"),
    "C反应蛋白": ("C-Reactive Protein", "CRP"),
    "D-二聚体测定": ("D-Dimer", "D-D"),
    "PT活动度": ("Prothrombin Activity", "PT%"),
    "α-L-岩藻糖苷酶": ("Alpha-L-Fucosidase", "AFU"),
    "α-羟丁酸脱氢酶": ("Alpha-Hydroxybutyrate Dehydrogenase", "α-HBDH"),
    "β2微量球蛋白": ("Beta-2 Microglobulin", "β2-MG"),
    "癌胚抗原": ("Carcinoembryonic Antigen", "CEA"),
    "白球比": ("Albumin/Globulin Ratio", "A/G"),
    "标准碳酸氢根": ("Standard Bicarbonate", "SB"),
    "不饱和铁结合力": ("Unsaturated Iron-Binding Capacity", "UIBC"),
    "部分凝血酶原时间": ("Activated Partial Thromboplastin Time", "APTT"),
    "超氧化物歧化酶": ("Superoxide Dismutase", "SOD"),
    "大型血小板比率": ("Platelet-Large Cell Ratio", "P-LCR"),
    "单核细胞百分比": ("Monocyte Percentage", None),
    "单核细胞计数": ("Monocyte Count", None),
    "胆碱酯酶": ("Cholinesterase", "ChE"),
    "低密度脂蛋白": ("Low-Density Lipoprotein Cholesterol", "LDL-C"),
    "淀粉酶": ("Amylase", "AMY"),
    "二氧化碳分压": ("Partial Pressure of Carbon Dioxide", "PaCO2"),
    "二氧化碳结合力": ("Carbon Dioxide Combining Power", "CO2-CP"),
    "二氧化碳总量": ("Total Carbon Dioxide", "TCO2"),
    "高密度脂蛋白": ("High-Density Lipoprotein Cholesterol", "HDL-C"),
    "高铁血红蛋白": ("Methemoglobin", "MetHb"),
    "谷草/谷丙": ("AST/ALT Ratio", "AST/ALT"),
    "胱抑素C": ("Cystatin C", "Cys-C"),
    "国际标准化比值": ("International Normalized Ratio", "INR"),
    "红细胞分布宽度CV": ("Red Cell Distribution Width (CV)", "RDW-CV"),
    "红细胞分布宽度SD": ("Red Cell Distribution Width (SD)", "RDW-SD"),
    "红细胞平均体积": ("Mean Corpuscular Volume", "MCV"),
    "红细胞压积": ("Hematocrit", "HCT"),
    "肌酸激酶": ("Creatine Kinase", "CK"),
    "肌酸酶同工酶": ("Creatine Kinase-MB", "CK-MB"),
    "钾": ("Potassium", "K"),
    "间接胆红素": ("Indirect Bilirubin", "IBIL"),
    "离子钙": ("Ionized Calcium", "iCa"),
    "淋巴细胞百分比": ("Lymphocyte Percentage", None),
    "淋巴细胞计数": ("Lymphocyte Count", None),
    "磷": ("Phosphorus", "P"),
    "氯": ("Chlorine", "Cl"),
    "氯化物": ("Chloride", "Cl-"),
    "镁": ("Magnesium", "Mg"),
    "钠": ("Sodium", "Na"),
    "凝血酶时间": ("Thrombin Time", "TT"),
    "凝血酶原时间": ("Prothrombin Time", "PT"),
    "平均血红蛋白量": ("Mean Corpuscular Hemoglobin", "MCH"),
    "平均血红蛋白浓度": ("Mean Corpuscular Hemoglobin Concentration", "MCHC"),
    "平均血小板体积": ("Mean Platelet Volume", "MPV"),
    "前白蛋白": ("Prealbumin", "PA"),
    "球蛋白": ("Globulin", "GLB"),
    "乳酸": ("Lactate", "Lac"),
    "肾小球滤过率": ("Glomerular Filtration Rate", "GFR"),
    "剩余碱": ("Base Excess", "BE"),
    "实际碳酸氢根": ("Actual Bicarbonate", "AB"),
    "视黄醇结合蛋白": ("Retinol-Binding Protein", "RBP"),
    "嗜碱性粒细胞百分比": ("Basophil Percentage", None),
    "嗜碱性粒细胞计数": ("Basophil Count", None),
    "嗜酸性粒细胞百分比": ("Eosinophil Percentage", None),
    "嗜酸性粒细胞计数": ("Eosinophil Count", None),
    "酸碱度": ("pH", None),
    "碳氧血红蛋白": ("Carboxyhemoglobin", "COHb"),
    "铁": ("Iron", "Fe"),
    "脱氧血红蛋白": ("Deoxyhemoglobin", "HHb"),
    "细胞外碱超": ("Extracellular Base Excess", "BEecf"),
    "纤维蛋白降解产物": ("Fibrin Degradation Products", "FDP"),
    "纤维蛋白原": ("Fibrinogen", "FIB"),
    "腺苷脱氨酶": ("Adenosine Deaminase", "ADA"),
    "血清5′核苷酸酶测定": ("Serum 5'-Nucleotidase", "5'-NT"),
    "血细胞比容": ("Hematocrit", "HCT"),
    "血小板分布宽度": ("Platelet Distribution Width", "PDW"),
    "血小板压积": ("Plateletcrit", "PCT"),
    "氧饱和度": ("Oxygen Saturation", "SO2"),
    "氧分压": ("Partial Pressure of Oxygen", "PaO2"),
    "氧合血红蛋白": ("Oxyhemoglobin", "O2Hb"),
    "载脂蛋白A1": ("Apolipoprotein A1", "ApoA1"),
    "载脂蛋白B": ("Apolipoprotein B", "ApoB"),
    "脂蛋白(a)": ("Lipoprotein(a)", "Lp(a)"),
    "脂肪酶": ("Lipase", None),
    "直接胆红素": ("Direct Bilirubin", "DBIL"),
    "中性粒细胞百分比": ("Neutrophil Percentage", None),
    "中性粒细胞计数": ("Neutrophil Count", None),
    "总胆汁酸": ("Total Bile Acid", "TBA"),
    "总铁结合力": ("Total Iron-Binding Capacity", "TIBC"),
    "总血红蛋白浓度": ("Total Hemoglobin Concentration", "tHb"),
    "严重程度": ("Severity (target column; raw 0=SAP, raw 1=non-SAP -- see docs/dataset_sources.md)", None),
}


def english_header(chinese_name: str) -> str:
    if chinese_name not in TRANSLATION:
        raise KeyError(f"No translation registered for column: {chinese_name!r}")
    full, abbr = TRANSLATION[chinese_name]
    return f"{abbr} ({full})" if abbr else full


def main():
    root = Path(__file__).resolve().parents[1]
    src_path = root / "data/public_sanitized/ap_lnn_sanitized.csv"
    out_csv = root / "data/public_sanitized/ap_lnn_sanitized_en.csv"
    out_md = root / "docs/lnn_column_translation.md"

    df = pd.read_csv(src_path)
    missing = [c for c in df.columns if c not in TRANSLATION]
    if missing:
        raise SystemExit(f"Missing translations for columns: {missing}")

    df_en = df.rename(columns={c: english_header(c) for c in df.columns})
    df_en.to_csv(out_csv, index=False)
    log.info("Wrote English-header CSV to %s", out_csv)

    lines = [
        "# ap_lnn_sanitized.csv: Chinese to English column translation\n",
        "Reference mapping for `ap_lnn_sanitized.csv`'s 107 Chinese column headers. "
        "Standard clinical-chemistry/hematology/coagulation term equivalents -- "
        "clinical review recommended before relying on any translation for a "
        "specific analyte's exact assay/unit semantics. See `docs/dataset_sources.md` "
        "for the dataset's provenance and the reversed-label caveat on the target column.\n",
        "An English-header copy of the full dataset is at `data/public_sanitized/ap_lnn_sanitized_en.csv` "
        "(generated by `scripts/translate_lnn_columns.py`).\n",
        "| # | Chinese | English | Abbreviation |",
        "|---|---|---|---|",
    ]
    for i, col in enumerate(df.columns, start=1):
        full, abbr = TRANSLATION[col]
        lines.append(f"| {i} | {col} | {full} | {abbr or '-'} |")

    out_md.write_text("\n".join(lines) + "\n")
    log.info("Wrote translation table to %s", out_md)


if __name__ == "__main__":
    main()
