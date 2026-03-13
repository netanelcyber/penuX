# 🦠 PATHOGEN IDENTIFICATION FOR SEPSIS - Clinical Decision Guide

## Overview
Identify likely causative organism from vital signs, laboratory markers, and clinical presentation.

---

## 10 PATHOGEN CLASSES (MIMIC-III)

| ID | Pathogen | Gram Stain | Typical Features | Mortality | First-Line Antibiotic |
|----|----------|-----------|------------------|-----------|----------------------|
| **0** | **Staph aureus / MRSA** | **Gram+** | Skin/soft tissue, moderate fever (38-39°C), WBC 12-18K, age 50-70 | 25% | Vancomycin, Doxycycline |
| **1** | **E. coli** | **Gram−** | UTI, intra-abdominal, HIGH fever (38-40°C), HIGH WBC (14-22K), age 40-75 | 15% | Cephalosporin, Carbapenems |
| **2** | **Klebsiella pneumoniae** | **Gram−** | Pneumonia, intra-abdominal, fever (38-40°C), HIGH WBC (15-25K), OLDER patients (55-80) | 20% | Cephalosporin, Carbapenems |
| **3** | **Acinetobacter baumannii** | **Gram−** | Hospital-acquired, LOW fever (37-38°C), wound/respiratory, ICU elderly (60-85) | **35%** | Colistin, Carbapenems |
| **4** | **Pseudomonas aeruginosa** | **Gram−** | Respiratory (VAP), hospital-acquired, variable signs, ICU patients (55-75) | 30% | Antipseudomonal beta-lactam |
| **5** | **Streptococcus species** | **Gram+** | Endocarditis, bacteremia, meningitis, HIGH fever (38-40°C), HIGH WBC (13-20K), age 50-70 | 20% | Penicillin, Cephalosporin |
| **6** | **Enterococcus species** | **Gram+** | UTI, endocarditis, intra-abdominal, LOW fever (37-39°C), WBC 11-16K, OLDER (65-85) | 25% | Ampicillin, Vancomycin |
| **7** | **Candida / Fungal** | **Eukaryotic** | Catheter-related, nosocomial, LOW fever (37-38°C), **LOW/NORMAL WBC** (8-14K), immunocompromised | **40%** | Fluconazole, Caspofungin |
| **8** | **Viral** | **Non-bacterial** | Respiratory, influenza, COVID-19, variable fever (37-39°C), **LOW WBC** (8-12K), any age | 10% | Antivirals (supportive) |
| **9** | **Other/Mixed/Anaerobic** | **Mixed** | Intra-abdominal, polymicrobial, fever (38-39°C), HIGH WBC (11-20K), age 50-75 | 30% | Broad-spectrum |

---

## CLINICAL DECISION TREE

### Step 1: Assess Temperature & WBC Pattern

```
HIGH FEVER (≥39°C) + HIGH WBC (≥15K)
  ├─ Most likely: E. coli, Klebsiella, Streptococcus
  └─ Common source: UTI, intra-abdominal, pneumonia, endocarditis

MODERATE FEVER (38-38.5°C) + MODERATE WBC (12-15K)
  ├─ Most likely: Staph aureus, Pseudomonas, Enterococcus
  └─ Common source: Skin/wound, respiratory (VAP), urinary

LOW FEVER (<38°C) or NO FEVER + LOW-NORMAL WBC (<12K)
  ├─ Most likely: Fungal (Candida), Viral, Acinetobacter
  └─ Common source: Catheter, hospital-acquired, respiratory

NORMAL TEMP (36.5-37.5°C) + NORMAL/LOW WBC (<10K)
  ├─ Most likely: Viral, Fungal
  └─ Consider: Immunosuppression, chronic illness
```

### Step 2: Assess Oxygenation & Respiratory Status

```
SpO₂ < 92% OR RR ≥ 22 (Tachypnea)
  ├─ Likely respiratory pathogen:
  ├─ Klebsiella pneumoniae (pneumonia)
  ├─ Pseudomonas aeruginosa (VAP)
  ├─ Viral (bronchiolitis)
  └─ Consider: Sepsis-induced ARDS

SpO₂ ≥ 95% + Normal RR (15-20)
  ├─ Non-respiratory source likely
  ├─ E. coli (UTI), Staph (skin), Enterococcus (catheter)
```

### Step 3: Assess Hemodynamics (BP & Shock Signs)

```
MAP < 65 mmHg OR SBP < 90 mmHg (Septic Shock)
  ├─ Risk for MORTALITY: 40-60%
  ├─ More common with Gram-negative organisms:
  │   - E. coli, Klebsiella, Pseudomonas, Acinetobacter
  ├─ Urgent need for: Vasopressors, broad-spectrum antibiotics
  └─ Consider source control (drainage, debridement)

Normal BP (SBP 110-130, MAP 75-90)
  ├─ Sepsis without shock
  ├─ More time for targeted therapy
  └─ Can consider narrower-spectrum empiric coverage
```

### Step 4: Assess Patient Risk Factors & ICU Status

```
AGE ≥ 65 years
  ├─ Higher risk: Acinetobacter, Klebsiella, Enterococcus
  ├─ Higher mortality
  └─ Organ dysfunction likely

ICU LENGTH OF STAY > 3 days
  ├─ Higher risk: Hospital-acquired organisms
  │   - Pseudomonas, Acinetobacter, Candida
  ├─ Consider: Multi-drug resistance
  └─ Likely resistant to standard agents

IMMUNOCOMPROMISED (transplant, cancer, HIV)
  ├─ Higher risk: Fungal (Candida, Aspergillus)
  ├─ Atypical organisms more common
  └─ May have minimal inflammatory response
```

---

## QUANTITATIVE SCORING SYSTEM

**Feature Importance Weights (from PenuX model):**

| Feature | Weight | Rationale |
|---------|--------|-----------|
| Age | 0.213 | Strong predictor of pathogen type & severity |
| WBC | 0.202 | Discriminates bacterial vs. viral vs. fungal |
| Temperature | 0.142 | Moderate correlation with organism class |
| SpO₂ | 0.159 | Indicates respiratory involvement |
| MAP | Derived | Critical for risk stratification |
| Pulse Pressure | 0.314 | High variability = discriminative power |

---

## CLINICAL EXAMPLES

### Example 1: Community-Acquired Pneumonia
- **Patient:** 68-year-old male
- **Vitals:** T=39.2°C, RR=24, SpO₂=89%, SBP=102, WBC=18K
- **Likely:** Klebsiella pneumoniae or Streptococcus
- **Empiric Rx:** Cephalosporin 3rd-gen or Respiratory FQ

### Example 2: Urinary Tract Infection → Sepsis
- **Patient:** 55-year-old female, catheterized
- **Vitals:** T=38.8°C, RR=18, SpO₂=97%, SBP=95, WBC=17K
- **Likely:** E. coli (most common UTI pathogen)
- **Empiric Rx:** Fluoroquinolone or Cephalosporin

### Example 3: Healthcare-Associated Infection
- **Patient:** 72-year-old, ICU day 5, post-operative
- **Vitals:** T=37.4°C, RR=22, SpO₂=91%, SBP=88 (MAP=62), WBC=13K
- **Likely:** Acinetobacter or Pseudomonas
- **Empiric Rx:** Antipseudomonal beta-lactam + Fluoroquinolone or Colistin

### Example 4: Immunocompromised Host
- **Patient:** 65-year-old, history of cancer, post-chemotherapy
- **Vitals:** T=37.8°C, RR=20, SpO₂=92%, SBP=105, WBC=4K (LOW!)
- **Likely:** Candida (fungal) or atypical pathogen
- **Empiric Rx:** Fluconazole + broad-spectrum antibacterial

### Example 5: Respiratory Viral Infection
- **Patient:** 42-year-old, no immunosuppression
- **Vitals:** T=38.2°C, RR=19, SpO₂=95%, SBP=118, WBC=7.5K (LOW)
- **Likely:** Viral (influenza, COVID-19, RSV)
- **Treatment:** Supportive care + antiviral (oseltamivir, remdesivir)

---

## CRITICAL THRESHOLDS FOR ACTION

| Marker | Threshold | Action |
|--------|-----------|--------|
| **MAP** | < 65 mmHg | Septic shock → Vasopressors + ICU |
| **Lactate** | > 2 mmol/L | Tissue hypoperfusion → Aggressive resuscitation |
| **SpO₂** | < 90% | Respiratory failure → O₂, consider intubation |
| **WBC** | < 4K or > 30K | Severe infection/immunocompromise → Urgent intervention |
| **Temperature** | < 36°C | Poor prognosis sign, increased mortality |
| **RR** | ≥ 30 | Severe respiratory distress → Airway management |

---

## LIMITATIONS & CAVEATS

⚠️ **This guide is for CLINICAL SUPPORT ONLY, not definitive diagnosis**

- **Pathogen identification requires:**
  - Blood cultures (gold standard, 48-72h turnaround)
  - Urinalysis/urine culture
  - Respiratory specimens if applicable
  - Imaging (CXR, ultrasound, CT)

- **Sepsis often involves MULTIPLE organisms** (polymicrobial infections)

- **Empiric antibiotic therapy should cover:**
  - Most likely pathogens
  - Local resistance patterns
  - Patient risk factors
  - Source of infection

- **Narrow therapy once cultures return** (stewardship)

---

## NEXT STEPS IN PenuX

1. **Train deep learning model** on full MIMIC-III cohort (n>10,000)
2. **Add ECG features** (if available) for additional discrimination
3. **Integrate LASSO/permutation importance** for feature selection
4. **Validate on external dataset** (external MIMIC-IV, other hospitals)
5. **Implement in clinical EHR** with real-time predictions & alerts

---

Generated: March 2025
Part of PenuX Clinical AI Platform
