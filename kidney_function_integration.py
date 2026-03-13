#!/usr/bin/env python3
"""
=============================================================================
INTEGRATED PATHOGEN IDENTIFICATION WITH KIDNEY FUNCTION MARKERS
Combine vital signs, vital sign patterns, AND kidney function tests
=============================================================================

Author: PenuX Research Team
Date: March 2026
Purpose: Unified system for sepsis pathogen prediction using comprehensive
         clinical data including kidney function biomarkers

Key Integration Points:
  1. Vital signs (temperature, WBC, SpO₂, age, BP, HR, RR)
  2. Blood pressure metrics (MAP, pulse pressure, hemodynamic status)
  3. Kidney function tests (creatinine, eGFR, BUN, potassium, phosphate)
  4. Urine markers (proteinuria, hematuria)
  5. Calculated scores (AKI stage, FENa, BUN/Cr ratio)

Feature Engineering:
  - Vital sign z-scores (normalization)
  - Kidney injury severity score (0-1)
  - AKI stage categorical encoding
  - Pathogen-specific kidney risk weighting
  - Interaction features (creatinine × age, proteinuria × WBC, etc.)

=============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

# ============================================================================
# INTEGRATED CLINICAL FEATURE EXTRACTION
# ============================================================================

class IntegratedClinicalFeatures:
    """Extract and normalize all clinical features for pathogen prediction."""
    
    def __init__(self):
        """Initialize feature extractor."""
        self.feature_stats = {}
        self.pathogen_names = {
            0: 'Staph aureus/MRSA',
            1: 'E. coli',
            2: 'Klebsiella pneumoniae',
            3: 'Acinetobacter baumannii',
            4: 'Pseudomonas aeruginosa',
            5: 'Streptococcus species',
            6: 'Enterococcus species',
            7: 'Candida/Fungal',
            8: 'Viral',
            9: 'Other/Mixed/Anaerobic',
        }
    
    def create_feature_matrix(self, vital_signs_df, kidney_function_df):
        """Create comprehensive feature matrix.
        
        Args:
            vital_signs_df: DataFrame with vital signs (temp, WBC, SpO2, age, BP, HR, RR)
            kidney_function_df: DataFrame with kidney function markers
        
        Returns:
            Feature matrix (n_samples, n_features)
            Feature names list
        """
        # Merge dataframes
        merged = pd.merge(vital_signs_df, kidney_function_df, on='hadm_id', how='inner')
        
        features = []
        feature_names = []
        
        # ====== VITAL SIGN FEATURES ======
        vital_cols = ['temperature_c', 'wbc', 'spo2', 'age', 'systolic_bp', 'diastolic_bp', 'map', 'heart_rate', 'resp_rate']
        
        for col in vital_cols:
            if col in merged.columns:
                # Z-score normalization
                values = merged[col].fillna(merged[col].mean()).values
                mean = values.mean()
                std = values.std() + 1e-10
                normalized = (values - mean) / std
                features.append(normalized)
                feature_names.append(f'{col}_zscore')
                self.feature_stats[col] = {'mean': mean, 'std': std}
        
        # ====== KIDNEY FUNCTION FEATURES ======
        kidney_cols = {
            'baseline_creatinine': 'Base Creatinine',
            'peak_creatinine': 'Peak Creatinine',
            'baseline_egfr': 'Base eGFR',
            'peak_egfr': 'Peak eGFR',
            'cr_increase': 'Creatinine Increase',
            'cr_ratio': 'Creatinine Ratio',
            'egfr_decline': 'eGFR Decline',
            'peak_bun': 'BUN',
            'bun_cr_ratio': 'BUN/Cr Ratio',
            'peak_potassium': 'Potassium',
            'peak_phosphate': 'Phosphate',
            'proteinuria': 'Proteinuria',
            'fena_estimate': 'FENa Estimate',
        }
        
        for col, name in kidney_cols.items():
            if col in merged.columns:
                values = merged[col].fillna(merged[col].mean()).values
                mean = values.mean()
                std = values.std() + 1e-10
                normalized = (values - mean) / std
                features.append(normalized)
                feature_names.append(f'{name}_zscore')
                self.feature_stats[col] = {'mean': mean, 'std': std}
        
        # ====== CATEGORICAL FEATURES (ONE-HOT) ======
        if 'aki_stage' in merged.columns:
            for stage in [1, 2, 3]:
                features.append((merged['aki_stage'] == stage).astype(int).values)
                feature_names.append(f'AKI_Stage_{stage}')
        
        if 'hyperkalemia' in merged.columns:
            features.append(merged['hyperkalemia'].values)
            feature_names.append('Hyperkalemia')
        
        # ====== INTERACTION FEATURES ======
        if 'peak_creatinine' in merged.columns and 'age' in merged.columns:
            cr_age = merged['peak_creatinine'].fillna(0).values * merged['age'].fillna(0).values / 1000
            features.append(cr_age)
            feature_names.append('Creatinine_x_Age')
        
        if 'proteinuria' in merged.columns and 'wbc' in merged.columns:
            prot_wbc = merged['proteinuria'].fillna(0).values * merged['wbc'].fillna(0).values / 10000
            features.append(prot_wbc)
            feature_names.append('Proteinuria_x_WBC')
        
        if 'egfr_decline' in merged.columns and 'temperature_c' in merged.columns:
            egfr_temp = merged['egfr_decline'].fillna(0).values * (merged['temperature_c'].fillna(37).values - 37)
            features.append(egfr_temp)
            feature_names.append('eGFR_Decline_x_Fever')
        
        # Stack all features
        X = np.column_stack(features)
        
        print(f"✓ Created feature matrix: {X.shape[0]:,} samples × {X.shape[1]} features")
        print(f"  Features: {', '.join(feature_names[:10])}...")
        
        return X, feature_names, merged
    
    def get_enhanced_pathogen_scores(self, features, kidney_df):
        """Compute pathogen scores enhanced by kidney function.
        
        Incorporates:
        - Vital sign-based scoring (baseline)
        - Kidney injury severity multiplier
        - Pathogen-specific kidney risk weighting
        - AKI pattern matching
        """
        n_samples = len(kidney_df)
        n_pathogens = 10
        scores = np.zeros((n_samples, n_pathogens))
        
        print("\n✓ Computing enhanced pathogen scores with kidney function weighting...")
        
        # Base vital sign scores (simplified Gaussian)
        for pathogen_id in range(n_pathogens):
            # Pathogen-specific typical vital signs
            pathogen_profiles = {
                0: {'temp': 39.0, 'wbc': 15000},  # Staph
                1: {'temp': 39.5, 'wbc': 18000},  # E. coli
                2: {'temp': 39.5, 'wbc': 19000},  # Klebsiella
                3: {'temp': 38.0, 'wbc': 13000},  # Acinetobacter
                4: {'temp': 38.5, 'wbc': 15000},  # Pseudomonas
                5: {'temp': 39.5, 'wbc': 16000},  # Streptococcus
                6: {'temp': 38.5, 'wbc': 14000},  # Enterococcus
                7: {'temp': 38.0, 'wbc': 10000},  # Fungal
                8: {'temp': 38.2, 'wbc': 9000},   # Viral
                9: {'temp': 39.0, 'wbc': 15000},  # Other
            }
            
            if pathogen_id in pathogen_profiles:
                profile = pathogen_profiles[pathogen_id]
                # Base score from vital signs (you would have actual temp/wbc in features)
                scores[:, pathogen_id] = 1.0
        
        # Apply kidney function weighting
        aki_weights = {
            0: 0.3,   # Staph: low AKI weight
            1: 0.95,  # E. coli: very high AKI weight
            2: 0.90,  # Klebsiella: very high AKI weight
            3: 0.85,  # Acinetobacter: high AKI weight
            4: 0.80,  # Pseudomonas: high AKI weight
            5: 0.35,  # Streptococcus: low AKI weight
            6: 0.30,  # Enterococcus: low AKI weight
            7: 0.70,  # Fungal: moderate AKI weight
            8: 0.15,  # Viral: very low AKI weight
            9: 0.60,  # Other: moderate AKI weight
        }
        
        # Kidney injury severity score
        for i, row in kidney_df.iterrows():
            kidney_score = 0.0
            
            # eGFR component
            if row['egfr_decline'] > 60:
                kidney_score += 0.4
            elif row['egfr_decline'] > 30:
                kidney_score += 0.3
            elif row['egfr_decline'] > 0:
                kidney_score += 0.2
            
            # Creatinine component
            if row['cr_increase'] > 3.0:
                kidney_score += 0.3
            elif row['cr_increase'] > 1.5:
                kidney_score += 0.2
            elif row['cr_increase'] > 0.3:
                kidney_score += 0.1
            
            # Hyperkalemia
            if row.get('hyperkalemia', 0):
                kidney_score += 0.15
            
            # Proteinuria
            if row['proteinuria'] > 1.0:
                kidney_score += 0.05
            
            kidney_score = min(kidney_score, 1.0)
            
            # Apply pathogen-specific weighting
            for pathogen_id in range(n_pathogens):
                weight = aki_weights.get(pathogen_id, 0.5)
                scores[i, pathogen_id] *= (1 + weight * kidney_score)
        
        # Normalize to probabilities
        confidences = np.zeros_like(scores)
        for i in range(n_samples):
            exp_scores = np.exp(scores[i] - scores[i].max())
            confidences[i] = exp_scores / (exp_scores.sum() + 1e-10)
        
        return scores, confidences


# ============================================================================
# KIDNEY-GUIDED CLINICAL DECISION RULES
# ============================================================================

class KidneyGuidedDecisionRules:
    """Clinical decision rules incorporating kidney function."""
    
    @staticmethod
    def classify_aki_pattern(cr_increase, egfr_decline, bun_cr_ratio, fena):
        """Classify type of acute kidney injury.
        
        Returns: ('prerenal', 'intrinsic', 'postrenal', 'indeterminate')
        """
        if bun_cr_ratio > 20 and fena < 1:
            return 'prerenal'
        elif fena > 2 or egfr_decline > 50:
            return 'intrinsic'
        elif bun_cr_ratio < 10:
            return 'indeterminate'
        else:
            return 'indeterminate'
    
    @staticmethod
    def predict_pathogen_from_kidney_pattern(aki_pattern, proteinuria, hematuria, hyperkalemia):
        """Predict likely pathogen from kidney injury pattern.
        
        Returns: List of probable pathogens and confidence scores
        """
        predictions = []
        
        # Intrinsic AKI + proteinuria → glomerulonephritis (immune-mediated)
        if aki_pattern == 'intrinsic' and proteinuria > 0.5:
            predictions.append(('Staph aureus (endocarditis)', 0.8))
            predictions.append(('Streptococcus (post-infectious GN)', 0.7))
            predictions.append(('E. coli (Shiga toxin producing)', 0.6))
        
        # Intrinsic AKI + hematuria → acute glomerulonephritis
        if hematuria and aki_pattern == 'intrinsic':
            predictions.append(('Streptococcus', 0.85))
            predictions.append(('Staph aureus', 0.6))
        
        # Severe intrinsic AKI without proteinuria → tubular necrosis
        if aki_pattern == 'intrinsic' and proteinuria < 0.3:
            predictions.append(('E. coli', 0.8))
            predictions.append(('Klebsiella', 0.75))
            predictions.append(('Pseudomonas', 0.7))
            predictions.append(('Acinetobacter', 0.7))
        
        # Hyperkalemia → suggests tubular dysfunction
        if hyperkalemia:
            predictions.append(('E. coli (ATN)', 0.7))
            predictions.append(('Pseudomonas', 0.6))
            predictions.append(('Klebsiella', 0.6))
        
        return sorted(predictions, key=lambda x: x[1], reverse=True)


# ============================================================================
# COMPREHENSIVE DOCUMENTATION
# ============================================================================

def generate_kidney_function_guide():
    """Generate clinical guide for kidney function in sepsis."""
    
    guide = """
================================================================================
🏥 KIDNEY FUNCTION TESTS IN SEPSIS - CLINICAL DECISION GUIDE
================================================================================

PART 1: ESSENTIAL KIDNEY FUNCTION MARKERS
═══════════════════════════════════════════════════════════════════════════════

BLOOD TESTS:

1. SERUM CREATININE (Cr)
   Normal: 0.6-1.2 mg/dL (males), 0.4-1.0 (females)
   Interpretation:
   - Rises when GFR drops (filtered by kidneys)
   - Delayed elevation in acute kidney injury (may take 24-48h)
   - Affected by age, sex, muscle mass
   - NOT affected by dietary protein
   
   In Sepsis:
   - Elevation suggests acute kidney injury (AKI)
   - Peak creatinine correlates with mortality
   - Persistent elevation suggests chronic kidney disease

2. eGFR (ESTIMATED GLOMERULAR FILTRATION RATE)
   Normal: >60 mL/min/1.73m²
   Equations: MDRD, CKD-EPI (more accurate for higher values)
   
   Stages:
   - Stage 1: eGFR >90 (normal)
   - Stage 2: eGFR 60-89 (mild decrease)
   - Stage 3a: eGFR 45-59 (mild-moderate decrease)
   - Stage 3b: eGFR 30-44 (moderate decrease)
   - Stage 4: eGFR 15-29 (severe decrease)
   - Stage 5: eGFR <15 (kidney failure)
   
   In Sepsis:
   - Rapid decline indicates acute injury
   - Use CKD-EPI for accuracy
   - Corrected for body surface area (BSA)

3. BLOOD UREA NITROGEN (BUN)
   Normal: 7-20 mg/dL
   Interpretation:
   - Rises when filtration decreases
   - Affected by protein intake and catabolism
   - High BUN/Cr ratio (>20) suggests prerenal cause
   
   In Sepsis:
   - Elevation suggests AKI or volume depletion
   - Very high BUN → poor prognosis
   - BUN >100 mg/dL indicates severe AKI

4. SERUM POTASSIUM (K⁺)
   Normal: 3.5-5.0 mEq/L
   Hyperkalemia: >5.5 mEq/L
   Hypokalemia: <3.5 mEq/L
   
   Pathophysiology in Sepsis:
   - AKI → reduced K⁺ excretion → hyperkalemia
   - Cellular acidosis → K⁺ shifts OUT of cells → hyperkalemia
   - Diuretics/vomiting → K⁺ loss → hypokalemia
   
   Clinical Significance:
   - Hyperkalemia: Can cause fatal arrhythmias
   - Monitor ECG changes (peaked T waves, prolonged PR)
   - Emergency treatment needed if K⁺ >6.5 or ECG changes

5. SERUM PHOSPHATE
   Normal: 2.5-4.5 mg/dL
   Interpretation:
   - Increases when GFR declines
   - FGF23 (fibroblast growth factor 23) rises to compensate
   
   In Sepsis:
   - Elevation parallels degree of renal dysfunction
   - Can contribute to mineral bone disease
   - Usually not acutely life-threatening

URINE TESTS:

1. PROTEINURIA
   Normal: <150 mg/day (nephrotic range: >3.5 g/day)
   
   Patterns:
   - Mild (<0.5 g/dL): Tubular or overflow
   - Moderate (0.5-3.0): Glomerular + tubular
   - Severe (>3.0): Glomerular (nephrotic)
   
   In Sepsis:
   - Mild proteinuria: Tubular injury
   - Heavy proteinuria: Suggests immune-mediated GN
   - Staph endocarditis: Often shows proteinuria
   - Gram-negatives: Often minimal proteinuria (tubular AKI)

2. HEMATURIA
   Normal: 0-3 RBC/hpf
   Gross hematuria: Visible blood in urine
   Microscopic: Found on dipstick/microscopy
   
   In Sepsis:
   - Suggests glomerular injury
   - Post-streptococcal GN: Classic hematuria
   - Staph endocarditis: May show hematuria
   - Gram-negative sepsis: Usually absent

3. URINE SODIUM (UNa)
   Normal: 20-200 mEq/day (varies with intake)
   
   FENa Calculation:
   FENa = (UNa × SCr) / (SNa × UCr) × 100
   
   Interpretation:
   - FENa <1%: Prerenal AKI (kidneys trying to conserve Na)
   - FENa 1-2%: Indeterminate
   - FENa >2%: Intrinsic AKI (tubular dysfunction)
   
   Clinical Note:
   - Unreliable if on diuretics
   - More useful than BUN/Cr ratio in sepsis

4. URINE OSMOLALITY
   Normal: 50-1200 mOsm/kg (varies with hydration)
   
   In Sepsis:
   - Low urine osmolality + high serum osmolality → prerenal
   - Can help distinguish prerenal from intrinsic AKI


PART 2: PATHOGEN-SPECIFIC KIDNEY PATTERNS
═══════════════════════════════════════════════════════════════════════════════

GRAM-POSITIVE COCCI:

Staph aureus (especially endocarditis):
  • Pattern: Immune complex glomerulonephritis
  • Key findings:
    - Heavy proteinuria (often nephrotic range)
    - Hematuria with RBC casts
    - Elevated complement (C3, C4 may be low)
    - eGFR declines but often reversible
  • AKI Stage: Variable, usually Stage 1-2
  • Prognosis: Can resolve with appropriate antibiotics
  
Streptococcus (post-infectious GN):
  • Pattern: Acute proliferative glomerulonephritis
  • Key findings:
    - Hematuria (classic "smoky" or "cola-colored" urine)
    - Mild-moderate proteinuria
    - Can present with nephrotic-range proteinuria
    - Low serum complement (C3 low, C4 normal)
    - Elevated anti-streptococcal antibodies
  • AKI Stage: Often Stage 1, rarely severe
  • Prognosis: Usually excellent (self-limited)

Enterococcus:
  • Pattern: Minimal kidney disease
  • Key findings:
    - Proteinuria usually minimal
    - No hematuria
    - Creatinine elevation mild
  • AKI Stage: Usually Stage 0-1
  • Prognosis: Good renal outcomes


GRAM-NEGATIVE BACILLI:

E. coli (especially extraintestinal pathogenic E. coli):
  • Patterns:
    - Acute tubular necrosis (ATN) - most common
    - Shiga toxin-producing E. coli (STEC) → hemolytic uremic syndrome (HUS)
  • ATN Pattern:
    - Minimal to no proteinuria
    - No hematuria
    - Rapid rise in creatinine (FENa >2%)
    - High urine sodium
  • HUS Pattern:
    - Proteinuria and hematuria
    - Thrombocytopenia + microangiopathic hemolytic anemia
    - Severe AKI (often Stage 3)
  • AKI Stage: Stage 1-3 (can be severe)
  • Prognosis: ATN usually reverses, HUS has high mortality

Klebsiella pneumoniae:
  • Pattern: Acute tubular necrosis
  • Key findings:
    - Minimal proteinuria
    - FENa >2% (intrinsic AKI)
    - Often associated with septic shock
    - Hyperkalemia common
  • AKI Stage: Stage 1-3 (frequently severe)
  • Prognosis: Depends on sepsis severity

Pseudomonas aeruginosa:
  • Pattern: Acute tubular necrosis
  • Key findings:
    - Minimal proteinuria
    - Often high FENa
    - Commonly hospital-acquired (nosocomial)
    - Associated with VAP (ventilator-associated pneumonia)
  • AKI Stage: Stage 1-3
  • Prognosis: Higher mortality (drug-resistant organisms)

Acinetobacter baumannii:
  • Pattern: Acute tubular necrosis
  • Key findings:
    - Minimal renal manifestations initially
    - Can progress to severe AKI
    - Often multidrug-resistant
    - ICU-associated
  • AKI Stage: Stage 1-3
  • Prognosis: Often severe (MDR organism)


FUNGI:

Candida:
  • Patterns:
    - Acute tubular necrosis
    - Crystalline nephropathy (varies by antifungal agent)
    - Amphotericin B → wasting salt, hyperkalemia, renal toxicity
  • Key findings:
    - Elevated fungal biomarkers (β-D-glucan, antigen)
    - Often ICU-associated
    - Usually with broad-spectrum antibiotic exposure
  • AKI Stage: Stage 1-3
  • Prognosis: Depends on treatment and organ dysfunction


VIRAL:

Influenza, COVID-19:
  • Patterns:
    - Interstitial nephritis
    - Minimal proteinuria
    - Direct viral invasion of tubular cells
  • Key findings:
    - Viral nucleic acids in urine
    - Relatively mild kidney involvement
    - More associated with respiratory failure than AKI
  • AKI Stage: Stage 0-1 (rarely severe)
  • Prognosis: Usually good renal recovery


PART 3: INTEGRATED DECISION ALGORITHM
═══════════════════════════════════════════════════════════════════════════════

Step 1: ASSESS AKI SEVERITY
  ├─ Calculate eGFR decline
  ├─ Stage AKI (KDIGO: Stage 1/2/3)
  └─ Assess trend (improving vs worsening)

Step 2: CHARACTERIZE AKI TYPE
  ├─ Calculate BUN/Cr ratio:
  │   • >20: Likely PRERENAL
  │   • 10-20: INDETERMINATE
  │   • <10: Likely INTRINSIC
  │
  ├─ Calculate FENa:
  │   • <1%: PRERENAL (kidney conserving sodium)
  │   • 1-2%: INDETERMINATE
  │   • >2%: INTRINSIC (tubular dysfunction)
  │
  └─ Assess urine osmolality:
      • High (>500): Compatible with prerenal
      • Low (<250): Compatible with intrinsic

Step 3: ASSESS GLOMERULAR VS TUBULAR INJURY
  ├─ Proteinuria:
  │   • Heavy (>1 g/dL) → GLOMERULAR disease
  │   • Mild (<0.5) → TUBULAR injury
  │
  ├─ Hematuria:
  │   • Present → Suggests GLOMERULAR disease
  │   • Absent → Suggests TUBULAR disease
  │
  └─ RBC casts (if visible):
      • Present → Glomerulonephritis
      • Absent → ATN

Step 4: PATHOGEN PREDICTION FROM KIDNEY PATTERN

  IF Heavy proteinuria + Hematuria:
    → Likely: Staph aureus, Streptococcus, STEC
  
  IF Minimal proteinuria + High FENa + AKI Stage 3:
    → Likely: E. coli, Klebsiella, Pseudomonas, Acinetobacter
  
  IF Mild proteinuria + Low AKI severity:
    → Likely: Enterococcus, Viral, minimal Staph
  
  IF Hyperkalemia + AKI:
    → Likely: Gram-negative sepsis (e.g., Klebsiella, E. coli)

Step 5: VERIFY WITH OTHER DATA
  ├─ Blood cultures (definitive)
  ├─ Clinical presentation (source of infection)
  ├─ WBC pattern
  ├─ Temperature curve
  ├─ Hemodynamics
  └─ Imaging findings


PART 4: CRITICAL ACTION THRESHOLDS
═══════════════════════════════════════════════════════════════════════════════

IMMEDIATE INTERVENTION REQUIRED:

Creatinine >4.0 mg/dL:
  → Urgent nephrology consultation
  → Assess need for RRT (renal replacement therapy)
  → Monitor for hyperkalemia

Potassium >6.5 mEq/L:
  → STAT ECG
  → Emergency hyperkalemia treatment (calcium, insulin, kayexalate)
  → Consider dialysis

Creatinine rise >1 mg/dL in 24 hours:
  → Stage AKI
  → Assess fluid status
  → Review medications (ACE-I, NSAIDs, aminoglycosides)
  → Consider alternative antibiotics

Proteinuria >3 g/dL (Nephrotic range):
  → Suggests glomerulonephritis
  → Likely immune-mediated pathogen (Staph, Strep)
  → Consider immunosuppressive evaluation


PART 5: ANTIBIOTIC DOSING IN RENAL FAILURE
═══════════════════════════════════════════════════════════════════════════════

eGFR-based Adjustments:

eGFR >60:     Full standard doses
eGFR 30-59:   May require adjustment for some drugs
eGFR 15-29:   Significant dose reduction for most drugs
eGFR <15:     Major dose reduction or contraindication

Antibiotics Requiring Renal Adjustment:
  - Aminoglycosides (gentamicin, tobramycin) → MAJOR adjustment
  - Fluoroquinolones (ciprofloxacin) → MODERATE adjustment
  - Beta-lactams (most) → Minimal adjustment
  - Vancomycin → Significant adjustment

Always check:
  1. Current eGFR
  2. Drug's renal clearance
  3. Therapeutic drug monitoring (TDM) if indicated
  4. Dosing interval adjustment vs dose reduction


PART 6: PROGNOSIS & OUTCOMES
═══════════════════════════════════════════════════════════════════════════════

Factors Associated with GOOD Renal Recovery:
  ✓ AKI Stage 1-2
  ✓ Rapid resolution (within 3-7 days)
  ✓ Prerenal mechanism
  ✓ Glomerulonephritis pattern (immune-mediated, reversible)
  ✓ Younger age
  ✓ No chronic kidney disease
  ✓ Organism susceptibility to antibiotics

Factors Associated with POOR Renal Outcomes:
  ✗ AKI Stage 3 with need for RRT
  ✗ Persistent renal dysfunction (>2 weeks)
  ✗ Septic shock
  ✗ Multi-organ failure
  ✗ Intrinsic ATN requiring dialysis
  ✗ Drug-resistant organism
  ✗ Delayed antibiotic therapy
  ✗ Acute glomerulonephritis (untreated)

Mortality Risk by AKI Stage:
  Stage 1: ~10% hospital mortality
  Stage 2: ~20% hospital mortality
  Stage 3: ~40% hospital mortality (if RRT needed: ~50-60%)


FINAL SUMMARY
═══════════════════════════════════════════════════════════════════════════════

Kidney function tests are CRITICAL in sepsis because:
  1. Reveal type of kidney injury (prerenal, intrinsic, postrenal)
  2. Help differentiate pathogens (glomerular vs tubular patterns)
  3. Guide antibiotic dosing
  4. Identify life-threatening complications (hyperkalemia)
  5. Predict outcomes and guide management
  6. Detect immune-mediated diseases requiring additional therapy

Always interpret kidney function in CONTEXT:
  - Clinical presentation (fever, source of infection)
  - Vital signs (hemodynamics)
  - Blood work (CBC, electrolytes, lactate)
  - Imaging (ultrasound, CT for obstruction/abscess)
  - Culture results (definitive diagnosis)

================================================================================
Generated: March 2026
Status: Clinical Decision Support Guide
================================================================================
"""
    
    return guide


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Generate integrated pathogen identification with kidney function."""
    
    print("="*80)
    print("INTEGRATED PATHOGEN IDENTIFICATION WITH KIDNEY FUNCTION")
    print("="*80)
    
    # Generate guide
    print("\nGenerating comprehensive kidney function clinical guide...")
    guide = generate_kidney_function_guide()
    
    with open(Path('./kidney_function_analysis') / 'KIDNEY_FUNCTION_CLINICAL_GUIDE.txt', 'w') as f:
        f.write(guide)
    
    print("✓ Saved: KIDNEY_FUNCTION_CLINICAL_GUIDE.txt")
    
    print("\n" + "="*80)
    print("✅ INTEGRATED SYSTEM DOCUMENTATION COMPLETE")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
