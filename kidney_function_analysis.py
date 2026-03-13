#!/usr/bin/env python3
"""
=============================================================================
KIDNEY FUNCTION ANALYSIS - SEPSIS PATHOGEN IDENTIFICATION EXTENSION
Blood and Urine Lab Tests for Renal Assessment in ICU Sepsis
=============================================================================

Author: PenuX Research Team
Date: March 2026
Purpose: Extract, analyze, and integrate kidney function markers into
         pathogen identification system for sepsis

KIDNEY FUNCTION MARKERS (MIMIC-III itemids):
═════════════════════════════════════════════════════════════════════════

BLOOD TESTS:
  Creatinine (serum):     itemid 50912 [mg/dL] - GFR marker
  Potassium (serum):      itemid 50971 [mEq/L] - Hyperkalemia sign
  BUN/Urea nitrogen:      itemid 51006 [mg/dL] - GFR marker
  eGFR (calculated):      Derived from creatinine & age
  Phosphate:              itemid 50897 [mg/dL] - Renal handling
  Magnesium:              itemid 50959 [mg/dL] - Renal reabsorption

URINE TESTS:
  Urine creatinine:       itemid 51084 [mg/dL] - 24h collection
  Urine sodium:           itemid 51084 [mEq/L] - FENa calculation
  Urine osmolality:       itemid 51103 [mOsm/kg] - Concentrating ability
  Proteinuria:            itemid 51084 [g/dL] - Glomerular damage
  Hematuria:              Presence of RBC in urine

CALCULATED METRICS:
  eGFR (MDRD):            [mL/min/1.73m²] - Kidney function stage
  Fractional Excretion Sodium (FENa): [%] - Prerenal vs intrinsic AKI
  BUN/Cr Ratio:           [dimensionless] - Prerenal vs intrinsic
  Urine Osmolal Gap:      [mOsm/kg] - Acid-base status
  RIFLE Score:            [Risk/Injury/Failure/Loss/ESRD] - AKI severity

CLINICAL INTERPRETATION:
═════════════════════════════════════════════════════════════════════════

NORMAL RANGES:
  Creatinine:      0.6-1.2 mg/dL (males), 0.4-1.0 (females)
  BUN:             7-20 mg/dL
  eGFR:            >60 mL/min/1.73m² (normal kidney function)
  K+ (potassium):  3.5-5.0 mEq/L
  Phosphate:       2.5-4.5 mg/dL

AKI STAGES (KDIGO):
  Stage 1: Cr 1.5-1.9x baseline OR ≥0.3 increase
  Stage 2: Cr 2-2.9x baseline
  Stage 3: Cr ≥3x baseline OR ≥4.0 OR RRT initiated

PATHOGEN-SPECIFIC KIDNEY PATTERNS:
  • E. coli (hemolytic uremic syndrome): Proteinuria, hematuria
  • Staph aureus (endocarditis): Immune complex GN, hematuria
  • Streptococcus: Post-infectious GN, proteinuria
  • Acinetobacter: Acute tubular necrosis (ATN), high FENa
  • Pseudomonas: Tubular injury, elevated Cr
  • Fungal (Candida): Crystalline nephropathy (varies by drug)
  • Viral: Interstitial nephritis, proteinuria

=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
from scipy import stats

# ============================================================================
# KIDNEY FUNCTION CALCULATOR
# ============================================================================

class KidneyFunctionCalculator:
    """Calculate kidney function metrics from lab values."""
    
    @staticmethod
    def egfr_mdrd(creatinine, age, sex, race='non_black'):
        """Calculate eGFR using MDRD equation.
        
        MDRD Study Equation:
        eGFR = 175 × (Cr)^(-1.154) × (age)^(-0.203) × [0.742 if female] × [1.212 if Black]
        
        Args:
            creatinine: Serum creatinine in mg/dL
            age: Age in years
            sex: 'male' or 'female'
            race: 'black' or 'non_black'
        
        Returns:
            eGFR in mL/min/1.73m²
        """
        if creatinine <= 0 or age <= 0:
            return np.nan
        
        egfr = 175 * np.power(creatinine, -1.154) * np.power(age, -0.203)
        
        if sex == 'female':
            egfr *= 0.742
        
        if race == 'black':
            egfr *= 1.212
        
        return egfr
    
    @staticmethod
    def egfr_ckd_epi(creatinine, age, sex, race='non_black'):
        """Calculate eGFR using CKD-EPI 2009 equation (more accurate for higher eGFR).
        
        For serum creatinine (mg/dL):
        If female:
          κ = 0.7, α = -0.329
          If Cr ≤ κ: eGFR = 144 × (Cr/κ)^α × (age)^(-0.025) × 1.018
          If Cr > κ: eGFR = 144 × (Cr/κ)^(-1.209) × (age)^(-0.025) × 1.018
        
        If male:
          κ = 0.9, α = -0.411
          If Cr ≤ κ: eGFR = 141 × (Cr/κ)^α × (age)^(-0.018)
          If Cr > κ: eGFR = 141 × (Cr/κ)^(-1.209) × (age)^(-0.018)
        """
        if creatinine <= 0 or age <= 0:
            return np.nan
        
        if sex == 'female':
            k = 0.7
            a = -0.329
            mult = 1.018
        else:
            k = 0.9
            a = -0.411
            mult = 1.0
        
        cr_k_ratio = creatinine / k
        
        if cr_k_ratio <= 1:
            egfr = 144 * np.power(cr_k_ratio, a) * np.power(age, -0.025 if sex == 'female' else -0.018) * mult
        else:
            egfr = 144 * np.power(cr_k_ratio, -1.209) * np.power(age, -0.025 if sex == 'female' else -0.018) * mult
        
        if race == 'black':
            egfr *= 1.159
        
        return egfr
    
    @staticmethod
    def estimate_fena(urine_na, urine_cr, serum_na, serum_cr):
        """Calculate Fractional Excretion of Sodium (FENa).
        
        FENa (%) = (UNa × SCr) / (SNa × UCr) × 100
        
        Interpretation:
          FENa < 1%:      Prerenal AKI (intact tubular reabsorption)
          FENa 1-2%:      Indeterminate
          FENa > 2%:      Intrinsic AKI or post-renal (tubular dysfunction)
        
        Args:
            urine_na: Urine sodium (mEq/L)
            urine_cr: Urine creatinine (mg/dL)
            serum_na: Serum sodium (mEq/L)
            serum_cr: Serum creatinine (mg/dL)
        
        Returns:
            FENa percentage
        """
        if serum_cr <= 0 or urine_cr <= 0 or serum_na <= 0:
            return np.nan
        
        fena = (urine_na * serum_cr) / (serum_na * urine_cr) * 100
        return fena
    
    @staticmethod
    def classify_aki_severity(creatinine_baseline, creatinine_peak):
        """Classify AKI severity using KDIGO criteria.
        
        Stage 1: Cr increase 1.5-1.9× baseline OR ≥0.3 mg/dL increase
        Stage 2: Cr increase 2.0-2.9× baseline
        Stage 3: Cr increase ≥3× baseline OR ≥4.0 mg/dL OR RRT initiated
        
        Args:
            creatinine_baseline: Baseline serum creatinine
            creatinine_peak: Peak serum creatinine during hospitalization
        
        Returns:
            Tuple: (stage, description)
        """
        if creatinine_baseline <= 0:
            return (0, "Unknown (no baseline)")
        
        ratio = creatinine_peak / creatinine_baseline
        increase = creatinine_peak - creatinine_baseline
        
        if ratio >= 3.0 or creatinine_peak >= 4.0:
            return (3, "Stage 3 (Failure)")
        elif ratio >= 2.0:
            return (2, "Stage 2 (Injury)")
        elif ratio >= 1.5 or increase >= 0.3:
            return (1, "Stage 1 (Risk)")
        else:
            return (0, "No AKI")
    
    @staticmethod
    def bun_cr_ratio(bun, creatinine):
        """Calculate BUN/Cr ratio.
        
        Normal: 10-20
        Prerenal AKI: >20 (intact tubular reabsorption)
        Intrinsic AKI: <10-15 (tubular injury)
        
        Args:
            bun: Blood urea nitrogen (mg/dL)
            creatinine: Serum creatinine (mg/dL)
        
        Returns:
            BUN/Cr ratio
        """
        if creatinine <= 0:
            return np.nan
        return bun / creatinine


# ============================================================================
# KIDNEY FUNCTION ANALYSIS
# ============================================================================

class KidneyFunctionAnalysis:
    """Comprehensive kidney function analysis for sepsis cohort."""
    
    def __init__(self, output_dir='.'):
        """Initialize kidney function analyzer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data = {}
        self.results = {}
    
    def create_synthetic_kidney_data(self, n_samples=5856, n_pathogens=10):
        """Create realistic synthetic kidney function data.
        
        Patterns by pathogen:
        - Gram-negative (E. coli, Klebsiella): Higher risk of AKI
        - Staph (endocarditis): Hematuria, proteinuria
        - Fungal: Crystalline nephropathy patterns
        - Viral: Interstitial nephritis pattern
        """
        np.random.seed(42)
        
        # Base data
        ages = np.random.normal(60, 15, n_samples)
        ages = np.clip(ages, 20, 100)
        
        pathogens = np.random.choice(range(n_pathogens), n_samples)
        sexes = np.random.choice(['male', 'female'], n_samples, p=[0.6, 0.4])
        
        # Baseline creatinine (normal)
        baseline_cr = np.random.normal(1.0, 0.3, n_samples)
        baseline_cr = np.clip(baseline_cr, 0.5, 2.5)
        
        # Peak creatinine varies by pathogen
        peak_cr = baseline_cr.copy()
        
        # Gram-negative bacteria → higher AKI risk
        gram_neg_mask = np.isin(pathogens, [1, 2, 3, 4])
        peak_cr[gram_neg_mask] += np.random.normal(1.5, 0.8, gram_neg_mask.sum())
        
        # Fungal → moderate AKI risk
        fungal_mask = (pathogens == 7)
        peak_cr[fungal_mask] += np.random.normal(0.8, 0.4, fungal_mask.sum())
        
        # Viral → lower AKI risk
        viral_mask = (pathogens == 8)
        peak_cr[viral_mask] += np.random.normal(0.3, 0.2, viral_mask.sum())
        
        # Others → moderate risk
        other_mask = ~(gram_neg_mask | fungal_mask | viral_mask)
        peak_cr[other_mask] += np.random.normal(0.7, 0.4, other_mask.sum())
        
        peak_cr = np.clip(peak_cr, 0.5, 8.0)
        
        # Calculate eGFR
        egfr_baseline = np.array([
            KidneyFunctionCalculator.egfr_ckd_epi(cr, age, sex)
            for cr, age, sex in zip(baseline_cr, ages, sexes)
        ])
        
        egfr_peak = np.array([
            KidneyFunctionCalculator.egfr_ckd_epi(cr, age, sex)
            for cr, age, sex in zip(peak_cr, ages, sexes)
        ])
        
        # BUN (mg/dL)
        bun_baseline = baseline_cr * np.random.uniform(10, 20, n_samples)
        bun_peak = peak_cr * np.random.uniform(12, 25, n_samples)
        
        # Potassium (mEq/L) - elevated in AKI
        k_baseline = np.random.normal(4.0, 0.4, n_samples)
        k_peak = k_baseline + (peak_cr - baseline_cr) * 0.3  # Rises with AKI
        k_peak = np.clip(k_peak, 2.5, 7.0)
        
        # Phosphate (mg/dL) - elevated in AKI
        phos_baseline = np.random.normal(3.5, 0.5, n_samples)
        phos_peak = phos_baseline + (peak_cr - baseline_cr) * 0.4
        phos_peak = np.clip(phos_peak, 1.5, 8.0)
        
        # Proteinuria (g/dL) - Gram+ and fungal higher
        proteinuria = np.random.uniform(0, 0.5, n_samples)
        staph_mask = (pathogens == 0)
        proteinuria[staph_mask] = np.random.uniform(0.5, 3.0, staph_mask.sum())
        fungal_mask = (pathogens == 7)
        proteinuria[fungal_mask] = np.random.uniform(0.5, 2.0, fungal_mask.sum())
        
        # Urine sodium (mEq/L)
        urine_na = np.random.uniform(20, 120, n_samples)
        
        # Create DataFrame
        df = pd.DataFrame({
            'hadm_id': np.arange(n_samples),
            'age': ages,
            'sex': sexes,
            'pathogen_id': pathogens,
            'baseline_creatinine': baseline_cr,
            'peak_creatinine': peak_cr,
            'baseline_egfr': egfr_baseline,
            'peak_egfr': egfr_peak,
            'baseline_bun': bun_baseline,
            'peak_bun': bun_peak,
            'baseline_potassium': k_baseline,
            'peak_potassium': k_peak,
            'baseline_phosphate': phos_baseline,
            'peak_phosphate': phos_peak,
            'proteinuria': proteinuria,
            'urine_sodium': urine_na,
        })
        
        return df
    
    def analyze_kidney_function(self, df):
        """Analyze kidney function data."""
        print("[1/4] Computing kidney function metrics...")
        
        # Calculate AKI severity
        aki_stages = []
        aki_descriptions = []
        for _, row in df.iterrows():
            stage, desc = KidneyFunctionCalculator.classify_aki_severity(
                row['baseline_creatinine'], row['peak_creatinine']
            )
            aki_stages.append(stage)
            aki_descriptions.append(desc)
        
        df['aki_stage'] = aki_stages
        df['aki_description'] = aki_descriptions
        
        # Calculate BUN/Cr ratio
        df['bun_cr_ratio'] = df.apply(
            lambda row: KidneyFunctionCalculator.bun_cr_ratio(
                row['peak_bun'], row['peak_creatinine']
            ),
            axis=1
        )
        
        # Calculate FENa (simplified without detailed urine creatinine)
        df['fena_estimate'] = df.apply(
            lambda row: KidneyFunctionCalculator.estimate_fena(
                row['urine_sodium'], row['peak_creatinine'] * 100,
                140, row['peak_creatinine']  # Assuming typical serum Na
            ),
            axis=1
        )
        
        # Creatinine increase
        df['cr_increase'] = df['peak_creatinine'] - df['baseline_creatinine']
        df['cr_ratio'] = df['peak_creatinine'] / df['baseline_creatinine']
        df['egfr_decline'] = df['baseline_egfr'] - df['peak_egfr']
        
        # Hyperkalemia
        df['hyperkalemia'] = (df['peak_potassium'] > 5.5).astype(int)
        
        # Proteinuria severity
        df['proteinuria_severity'] = pd.cut(
            df['proteinuria'],
            bins=[0, 0.3, 1.0, 3.0, np.inf],
            labels=['Normal', 'Mild', 'Moderate', 'Severe']
        )
        
        self.data = df
        print(f"  ✓ Computed metrics for {len(df):,} patients")
        
        return df
    
    def generate_summary_statistics(self):
        """Generate summary statistics by AKI stage."""
        print("\n[2/4] Computing summary statistics...")
        
        summary = self.data.groupby('aki_stage').agg({
            'baseline_creatinine': ['mean', 'std'],
            'peak_creatinine': ['mean', 'std'],
            'baseline_egfr': ['mean', 'std'],
            'peak_egfr': ['mean', 'std'],
            'cr_increase': ['mean', 'std'],
            'bun_cr_ratio': ['mean', 'std'],
            'peak_potassium': ['mean', 'std'],
            'proteinuria': ['mean', 'std'],
            'hyperkalemia': ['mean'],  # Frequency
            'hadm_id': ['count'],
        }).round(3)
        
        print("\n  Kidney Function by AKI Stage:")
        print(summary)
        
        self.results['aki_summary'] = summary
        
        return summary
    
    def analyze_by_pathogen(self):
        """Analyze kidney function patterns by pathogen."""
        print("\n[3/4] Analyzing kidney function by pathogen...")
        
        pathogen_names = {
            0: 'Staph aureus',
            1: 'E. coli',
            2: 'Klebsiella',
            3: 'Acinetobacter',
            4: 'Pseudomonas',
            5: 'Streptococcus',
            6: 'Enterococcus',
            7: 'Candida/Fungal',
            8: 'Viral',
            9: 'Other/Mixed',
        }
        
        pathogen_analysis = []
        
        for pathogen_id in range(10):
            mask = self.data['pathogen_id'] == pathogen_id
            if mask.sum() == 0:
                continue
            
            subset = self.data[mask]
            
            analysis_dict = {
                'pathogen_id': pathogen_id,
                'pathogen_name': pathogen_names[pathogen_id],
                'n_samples': mask.sum(),
                'mean_cr_baseline': subset['baseline_creatinine'].mean(),
                'mean_cr_peak': subset['peak_creatinine'].mean(),
                'mean_cr_increase': subset['cr_increase'].mean(),
                'mean_egfr_decline': subset['egfr_decline'].mean(),
                'aki_stage_3_freq': (subset['aki_stage'] == 3).mean(),
                'hyperkalemia_freq': subset['hyperkalemia'].mean(),
                'proteinuria_mean': subset['proteinuria'].mean(),
                'bun_cr_ratio_mean': subset['bun_cr_ratio'].mean(),
            }
            
            pathogen_analysis.append(analysis_dict)
        
        pathogen_df = pd.DataFrame(pathogen_analysis)
        
        print("\n  Top Pathogens by AKI Risk (Stage 3 Frequency):")
        print(pathogen_df[['pathogen_name', 'aki_stage_3_freq', 'mean_cr_increase']].sort_values(
            'aki_stage_3_freq', ascending=False
        ))
        
        self.results['pathogen_analysis'] = pathogen_df
        
        return pathogen_df
    
    def save_results(self):
        """Save all results."""
        print("\n[4/4] Saving results...")
        
        # Save detailed data
        self.data.to_csv(self.output_dir / 'kidney_function_data.csv', index=False)
        print(f"  ✓ Saved: kidney_function_data.csv ({len(self.data):,} rows)")
        
        # Save summary statistics
        self.results['aki_summary'].to_csv(self.output_dir / 'aki_summary_by_stage.csv')
        print(f"  ✓ Saved: aki_summary_by_stage.csv")
        
        # Save pathogen analysis
        self.results['pathogen_analysis'].to_csv(self.output_dir / 'kidney_function_by_pathogen.csv', index=False)
        print(f"  ✓ Saved: kidney_function_by_pathogen.csv")
        
        # Save as JSON
        results_dict = {
            'analysis_date': datetime.now().isoformat(),
            'n_samples': len(self.data),
            'aki_distribution': self.data['aki_stage'].value_counts().to_dict(),
            'pathogen_aki_risk': self.results['pathogen_analysis'][
                ['pathogen_name', 'aki_stage_3_freq', 'mean_cr_increase']
            ].to_dict('records'),
        }
        
        with open(self.output_dir / 'kidney_function_summary.json', 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        print(f"  ✓ Saved: kidney_function_summary.json")
    
    def generate_visualizations(self):
        """Generate kidney function visualizations."""
        print("\n  Generating visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        # 1. AKI stage distribution
        ax = axes[0, 0]
        aki_counts = self.data['aki_stage'].value_counts().sort_index()
        ax.bar(aki_counts.index, aki_counts.values, color=['green', 'yellow', 'orange', 'red'])
        ax.set_xlabel('AKI Stage')
        ax.set_ylabel('Number of Patients')
        ax.set_title('AKI Stage Distribution')
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(['No AKI', 'Stage 1', 'Stage 2', 'Stage 3'])
        ax.grid(True, alpha=0.3, axis='y')
        
        # 2. Creatinine change by AKI stage
        ax = axes[0, 1]
        ax.boxplot([
            self.data[self.data['aki_stage'] == i]['cr_increase'].dropna()
            for i in range(4)
        ], labels=['No AKI', 'Stage 1', 'Stage 2', 'Stage 3'])
        ax.set_ylabel('Creatinine Increase (mg/dL)')
        ax.set_title('Creatinine Change by AKI Stage')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. eGFR decline by pathogen
        ax = axes[0, 2]
        pathogen_egfr = self.results['pathogen_analysis'].sort_values('mean_egfr_decline', ascending=False).head(10)
        ax.barh(range(len(pathogen_egfr)), pathogen_egfr['mean_egfr_decline'], color='crimson', alpha=0.7)
        ax.set_yticks(range(len(pathogen_egfr)))
        ax.set_yticklabels(pathogen_egfr['pathogen_name'], fontsize=9)
        ax.set_xlabel('Mean eGFR Decline (mL/min/1.73m²)')
        ax.set_title('Top 10 Pathogens by eGFR Decline')
        ax.grid(True, alpha=0.3, axis='x')
        
        # 4. Hyperkalemia frequency by pathogen
        ax = axes[1, 0]
        pathogen_hyper = self.results['pathogen_analysis'].sort_values('hyperkalemia_freq', ascending=False).head(10)
        ax.barh(range(len(pathogen_hyper)), pathogen_hyper['hyperkalemia_freq'] * 100, color='orange', alpha=0.7)
        ax.set_yticks(range(len(pathogen_hyper)))
        ax.set_yticklabels(pathogen_hyper['pathogen_name'], fontsize=9)
        ax.set_xlabel('Hyperkalemia Frequency (%)')
        ax.set_title('Top 10 Pathogens by Hyperkalemia Risk')
        ax.grid(True, alpha=0.3, axis='x')
        
        # 5. Proteinuria by pathogen
        ax = axes[1, 1]
        pathogen_prot = self.results['pathogen_analysis'].sort_values('proteinuria_mean', ascending=False).head(10)
        ax.barh(range(len(pathogen_prot)), pathogen_prot['proteinuria_mean'], color='purple', alpha=0.7)
        ax.set_yticks(range(len(pathogen_prot)))
        ax.set_yticklabels(pathogen_prot['pathogen_name'], fontsize=9)
        ax.set_xlabel('Mean Proteinuria (g/dL)')
        ax.set_title('Top 10 Pathogens by Proteinuria')
        ax.grid(True, alpha=0.3, axis='x')
        
        # 6. AKI Stage 3 risk by pathogen
        ax = axes[1, 2]
        pathogen_aki3 = self.results['pathogen_analysis'].sort_values('aki_stage_3_freq', ascending=False).head(10)
        ax.barh(range(len(pathogen_aki3)), pathogen_aki3['aki_stage_3_freq'] * 100, color='darkred', alpha=0.7)
        ax.set_yticks(range(len(pathogen_aki3)))
        ax.set_yticklabels(pathogen_aki3['pathogen_name'], fontsize=9)
        ax.set_xlabel('AKI Stage 3 Frequency (%)')
        ax.set_title('Top 10 Pathogens by Severe AKI Risk')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'kidney_function_analysis.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: kidney_function_analysis.png")
        plt.close()
    
    def run_complete_analysis(self):
        """Execute complete kidney function analysis."""
        print("="*80)
        print("KIDNEY FUNCTION ANALYSIS - COMPLETE PIPELINE")
        print("="*80)
        
        # Create synthetic data
        print("\nCreating synthetic kidney function data...")
        df = self.create_synthetic_kidney_data(n_samples=5856, n_pathogens=10)
        
        # Analyze
        self.analyze_kidney_function(df)
        self.generate_summary_statistics()
        self.analyze_by_pathogen()
        
        # Visualize
        self.generate_visualizations()
        
        # Save
        self.save_results()
        
        print("\n" + "="*80)
        print("✅ KIDNEY FUNCTION ANALYSIS COMPLETE")
        print("="*80)
        print("\nOutput files:")
        print("  - kidney_function_data.csv (5,856 patients)")
        print("  - aki_summary_by_stage.csv")
        print("  - kidney_function_by_pathogen.csv")
        print("  - kidney_function_summary.json")
        print("  - kidney_function_analysis.png (6-panel visualization)")
        print("="*80 + "\n")


# ============================================================================
# INTEGRATION WITH PATHOGEN IDENTIFICATION
# ============================================================================

class KidneyFunctionScoring:
    """Score kidney function impact on pathogen identification."""
    
    @staticmethod
    def get_aki_risk_weight(pathogen_id):
        """Get AKI risk weighting by pathogen.
        
        Used to adjust pathogen scores based on kidney injury severity.
        """
        aki_weights = {
            0: 0.3,   # Staph: moderate risk
            1: 0.9,   # E. coli: very high AKI risk
            2: 0.85,  # Klebsiella: very high AKI risk
            3: 0.8,   # Acinetobacter: very high AKI risk (nosocomial)
            4: 0.75,  # Pseudomonas: high AKI risk
            5: 0.4,   # Streptococcus: moderate risk
            6: 0.35,  # Enterococcus: moderate risk
            7: 0.7,   # Fungal: high risk (crystalline nephropathy)
            8: 0.2,   # Viral: low AKI risk
            9: 0.6,   # Other/Mixed: moderate-high risk
        }
        return aki_weights.get(pathogen_id, 0.5)
    
    @staticmethod
    def score_kidney_function(egfr_decline, cr_increase, hyperkalemia, proteinuria):
        """Compute kidney function severity score (0-1).
        
        Args:
            egfr_decline: eGFR decline (mL/min/1.73m²)
            cr_increase: Creatinine increase (mg/dL)
            hyperkalemia: 1 if K+ > 5.5, else 0
            proteinuria: Proteinuria level (g/dL)
        
        Returns:
            Kidney injury severity score (0-1)
        """
        score = 0.0
        
        # eGFR component (max 0.4)
        if egfr_decline > 60:
            score += 0.4
        elif egfr_decline > 30:
            score += 0.3
        elif egfr_decline > 0:
            score += 0.2
        
        # Creatinine component (max 0.3)
        if cr_increase > 3.0:
            score += 0.3
        elif cr_increase > 1.5:
            score += 0.2
        elif cr_increase > 0.3:
            score += 0.1
        
        # Hyperkalemia (max 0.2)
        if hyperkalemia:
            score += 0.2
        
        # Proteinuria (max 0.1)
        if proteinuria > 1.0:
            score += 0.1
        
        return min(score, 1.0)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run kidney function analysis."""
    
    output_dir = Path('./kidney_function_analysis')
    
    analyzer = KidneyFunctionAnalysis(output_dir=output_dir)
    analyzer.run_complete_analysis()


if __name__ == '__main__':
    main()
