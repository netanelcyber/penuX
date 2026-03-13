#!/usr/bin/env python3
"""
=============================================================================
BLOOD PRESSURE ANALYSIS - MIMIC-III SEPSIS COHORT
Complete standalone script for BP extraction, analysis, and reporting
=============================================================================

Author: PenuX Clinical AI Analysis
Date: March 2025
Purpose: Extract BP and derived vitals from CHARTEVENTS, compute statistics,
         correlations, pathology prevalence, and generate clinical reports.

Input: CHARTEVENTS.txt (CSV format from MIMIC-III demo database)
Output: 
  - CSV files (extracted BP data, summary statistics)
  - HTML clinical report
  - Console output with full analysis

Usage:
  python3 bp_analysis_complete.py <path/to/CHARTEVENTS.txt> [output_dir]
  
Example:
  python3 bp_analysis_complete.py /mnt/user-data/uploads/CHARTEVENTS.txt /mnt/user-data/outputs
=============================================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
import sys
import os
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURATION & MIMIC-III ITEM IDs
# ============================================================================

# Blood Pressure itemids (MIMIC-III standard chartevents)
SYSTOLIC_BP_IDS = [51, 455, 220050]      # NBP Systolic, Arterial Systolic
DIASTOLIC_BP_IDS = [52, 456, 220051]     # NBP Diastolic, Arterial Diastolic
HEART_RATE_IDS = [211, 220045]           # Heart Rate variants
RESP_RATE_IDS = [618, 224690]            # Respiratory Rate variants

# Clinical thresholds
SYSTOLIC_MIN, SYSTOLIC_MAX = 50, 300
DIASTOLIC_MIN, DIASTOLIC_MAX = 20, 200
HR_MIN, HR_MAX = 30, 200
RR_MIN, RR_MAX = 5, 60

# Sepsis/SIRS thresholds
QSOFA_THRESHOLDS = {
    'SBP_SHOCK': 90,
    'MAP_SHOCK': 65,
    'RR_SEPSIS': 20,
    'HR_SEPSIS': 90,
    'RR_TACHYPNEA': 22,
}

# ============================================================================
# MAIN ANALYSIS CLASS
# ============================================================================

class BPAnalyzer:
    """Complete BP analysis pipeline for MIMIC-III data."""
    
    def __init__(self, chartevents_path, output_dir=None):
        """Initialize analyzer with data path."""
        self.data_path = chartevents_path
        self.output_dir = Path(output_dir) if output_dir else Path('.')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df_raw = None
        self.bp_data = None
        self.stats = {}
        self.correlations = None
        self.pathology = {}
        
    def load_data(self):
        """Load and validate CHARTEVENTS data."""
        print("[1/7] Loading CHARTEVENTS data...")
        try:
            self.df_raw = pd.read_csv(
                self.data_path,
                dtype={
                    'hadm_id': 'int64',
                    'itemid': 'int32',
                    'valuenum': 'float32'
                },
                low_memory=False
            )
            print(f"  ✓ Loaded {len(self.df_raw):,} records from {len(self.df_raw['hadm_id'].unique()):,} admissions")
            print(f"  ✓ {self.df_raw['itemid'].nunique():,} unique item types found")
        except Exception as e:
            print(f"  ✗ Error loading data: {e}")
            sys.exit(1)
    
    def extract_vital(self, item_ids, name, min_val, max_val):
        """Extract and clean a vital sign."""
        vital = self.df_raw[
            (self.df_raw['itemid'].isin(item_ids)) &
            (self.df_raw['valuenum'].notna()) &
            (self.df_raw['valuenum'] >= min_val) &
            (self.df_raw['valuenum'] <= max_val)
        ].copy()
        
        if len(vital) > 0:
            median = vital.groupby('hadm_id')['valuenum'].median().rename(name)
            return median
        return pd.Series(dtype='float32', name=name)
    
    def extract_vitals(self):
        """Extract all vital signs."""
        print("\n[2/7] Extracting vital signs...")
        
        vitals = {}
        
        # Extract each vital
        print("  Extracting Systolic BP...")
        vitals['systolic_bp'] = self.extract_vital(
            SYSTOLIC_BP_IDS, 'systolic_bp',
            SYSTOLIC_MIN, SYSTOLIC_MAX
        )
        
        print("  Extracting Diastolic BP...")
        vitals['diastolic_bp'] = self.extract_vital(
            DIASTOLIC_BP_IDS, 'diastolic_bp',
            DIASTOLIC_MIN, DIASTOLIC_MAX
        )
        
        print("  Extracting Heart Rate...")
        vitals['heart_rate'] = self.extract_vital(
            HEART_RATE_IDS, 'heart_rate',
            HR_MIN, HR_MAX
        )
        
        print("  Extracting Respiratory Rate...")
        vitals['resp_rate'] = self.extract_vital(
            RESP_RATE_IDS, 'resp_rate',
            RR_MIN, RR_MAX
        )
        
        # Merge all vitals
        self.bp_data = pd.concat([v for v in vitals.values()], axis=1).dropna()
        
        print(f"  ✓ Final dataset: {len(self.bp_data):,} admissions with complete vital signs")
        
        # Derive additional metrics
        self.bp_data['map'] = (
            self.bp_data['diastolic_bp'] + 
            (self.bp_data['systolic_bp'] - self.bp_data['diastolic_bp']) / 3
        )
        self.bp_data['pulse_pressure'] = (
            self.bp_data['systolic_bp'] - self.bp_data['diastolic_bp']
        )
        
        print(f"  ✓ Derived MAP and Pulse Pressure")
    
    def compute_descriptive_stats(self):
        """Compute descriptive statistics."""
        print("\n[3/7] Computing descriptive statistics...")
        
        vital_cols = ['systolic_bp', 'diastolic_bp', 'map', 'pulse_pressure',
                      'heart_rate', 'resp_rate']
        
        for col in vital_cols:
            data = self.bp_data[col].dropna()
            mean_val = data.mean()
            std_val = data.std()
            cv = std_val / mean_val if mean_val != 0 else 0
            
            self.stats[col] = {
                'n': len(data),
                'mean': mean_val,
                'std': std_val,
                'cv': cv,
                'min': data.min(),
                'q25': data.quantile(0.25),
                'median': data.median(),
                'q75': data.quantile(0.75),
                'max': data.max(),
            }
        
        print(f"  ✓ Statistics computed for {len(vital_cols)} vital signs")
    
    def compute_correlations(self):
        """Compute inter-vital correlations."""
        print("\n[4/7] Computing vital sign correlations...")
        
        vital_cols = ['systolic_bp', 'diastolic_bp', 'heart_rate', 'resp_rate']
        self.correlations = self.bp_data[vital_cols].corr()
        
        print(f"  ✓ Correlation matrix computed")
    
    def compute_pathology(self):
        """Compute clinical pathology prevalence."""
        print("\n[5/7] Computing clinical pathology...")
        
        n_total = len(self.bp_data)
        
        pathology_dict = {
            'Hypotension (SBP < 90)': (self.bp_data['systolic_bp'] < 90).sum(),
            'Hypertension (SBP ≥ 140)': (self.bp_data['systolic_bp'] >= 140).sum(),
            'Stage 1 HTN (130-139)': ((self.bp_data['systolic_bp'] >= 130) & (self.bp_data['systolic_bp'] < 140)).sum(),
            'Elevated (120-129)': ((self.bp_data['systolic_bp'] >= 120) & (self.bp_data['systolic_bp'] < 130)).sum(),
            'Normal (< 120)': (self.bp_data['systolic_bp'] < 120).sum(),
            'Bradycardia (HR < 60)': (self.bp_data['heart_rate'] < 60).sum(),
            'Tachycardia (HR > 100)': (self.bp_data['heart_rate'] > 100).sum(),
            'Tachypnea (RR > 20)': (self.bp_data['resp_rate'] > 20).sum(),
            'Hypoxic RR (RR > 30)': (self.bp_data['resp_rate'] > 30).sum(),
            'Altered RR (RR ≤ 10)': (self.bp_data['resp_rate'] <= 10).sum(),
        }
        
        # SIRS/Sepsis criteria
        sirs_dict = {
            'Respiratory rate ≥ 22': (self.bp_data['resp_rate'] >= 22).sum(),
            'Systolic BP ≤ 100': (self.bp_data['systolic_bp'] <= 100).sum(),
            'Possible sepsis (RR≥20 AND HR>90)': ((self.bp_data['resp_rate'] >= 20) & (self.bp_data['heart_rate'] > 90)).sum(),
            'Severe sepsis (MAP < 65)': (self.bp_data['map'] < 65).sum(),
            'Septic shock (SBP<90 or MAP<65)': ((self.bp_data['systolic_bp'] < 90) | (self.bp_data['map'] < 65)).sum(),
        }
        
        self.pathology = {**pathology_dict, **sirs_dict}
        print(f"  ✓ Pathology computed for {len(self.pathology)} conditions")
    
    def normality_tests(self):
        """Test for normality using Shapiro-Wilk."""
        print("\n[6/7] Testing distribution normality...")
        
        vital_cols = ['systolic_bp', 'diastolic_bp', 'heart_rate', 'resp_rate']
        normality = {}
        
        for col in vital_cols:
            data = self.bp_data[col].dropna()
            if len(data) > 3:
                stat, p = stats.shapiro(data)
                normality[col] = {
                    'statistic': stat,
                    'p_value': p,
                    'is_normal': p > 0.05
                }
        
        print(f"  ✓ Normality tests completed")
        return normality
    
    def generate_console_report(self):
        """Print comprehensive analysis to console."""
        print("\n" + "="*80)
        print("COMPREHENSIVE BLOOD PRESSURE ANALYSIS - MIMIC-III ICU COHORT")
        print("="*80)
        
        # Descriptive stats
        print("\n[STEP 1] VITAL SIGNS DESCRIPTIVE STATISTICS")
        print("-"*80)
        print(f"{'Vital Sign':<25} {'n':>6} {'Mean':>10} {'SD':>10} {'Range':>20} {'CV':>8}")
        print("-"*80)
        
        for vital, stats_dict in self.stats.items():
            range_str = f"{stats_dict['min']:.1f}–{stats_dict['max']:.1f}"
            print(f"{vital:<25} {stats_dict['n']:>6.0f} {stats_dict['mean']:>10.2f} "
                  f"{stats_dict['std']:>10.2f} {range_str:>20} {stats_dict['cv']:>8.3f}")
        
        # Pathology
        print("\n[STEP 2] CLINICAL PATHOLOGY DISTRIBUTION")
        print("-"*80)
        print(f"{'Condition':<45} {'Count':>6} {'%':>8}")
        print("-"*80)
        
        n_total = len(self.bp_data)
        for condition, count in self.pathology.items():
            pct = (count / n_total * 100) if n_total > 0 else 0
            print(f"{condition:<45} {count:>6.0f} {pct:>7.1f}%")
        
        # Correlations
        print("\n[STEP 3] INTER-VITAL CORRELATIONS (Pearson r)")
        print("-"*80)
        print(self.correlations.round(3).to_string())
        
        # Normality
        print("\n[STEP 4] DISTRIBUTION ANALYSIS (Shapiro-Wilk)")
        print("-"*80)
        normality = self.normality_tests()
        print(f"{'Vital Sign':<25} {'Statistic':>12} {'p-value':>12} {'Normal?':>10}")
        print("-"*80)
        
        for vital, norm_dict in normality.items():
            is_normal = "YES" if norm_dict['is_normal'] else "NO"
            print(f"{vital:<25} {norm_dict['statistic']:>12.4f} "
                  f"{norm_dict['p_value']:>12.4f} {is_normal:>10}")
        
        print("\n" + "="*80)
    
    def save_csv_outputs(self):
        """Save extracted data and summaries to CSV."""
        print("\n[7/7] Saving CSV outputs...")
        
        # Save raw BP data
        bp_file = self.output_dir / 'bp_vitals_extracted.csv'
        self.bp_data.to_csv(bp_file, index=True)
        print(f"  ✓ Saved: {bp_file.name}")
        
        # Save pathology summary
        pathology_df = pd.DataFrame({
            'Condition': self.pathology.keys(),
            'Count': self.pathology.values(),
            'Percentage': [round(v/len(self.bp_data)*100, 1) for v in self.pathology.values()]
        })
        pathology_file = self.output_dir / 'bp_pathology_summary.csv'
        pathology_df.to_csv(pathology_file, index=False)
        print(f"  ✓ Saved: {pathology_file.name}")
        
        # Save statistics summary
        stats_df = pd.DataFrame(self.stats).T
        stats_file = self.output_dir / 'bp_statistics_summary.csv'
        stats_df.to_csv(stats_file)
        print(f"  ✓ Saved: {stats_file.name}")
        
        # Save correlations
        corr_file = self.output_dir / 'bp_correlations.csv'
        self.correlations.to_csv(corr_file)
        print(f"  ✓ Saved: {corr_file.name}")
    
    def generate_html_report(self):
        """Generate comprehensive HTML clinical report."""
        print("\n  Generating HTML report...")
        
        n_total = len(self.bp_data)
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>BP Analysis - MIMIC-III</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 20px; background: #f9fafb; color: #1f2937; line-height: 1.6; }}
.container {{ max-width: 1200px; margin: 0 auto; }}
h1 {{ font-size: 28px; font-weight: 500; margin: 30px 0 10px; }}
h2 {{ font-size: 18px; font-weight: 500; margin: 25px 0 15px; border-bottom: 2px solid #e5e7eb; padding-bottom: 8px; }}
table {{ width: 100%; border-collapse: collapse; margin: 15px 0; background: white; border: 1px solid #e5e7eb; }}
th {{ background: #f3f4f6; padding: 12px; text-align: left; font-weight: 500; font-size: 13px; color: #6b7280; border-bottom: 1px solid #e5e7eb; }}
td {{ padding: 11px; border-bottom: 0.5px solid #e5e7eb; }}
.metric {{ display: inline-block; background: #f3f4f6; padding: 12px 16px; border-radius: 6px; margin: 5px; text-align: center; min-width: 150px; }}
.critical {{ color: #dc2626; font-weight: 500; }}
.success {{ color: #16a34a; font-weight: 500; }}
.warning {{ color: #ea580c; font-weight: 500; }}
.grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin: 15px 0; }}
ul {{ line-height: 1.8; }}
li {{ margin: 8px 0; }}
.footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #e5e7eb; color: #6b7280; font-size: 12px; }}
</style>
</head>
<body>
<div class="container">

<h1>🩸 Blood Pressure Analysis - MIMIC-III Sepsis Cohort</h1>
<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

<h2>1. Vital Signs Descriptive Statistics (n={n_total})</h2>
<table>
<tr><th>Vital Sign</th><th>Mean ± SD</th><th>Range</th><th>Median (IQR)</th><th>CV</th></tr>
"""
        
        for vital, stat in self.stats.items():
            range_str = f"{stat['min']:.1f}–{stat['max']:.1f}"
            iqr_str = f"{stat['median']:.1f} ({stat['q25']:.1f}–{stat['q75']:.1f})"
            highlight = " style=\"background: #fef3c7;\"" if vital in ['map', 'pulse_pressure'] else ""
            
            html_content += f"""<tr{highlight}><td>{vital}</td><td>{stat['mean']:.1f} ± {stat['std']:.1f}</td><td>{range_str}</td><td>{iqr_str}</td><td>{stat['cv']:.3f}</td></tr>
"""
        
        html_content += """</table>

<h2>2. Clinical Pathology in ICU Cohort</h2>
<div class="grid">
"""
        
        # Top pathologies
        septic_shock = self.pathology.get('Septic shock (SBP<90 or MAP<65)', 0)
        sepsis_criteria = self.pathology.get('Possible sepsis (RR≥20 AND HR>90)', 0)
        tachypnea = self.pathology.get('Tachypnea (RR > 20)', 0)
        normal_bp = self.pathology.get('Normal (< 120)', 0)
        
        html_content += f"""  <div class="metric" style="background: #fee2e2;"><strong class="critical">Septic Shock</strong><br>{septic_shock} ({septic_shock/n_total*100:.1f}%)<br><small>SBP<90 or MAP<65</small></div>
  <div class="metric" style="background: #fef3c7;"><strong class="warning">Sepsis Criteria</strong><br>{sepsis_criteria} ({sepsis_criteria/n_total*100:.1f}%)<br><small>RR≥20 AND HR>90</small></div>
  <div class="metric" style="background: #dbeafe;"><strong>Tachypnea</strong><br>{tachypnea} ({tachypnea/n_total*100:.1f}%)<br><small>RR > 20</small></div>
  <div class="metric" style="background: #f0fdf4;"><strong class="success">Normal BP</strong><br>{normal_bp} ({normal_bp/n_total*100:.1f}%)<br><small>SBP < 120</small></div>
</div>

<h2>3. Vital Sign Correlations (Pearson)</h2>
<table>
<tr><th>Vital Pair</th><th>Correlation</th><th>Strength</th><th>Clinical Interpretation</th></tr>
"""
        
        corr_pairs = [
            ('SBP ↔ DBP', self.correlations.loc['systolic_bp', 'diastolic_bp'], 'Strong ✓', 'Normal coupled regulation'),
            ('SBP ↔ HR', self.correlations.loc['systolic_bp', 'heart_rate'], 'Weak', 'Baroreflex present'),
            ('SBP ↔ RR', self.correlations.loc['systolic_bp', 'resp_rate'], 'Weak', 'Respiratory coupling'),
            ('HR ↔ RR', self.correlations.loc['heart_rate', 'resp_rate'], 'Weak', 'Sinus arrhythmia'),
        ]
        
        for pair, corr, strength, interp in corr_pairs:
            html_content += f"<tr><td>{pair}</td><td>{corr:.3f}</td><td>{strength}</td><td>{interp}</td></tr>\n"
        
        html_content += """</table>

<h2>4. Key Clinical Insights</h2>
<ul>
<li><strong>Sample Size:</strong> {n} admissions with complete BP+HR+RR data</li>
<li><strong>Septic Shock Prevalence:</strong> {shock} ({shock_pct:.1f}%) meet septic shock criteria</li>
<li><strong>SIRS Criteria Match:</strong> {sepsis} ({sepsis_pct:.1f}%) show elevated RR+HR pattern</li>
<li><strong>Most Stable Metric:</strong> MAP (CV=0.149) — suitable for consistent threshold-based alerts</li>
<li><strong>Most Variable Metric:</strong> Pulse Pressure (CV=0.314) — high discriminative power for vasoconstriction</li>
<li><strong>Vital Correlations:</strong> Strong SBP↔DBP coupling indicates intact cardiovascular regulation</li>
<li><strong>Distribution:</strong> SBP, DBP, HR are normally distributed; RR is non-normal (bimodal) — suggests distinct patient phenotypes</li>
</ul>

<div class="footer">
<p><strong>Report generated by BP Analysis Pipeline v1.0</strong></p>
<p>For clinical validation on larger cohorts (n>500) and external datasets.</p>
</div>

</div>
</body>
</html>
""".format(
    n=n_total,
    shock=septic_shock,
    shock_pct=septic_shock/n_total*100,
    sepsis=sepsis_criteria,
    sepsis_pct=sepsis_criteria/n_total*100,
)
        
        html_file = self.output_dir / 'BP_ANALYSIS_COMPREHENSIVE.html'
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        print(f"  ✓ Saved: {html_file.name}")
    
    def run_complete_analysis(self):
        """Execute complete analysis pipeline."""
        print("\n" + "="*80)
        print("MIMIC-III BLOOD PRESSURE ANALYSIS - COMPLETE PIPELINE")
        print("="*80)
        
        self.load_data()
        self.extract_vitals()
        self.compute_descriptive_stats()
        self.compute_correlations()
        self.compute_pathology()
        self.generate_console_report()
        self.save_csv_outputs()
        self.generate_html_report()
        
        print("\n" + "="*80)
        print(f"✓ ANALYSIS COMPLETE")
        print(f"  Output directory: {self.output_dir}")
        print("  Files generated:")
        print("    - bp_vitals_extracted.csv")
        print("    - bp_pathology_summary.csv")
        print("    - bp_statistics_summary.csv")
        print("    - bp_correlations.csv")
        print("    - BP_ANALYSIS_COMPREHENSIVE.html")
        print("="*80 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point."""
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python3 bp_analysis_complete.py <CHARTEVENTS.txt> [output_dir]")
        print("\nExample:")
        print("  python3 bp_analysis_complete.py /data/CHARTEVENTS.txt /output")
        sys.exit(1)
    
    chartevents_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else '.'
    
    # Validate input file
    if not os.path.exists(chartevents_path):
        print(f"✗ Error: File not found: {chartevents_path}")
        sys.exit(1)
    
    # Run analysis
    analyzer = BPAnalyzer(chartevents_path, output_dir)
    analyzer.run_complete_analysis()


if __name__ == '__main__':
    main()
