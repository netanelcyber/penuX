# EHR System Integration Guide

PenuX-AP-Severity prediction API supports three integration standards to work with any hospital information system. This guide shows how to integrate with each.

---

## 1. HL7 v2.x Integration (Epic, Cerner, OpenEMR, VistA, Allscripts)

**Endpoint:** `POST /hl7/predict`

**Supported EHR systems:**
- Epic Clarity / EHR
- Cerner PowerChart
- OpenEMR
- Medidata
- Allscripts
- VistA (VA system)
- Any system that exports HL7 v2.x segments

### Message Format

Send an HL7 v2.x message (line-separated segments, `|` field delimiters):

```
MSH|^~\&|EpicCare|Hospital|Receiver||||20240618
PID|1||MRN123456||Doe^John||19640101|M
OBX|1|NM|2160-0^Creatinine||1.5|mg/dL|||F
OBX|2|NM|6690-2^WBC||18.2|10*9/L|||F
OBX|3|NM|1988-5^CRP||210|mg/L|||F
OBX|4|NM|14804-9^LDH||480|U/L|||F
OBX|5|NM|8867-4^Heart Rate||105|bpm|||F
```

### Segment Reference

| Segment | Field | Purpose |
|---------|-------|---------|
| **MSH** | 1 | Message header (not parsed for prediction) |
| **PID** | 7 | Date of birth (YYYYMMDD) → age |
| **PID** | 8 | Sex (M/F/O/U) → sex |
| **OBX** | 3 | Observation ID (LOINC^text^system) |
| **OBX** | 5 | Numeric value (e.g., `1.5`) |

### Supported Observation Codes

#### LOINC Codes (Standard)

| LOINC | Description | Field |
|-------|-------------|-------|
| 8867-4 | Heart rate | heart_rate |
| 8480-6 | Systolic BP | systolic_bp |
| 8462-4 | Diastolic BP | diastolic_bp |
| 9279-1 | Respiratory rate | respiratory_rate |
| 8310-5 | Temperature | temperature |
| 59408-5 | SpO2 | spo2 |
| 6690-2 | WBC | wbc |
| 20570-8 | Hematocrit | hematocrit |
| 1988-5 | CRP | crp |
| 3094-0 | BUN | bun |
| 2160-0 | Creatinine | creatinine |
| 17861-6 | Calcium | calcium |
| 2345-7 | Glucose | glucose |
| 14804-9 | LDH | ldh |
| 1920-8 | AST | ast |
| 1742-6 | ALT | alt |
| 1751-7 | Albumin | albumin |
| 2571-8 | Triglycerides | triglycerides |

#### Vendor-Specific LIS Codes (Fallback)

Epic codes: `WBC`, `HCT`, `CREAT`, `BUN`, `CA`, `GLU`, `LDH`, `AST`, `ALT`, `ALB`, `TRIG`, `CRP`

Example:
```
OBX|1|NM|WBC||16.5|10*9/L|||F
OBX|2|NM|CREAT||1.8|mg/dL|||F
```

### Example with cURL

```bash
cat > hl7_message.txt << 'EOF'
MSH|^~\&|Epic|Hospital|Receiver||||20240618
PID|1||MRN123456||Doe^John||19640101|M
OBX|1|NM|2160-0^Creatinine||1.5|mg/dL|||F
OBX|2|NM|6690-2^WBC||18.2|10*9/L|||F
OBX|3|NM|1988-5^CRP||210|mg/L|||F
OBX|4|NM|14804-9^LDH||480|U/L|||F
EOF

curl -X POST http://localhost:8000/hl7/predict \
  -H "Content-Type: text/plain" \
  --data-binary @hl7_message.txt
```

**Response:**
```json
{
  "severe_ap_probability": 0.7234,
  "risk_group": "high",
  "threshold_used": 0.5,
  "model_version": "0.1.0",
  "warning": "This is a research prototype only...",
  "error": null
}
```

---

## 2. FHIR R4 Integration (Standards-Based, All EHRs)

**Endpoint:** `POST /fhir/predict`

**Supported EHR systems:**
- Epic (via FHIR STU3 / R4 gateway)
- Cerner (via FHIR interface)
- OpenEMR (native FHIR support)
- Any FHIR-compliant system (Apple HealthKit, Google Health Connect, etc.)

### Message Format

Send a FHIR R4 Bundle with Patient + Observation resources:

```json
{
  "resourceType": "Bundle",
  "type": "collection",
  "entry": [
    {
      "resource": {
        "resourceType": "Patient",
        "birthDate": "1964-01-01",
        "gender": "male"
      }
    },
    {
      "resource": {
        "resourceType": "Observation",
        "status": "final",
        "code": {
          "coding": [{
            "system": "http://loinc.org",
            "code": "2160-0",
            "display": "Creatinine"
          }]
        },
        "valueQuantity": {
          "value": 1.5,
          "unit": "mg/dL"
        }
      }
    },
    {
      "resource": {
        "resourceType": "Observation",
        "status": "final",
        "code": {
          "coding": [{
            "system": "http://loinc.org",
            "code": "6690-2",
            "display": "WBC"
          }]
        },
        "valueQuantity": {
          "value": 18.2,
          "unit": "10*9/L"
        }
      }
    }
  ]
}
```

### Example with cURL

```bash
curl -X POST http://localhost:8000/fhir/predict \
  -H "Content-Type: application/json" \
  -d @bundle.json
```

**Response:**
```json
{
  "resourceType": "RiskAssessment",
  "status": "final",
  "prediction": [
    {
      "outcome": {
        "coding": [{
          "system": "http://snomed.info/sct",
          "code": "67630002",
          "display": "Severe acute pancreatitis"
        }],
        "text": "Severe Acute Pancreatitis"
      },
      "probabilityDecimal": 0.7234,
      "qualitativeRisk": {
        "coding": [{
          "system": "http://snomed.info/sct",
          "code": "723507007",
          "display": "High risk"
        }]
      },
      "rationale": "This is a research prototype only..."
    }
  ],
  "note": [
    {"text": "This is a research prototype only..."}
  ]
}
```

---

## 3. Camelion (קמיליון) Israeli HIS Integration

### Option A: Native JSON Endpoint

**Endpoint:** `POST /camelion/predict`

Accepts flat Hebrew/English key-value JSON (as exported by Camelion HL7 gateway):

```json
{
  "encounter_id": "ENC-2024-00123",
  "גיל": 62,
  "מין": "זכר",
  "דופק": 108,
  "חום": 38.9,
  "כדוריות דם לבנות": 19.0,
  "crp": 250,
  "קראטינין": 1.5,
  "גלוקוז": 230,
  "סידן": 7.8,
  "ldh": 480,
  "ast": 310,
  "אוריאה": 30
}
```

**Response:**
```json
{
  "encounter_id": "ENC-2024-00123",
  "severe_ap_probability": 0.7891,
  "risk_group": "high",
  "fields_used": ["age", "sex", "heart_rate", "temperature", "wbc", "crp", "creatinine", "glucose", "calcium", "ldh", "ast", "bun"],
  "missing_fields": ["bmi", "systolic_bp", "diastolic_bp", "respiratory_rate", "spo2", "hematocrit", "alt", "albumin", "triglycerides"],
  "model_version": "0.1.0",
  "warning": "This is a research prototype only...",
  "error": null
}
```

### Option B: FHIR R4 Endpoint

Camelion also supports FHIR R4 via the `/fhir/predict` endpoint (see Section 2).

---

## Privacy and Security

**Patient Identifier Handling:**

All endpoints follow strict privacy protocols:

| Integration | Identifiers Accepted | Storage | Logging | Downstream |
|-------------|---------------------|---------|---------|------------|
| HL7 v2.x | MRN, SSN, Name (PID) | ❌ Discarded | ❌ Not logged | ❌ Not forwarded |
| FHIR R4 | Teudat Zehut, MRN (identifier) | ❌ Discarded | ❌ Not logged | ❌ Not forwarded |
| Camelion JSON | patient_id, encounter_id (JSON keys) | ❌ Discarded | ❌ Not logged | ❌ Not forwarded |

**Clinical Values Only:**

Only numerical/categorical clinical values are passed to the prediction pipeline:
- Age, sex, temperature, heart rate, labs (WBC, CRP, creatinine, etc.)
- No personal identifiers ever transmitted downstream

---

## Integration Checklist

- [ ] Choose integration standard (HL7 v2, FHIR R4, or Camelion native)
- [ ] Configure EHR system export to match segment/resource format
- [ ] Test with sample message/bundle via cURL
- [ ] Deploy PenuX-AP-Severity API (set `PENUX_AP_MODEL_PATH` env var)
- [ ] Route EHR POST requests to chosen endpoint
- [ ] Monitor `/health` endpoint for model availability
- [ ] Log predictions for audit trail (if required by regulation)
- [ ] Document clinical workflow and user training

---

## FAQ

**Q: Which EHR should I choose?**
A: Use HL7 v2.x for Epic, Cerner, OpenEMR, Allscripts, VistA. Use FHIR R4 if your EHR is FHIR-native. Use Camelion native JSON if integrating with קמיליון.

**Q: Can I mix LOINC and vendor codes?**
A: Yes. The HL7 parser supports both in the same message. Unknown codes are logged and skipped.

**Q: What if a lab result is missing?**
A: The API returns prediction with `missing_fields` list. Prediction accuracy may degrade with sparse data.

**Q: Are patient identifiers logged?**
A: No. MRN, names, SSNs are extracted for routing correlation only and discarded immediately. Only clinical values enter the model.

**Q: Does this work in production?**
A: No. This is a research-only API. Do not use for clinical decision-making. See `RESEARCH_WARNING` in all responses.
