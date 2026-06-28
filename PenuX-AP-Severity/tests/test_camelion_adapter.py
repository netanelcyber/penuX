"""Tests for the Camelion HIS adapter and FHIR endpoint."""
import pytest
from api.camelion_adapter import bundle_to_admission_input, camelion_json_to_admission_input
from api.fhir_schemas import FHIRBundle


def _obs_entry(loinc: str, value: float, unit: str = "mg/dL") -> dict:
    return {
        "resource": {
            "resourceType": "Observation",
            "status": "final",
            "code": {
                "coding": [{"system": "http://loinc.org", "code": loinc}]
            },
            "valueQuantity": {"value": value, "unit": unit},
        }
    }


def _patient_entry(birth_date: str = "1960-01-01", gender: str = "male") -> dict:
    return {
        "resource": {
            "resourceType": "Patient",
            "birthDate": birth_date,
            "gender": gender,
        }
    }


class TestFHIRAdapter:
    def test_vitals_mapped(self):
        bundle = FHIRBundle(
            entry=[
                _patient_entry("1964-06-01", "female"),
                _obs_entry("8867-4", 102, "bpm"),   # heart_rate
                _obs_entry("8310-5", 38.7, "Cel"),  # temperature
                _obs_entry("6690-2", 17.5, "10*9/L"),  # wbc
                _obs_entry("1988-5", 210.0),  # crp
            ]
        )
        result = bundle_to_admission_input(bundle)
        assert result.heart_rate == pytest.approx(102.0)
        assert result.temperature == pytest.approx(38.7)
        assert result.wbc == pytest.approx(17.5)
        assert result.crp == pytest.approx(210.0)
        assert result.sex == "F"

    def test_age_from_birthdate(self):
        bundle = FHIRBundle(entry=[_patient_entry("1964-01-01", "male")])
        result = bundle_to_admission_input(bundle)
        assert result.age is not None
        assert 55 < result.age < 70  # born 1964, current year ~2026

    def test_unknown_loinc_ignored(self):
        bundle = FHIRBundle(
            entry=[_obs_entry("9999-9", 42.0)]
        )
        result = bundle_to_admission_input(bundle)
        assert result.heart_rate is None

    def test_empty_bundle(self):
        bundle = FHIRBundle(entry=[])
        result = bundle_to_admission_input(bundle)
        assert result.age is None


class TestCamelionNativeAdapter:
    def test_hebrew_keys(self):
        payload = {
            "גיל": 62,
            "מין": "זכר",
            "דופק": 108,
            "חום": 38.9,
            "כדוריות דם לבנות": 19.0,
            "crp": 250,
            "קראטינין": 1.5,
        }
        result = camelion_json_to_admission_input(payload)
        assert result.age == pytest.approx(62.0)
        assert result.sex == "M"
        assert result.heart_rate == pytest.approx(108.0)
        assert result.temperature == pytest.approx(38.9)
        assert result.wbc == pytest.approx(19.0)
        assert result.crp == pytest.approx(250.0)
        assert result.creatinine == pytest.approx(1.5)

    def test_english_keys(self):
        payload = {"wbc": 14.5, "crp": 180.0, "ldh": 460.0, "ast": 290.0}
        result = camelion_json_to_admission_input(payload)
        assert result.wbc == pytest.approx(14.5)
        assert result.ldh == pytest.approx(460.0)

    def test_nested_labs_section(self):
        payload = {
            "גיל": 55,
            "labs": {"crp": 195.0, "ldh": 370.0},
            "vitals": {"דופק": 95},
        }
        result = camelion_json_to_admission_input(payload)
        assert result.age == pytest.approx(55.0)
        assert result.crp == pytest.approx(195.0)
        assert result.heart_rate == pytest.approx(95.0)

    def test_identifiers_not_mapped(self):
        payload = {"patient_id": "IL123456", "תעודת_זהות": "999999999", "crp": 100.0}
        result = camelion_json_to_admission_input(payload)
        # Identifiers should not appear as clinical fields
        assert result.crp == pytest.approx(100.0)
