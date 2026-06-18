"""Tests for HL7 v2.x EHR system adapter."""
import pytest
from api.hl7_adapter import hl7_message_to_admission_input, hl7_batch_results_to_admission_input


class TestHL7MessageParser:
    def test_simple_message(self):
        """Parse a complete HL7 v2.x message (PID + OBX segments)."""
        message = """MSH|^~\\&|Epic|Hospital|Receiver||||20240618
PID|1||MRN123456^^Hospital||Doe^John||19640101|M
PV1|1|I|ICU^1^A|H
OBX|1|NM|2160-0^Creatinine||1.5|mg/dL|||F
OBX|2|NM|6690-2^WBC||18.2|10*9/L|||F
OBX|3|NM|1988-5^CRP||210|mg/L|||F
OBX|4|NM|14804-9^LDH||480|U/L|||F"""
        result = hl7_message_to_admission_input(message)
        assert result.age is not None
        assert 61 < result.age < 63  # born 1964, current year ~2026
        assert result.sex == "M"
        assert result.creatinine == pytest.approx(1.5)
        assert result.wbc == pytest.approx(18.2)
        assert result.crp == pytest.approx(210.0)
        assert result.ldh == pytest.approx(480.0)

    def test_minimal_message(self):
        """Parse a minimal HL7 message (only some OBX, no PID)."""
        message = """MSH|^~\\&
OBX|1|NM|2160-0^Creatinine||2.1|mg/dL|||F
OBX|2|NM|2345-7^Glucose||320|mg/dL|||F"""
        result = hl7_message_to_admission_input(message)
        assert result.age is None
        assert result.creatinine == pytest.approx(2.1)
        assert result.glucose == pytest.approx(320.0)

    def test_multiple_vitals(self):
        """Parse vitals from OBX segments."""
        message = """PID|1||MRN||Doe^Jane||19800615|F
OBX|1|NM|8867-4^Heart Rate||105|bpm|||F
OBX|2|NM|8480-6^SystolicBP||145|mmHg|||F
OBX|3|NM|8462-4^DiastolicBP||92|mmHg|||F
OBX|4|NM|8310-5^Temperature||38.9|Cel|||F
OBX|5|NM|9279-1^RespiratoryRate||24|/min|||F"""
        result = hl7_message_to_admission_input(message)
        assert result.heart_rate == pytest.approx(105.0)
        assert result.systolic_bp == pytest.approx(145.0)
        assert result.diastolic_bp == pytest.approx(92.0)
        assert result.temperature == pytest.approx(38.9)
        assert result.respiratory_rate == pytest.approx(24.0)

    def test_epic_lis_codes(self):
        """Parse Epic-specific LIS codes (vendor fallback)."""
        message = """OBX|1|NM|WBC||16.5|10*9/L|||F
OBX|2|NM|CREAT||1.8|mg/dL|||F
OBX|3|NM|LDH||450|U/L|||F"""
        result = hl7_message_to_admission_input(message)
        assert result.wbc == pytest.approx(16.5)
        assert result.creatinine == pytest.approx(1.8)
        assert result.ldh == pytest.approx(450.0)

    def test_missing_fields(self):
        """Handle malformed OBX segments gracefully."""
        message = """OBX|1|NM|2160-0^Creatinine||1.5|mg/dL
OBX|2|NM|||invalid
OBX|3|NM|6690-2^WBC||19.0|10*9/L"""
        result = hl7_message_to_admission_input(message)
        # Should parse valid segments and ignore invalid ones
        assert result.creatinine == pytest.approx(1.5)
        assert result.wbc == pytest.approx(19.0)

    def test_batch_parsing(self):
        """Parse HL7 segments submitted separately (streaming)."""
        pid = "PID|1||MRN789||Patient^Test||19700315|M"
        obx = [
            "OBX|1|NM|2160-0^Creatinine||1.2|mg/dL|||F",
            "OBX|2|NM|2345-7^Glucose||280|mg/dL|||F",
            "OBX|3|NM|6690-2^WBC||15.0|10*9/L|||F",
        ]
        result = hl7_batch_results_to_admission_input(pid, obx)
        assert result.age is not None
        assert 55 < result.age < 57  # born 1970, current year ~2026
        assert result.sex == "M"
        assert result.creatinine == pytest.approx(1.2)
        assert result.glucose == pytest.approx(280.0)
        assert result.wbc == pytest.approx(15.0)

    def test_empty_message(self):
        """Handle empty or whitespace-only messages."""
        result = hl7_message_to_admission_input("")
        assert result.age is None
        assert result.creatinine is None

    def test_unknown_obs_codes_ignored(self):
        """Unknown observation codes are logged and skipped."""
        message = """OBX|1|NM|9999-9^UnknownTest||42.0|unit|||F
OBX|2|NM|2160-0^Creatinine||1.5|mg/dL|||F"""
        result = hl7_message_to_admission_input(message)
        assert result.creatinine == pytest.approx(1.5)
        # Unknown code should not appear
        assert not hasattr(result, "unknown_test")
