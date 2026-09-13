# Ported from chatgpt-archive-clean/schema/safeguard.py @ f9b2b636 (master) during the conversation-analysis
# consolidation (2026-09-12). See docs/PROVENANCE.md.
# Changes vs original: moved to quarantined schema_tools subpackage; flat imports fixed to package-relative

# safeguard.py
from typing import List, Dict, Tuple
from .change_validator import SchemaChangeValidator
from .llm_analyzer import LLMAnalyzer


class SchemaSafeguard:
    def __init__(self, validator: SchemaChangeValidator):
        self.validator = validator

    def validate_changes(self, changes: List[Dict]) -> Tuple[bool, List[Dict]]:
        """Validate each detected change with LLM"""
        validated_changes = []
        for change in changes:
            analysis = self.validator.llm.validate_change(change)
            # Extract confidence score using regex (assuming LLM returns it in a consistent format)
            score_match = re.search(r'confidence score.*?(\d\.\d+)', analysis, re.IGNORECASE)
            score = float(score_match.group(1)) if score_match else 0.0
            change['llm_analysis'] = analysis
            change['confidence_score'] = score
            validated_changes.append(change)

        # Check if any change exceeds risk threshold
        all_safe = all(change['confidence_score'] >= self.validator.risk_threshold for change in validated_changes)
        return all_safe, validated_changes
