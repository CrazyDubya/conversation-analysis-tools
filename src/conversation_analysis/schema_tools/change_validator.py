# Ported from chatgpt-archive-clean/schema/change_validator.py @ f9b2b636 (master) during the conversation-analysis
# consolidation (2026-09-12). See docs/PROVENANCE.md.
# Changes vs original: moved to quarantined schema_tools subpackage; flat imports fixed to package-relative

# change_validator.py
import re
from typing import List, Tuple, Dict
from .schema_manager import SchemaManager, SchemaComponent
from .llm_analyzer import LLMAnalyzer


class SchemaChangeValidator:
    def __init__(self, schema_manager: SchemaManager, llm_analyzer: LLMAnalyzer, risk_threshold: float = 0.8):
        self.schema_manager = schema_manager
        self.llm = llm_analyzer
        self.risk_threshold = risk_threshold

    def detect_changes(self, original_schema: str, proposed_schema: str) -> List[Dict]:
        """Detect and categorize all schema changes"""
        original_components = self.schema_manager.parse_schema(original_schema)
        proposed_components = self.schema_manager.parse_schema(proposed_schema)

        changes = []

        # Detect additions and modifications
        for name, component in proposed_components.items():
            if name not in original_components:
                changes.append({
                    "change_type": "ADD",
                    "object_type": "table",
                    "original_name": None,
                    "new_name": name,
                    "definition": component.definition
                })
            else:
                if component.definition != original_components[name].definition:
                    changes.append({
                        "change_type": "MODIFY",
                        "object_type": "table",
                        "original_name": name,
                        "new_name": name,
                        "definition": component.definition
                    })

        # Detect deletions
        for name in original_components:
            if name not in proposed_components:
                changes.append({
                    "change_type": "DELETE",
                    "object_type": "table",
                    "original_name": name,
                    "new_name": None,
                    "definition": f"DROP TABLE {name};"
                })

        # Detect renames (simple heuristic based on similarity)
        # This can be enhanced with more sophisticated methods
        for name in original_components:
            if name not in proposed_components:
                for new_name in proposed_components:
                    if self._is_rename(original_components[name], proposed_components[new_name]):
                        changes.append({
                            "change_type": "RENAME",
                            "object_type": "table",
                            "original_name": name,
                            "new_name": new_name,
                            "definition": f"ALTER TABLE {name} RENAME TO {new_name};"
                        })
                        break

        return changes

    def _is_rename(self, original: SchemaComponent, proposed: SchemaComponent) -> bool:
        """Heuristic to determine if a table has been renamed"""
        original_columns = set(re.findall(r'\b\w+\b', original.definition))
        proposed_columns = set(re.findall(r'\b\w+\b', proposed.definition))
        common = original_columns.intersection(proposed_columns)
        similarity = len(common) / max(len(original_columns), len(proposed_columns))
        return similarity > 0.5  # Threshold can be adjusted
