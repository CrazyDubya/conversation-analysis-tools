# Ported from chatgpt-archive-clean/schema/main.py @ f9b2b636 (master) during the conversation-analysis
# consolidation (2026-09-12). See docs/PROVENANCE.md.
# Changes vs original: moved to quarantined schema_tools subpackage; flat imports fixed to package-relative

# main.py
import sys
from .schema_manager import SchemaManager
from .llm_analyzer import LLMAnalyzer
from .change_validator import SchemaChangeValidator
from .safeguard import SchemaSafeguard


def load_file(filepath: str) -> str:
    with open(filepath, 'r') as file:
        return file.read()


def main(original_schema_path: str, proposed_schema_path: str, llm_api_key: str):
    # Load schemas
    original_schema = load_file(original_schema_path)
    proposed_schema = load_file(proposed_schema_path)

    # Initialize components
    schema_manager = SchemaManager()
    llm_analyzer = LLMAnalyzer(api_key=llm_api_key)
    validator = SchemaChangeValidator(schema_manager, llm_analyzer)
    safeguard = SchemaSafeguard(validator)

    # Detect changes
    changes = validator.detect_changes(original_schema, proposed_schema)
    print(f"Detected {len(changes)} changes.")

    # Validate changes
    all_safe, validated_changes = safeguard.validate_changes(changes)

    if all_safe:
        print("All changes are safe to apply.")
        # Here you would apply the changes to the database
    else:
        print("Some changes may be unsafe:")
        for change in validated_changes:
            if change['confidence_score'] < validator.risk_threshold:
                print(
                    f"- {change['change_type']} {change['object_type']} {change.get('original_name')} -> {change.get('new_name')}")
                print(f"  Analysis: {change['llm_analysis']}")
                print(f"  Confidence Score: {change['confidence_score']}\n")
        # Here you might prompt for manual review or abort the operation


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python main.py <original_schema.sql> <proposed_schema.sql> <llm_api_key>")
        sys.exit(1)

    original_schema_path = sys.argv[1]
    proposed_schema_path = sys.argv[2]
    llm_api_key = sys.argv[3]

    main(original_schema_path, proposed_schema_path, llm_api_key)
