# Ported from chatgpt-archive-clean/schema/llm_analyzer.py @ f9b2b636 (master) during the conversation-analysis
# consolidation (2026-09-12). See docs/PROVENANCE.md.
# Changes vs original: moved to quarantined schema_tools subpackage; flat imports fixed to package-relative

# llm_analyzer.py
try:
    import openai  # requirements-optional.txt
except ImportError:
    openai = None
from typing import Dict
from .schema_manager import SchemaComponent


def _require_openai():
    if openai is None:
        raise RuntimeError("The 'openai' package is required (see requirements-optional.txt)")
    return openai


class LLMAnalyzer:
    def __init__(self, api_key: str):
        _require_openai().api_key = api_key

    def analyze_schema_quality(self, component: SchemaComponent) -> Dict:
        """Use LLM to analyze schema quality"""
        prompt = f"""
        Analyze this SQL schema component for best practices:
        {component.definition}

        Consider:
        1. Naming conventions
        2. Index usage
        3. Constraint design
        4. Performance implications

        Provide a detailed analysis and suggest improvements.
        """
        response = _require_openai().ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a SQL schema best practices advisor."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content

    def validate_change(self, change: Dict) -> Dict:
        """Validate a proposed schema change"""
        prompt = f"""
        Analyze the following schema change for safety and potential impacts:

        Change Type: {change['change_type']}
        Object Type: {change['object_type']}
        Original: {change.get('original_name', 'N/A')}
        New: {change.get('new_name', 'N/A')}
        Definition: {change['definition']}

        Please analyze:
        1. Data loss risks
        2. Foreign key integrity
        3. Application compatibility
        4. Performance implications
        5. Backup requirements

        Provide reasoning and a confidence score between 0 and 1.
        """
        response = _require_openai().ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a database schema safety validator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content
