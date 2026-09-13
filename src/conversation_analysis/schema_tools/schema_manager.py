# Ported from chatgpt-archive-clean/schema/schema_manager.py @ f9b2b636 (master) during the conversation-analysis
# consolidation (2026-09-12). See docs/PROVENANCE.md.
# Changes vs original: moved to quarantined schema_tools subpackage; flat imports fixed to package-relative

# schema_manager.py
import re
try:
    import sqlparse  # requirements-optional.txt
except ImportError:
    sqlparse = None
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class SchemaComponent:
    name: str
    type: str  # table, column, index, etc.
    definition: str
    dependencies: List[str]
    metadata: Dict


class SchemaManager:
    def __init__(self):
        self.schema_components = {}
        self.dependency_graph = {}

    def parse_schema(self, sql_file: str) -> Dict[str, SchemaComponent]:
        """Parse SQL schema using regex and sqlparse"""
        if sqlparse is None:
            raise RuntimeError("The 'sqlparse' package is required (see requirements-optional.txt)")
        parsed = sqlparse.parse(sql_file)
        components = {}

        for statement in parsed:
            # Extract table definitions, constraints, etc.
            if self._is_table_definition(statement):
                table = self._parse_table(statement)
                components[table.name] = table

        self.schema_components = components
        self.dependency_graph = self.analyze_dependencies()
        return components

    def _is_table_definition(self, statement) -> bool:
        """Check if the statement is a CREATE TABLE statement"""
        return statement.get_type() == 'CREATE' and 'TABLE' in statement.tokens[0].value.upper()

    def _parse_table(self, statement) -> SchemaComponent:
        """Parse a CREATE TABLE statement into SchemaComponent"""
        tokens = statement.tokens
        table_name = None
        definition = str(statement)
        dependencies = []

        for token in tokens:
            if token.ttype is None and token.is_group:
                for subtoken in token.tokens:
                    if subtoken.ttype is None and subtoken.value.upper().startswith('TABLE'):
                        table_name = subtoken.get_name()
                        break

        # Find REFERENCES for dependencies
        refs = re.findall(r'REFERENCES\s+(\w+)', definition, re.IGNORECASE)
        dependencies.extend(refs)

        return SchemaComponent(
            name=table_name,
            type='table',
            definition=definition,
            dependencies=dependencies,
            metadata={}
        )

    def analyze_dependencies(self) -> Dict[str, List[str]]:
        """Build dependency graph based on REFERENCES"""
        dependencies = {}
        for name, component in self.schema_components.items():
            dependencies[name] = component.dependencies
        return dependencies
