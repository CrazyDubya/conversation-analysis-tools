# Ported from ai-conversation-analyzer/graph.py @ a616deaf (master) during the conversation-analysis
# consolidation (2026-09-12). See docs/PROVENANCE.md.
# Changes vs original: rewrapped as build_graph(input, output) with argparse; graphviz binary still required at render time

import json

import graphviz

# Load data
def build_graph(input_path, output='conversation_flowchart'):
    with open(input_path) as file:
        data = json.load(file)

    conversation = data[0]
    dot = graphviz.Digraph(comment='Conversation Flow')

    for node_id, node in conversation['mapping'].items():
        message = node.get('message')
        if message:
            author = message['author']['role']
            content = ' '.join(message['content'].get('parts', []))
            dot.node(node_id, f"{author}: {content}")

        parent_id = node.get('parent')
        if parent_id:
            dot.edge(parent_id, node_id)

    dot.render(output, format='png')
    return output + '.png'


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Render a conversation mapping as a flowchart.')
    ap.add_argument('--input', required=True, help='Path to a parsed conversation JSON file')
    ap.add_argument('--output', default='conversation_flowchart')
    args = ap.parse_args()
    print(build_graph(args.input, args.output))


if __name__ == '__main__':
    main()
