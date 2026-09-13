# Ported from ai-conversation-analyzer/claude_db_advanced-2.py @ a616deaf (master) during the conversation-analysis
# consolidation (2026-09-12). See docs/PROVENANCE.md.
# Changes vs original: newer of the claude_db pair (enhanced heatmap, __init__ exclusion); heavy imports (python-louvain, pyvis, anthropic) made lazy/guarded -> requirements-optional.txt

import csv
import os
import time
from collections import defaultdict
from typing import Dict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import community  # python-louvain (requirements-optional.txt)
except ImportError:
    community = None
try:
    from pyvis.network import Network  # requirements-optional.txt
except ImportError:
    Network = None


def _get_anthropic_client():
    """Lazily create the Anthropic client (requires requirements-optional.txt)."""
    try:
        from anthropic import Anthropic
    except ImportError:
        raise RuntimeError("The 'anthropic' package is required (see requirements-optional.txt)")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
    return Anthropic(api_key=api_key)


def _require(name, value):
    if value is None:
        raise RuntimeError(f"'{name}' is required for this function (see requirements-optional.txt)")
    return value


def load_function_counts(file_path: str) -> Dict[str, Dict[str, int]]:
    """Load function counts from CSV file, excluding __init__."""
    function_counts = defaultdict(lambda: defaultdict(int))
    with open(file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for function, conversation, count in reader:
            if function != '__init__':  # Exclude __init__
                function_counts[conversation][function] = int(count)
    return function_counts


def build_function_graph(function_counts: Dict[str, Dict[str, int]]) -> nx.Graph:
    """Build a graph of function co-occurrences, excluding __init__."""
    G = nx.Graph()
    for functions in function_counts.values():
        for f1 in functions:
            for f2 in functions:
                if f1 != f2 and f1 != '__init__' and f2 != '__init__':
                    if G.has_edge(f1, f2):
                        G[f1][f2]['weight'] += 1
                    else:
                        G.add_edge(f1, f2, weight=1)
    return G


def analyze_function_communities(G: nx.Graph) -> Dict[str, int]:
    """Detect communities in the function graph."""
    return _require('python-louvain', community).best_partition(G)


def create_enhanced_heatmap(G: nx.Graph, communities: Dict[str, int], output_file: str, top_n: int = 30,
                            threshold: float = 0.1):
    """Create an improved enhanced heatmap of function co-occurrences."""
    adj_matrix = nx.to_pandas_adjacency(G)

    # Select top N functions by degree, excluding __init__
    top_functions = sorted([(n, d) for n, d in G.degree() if n != '__init__'], key=lambda x: x[1], reverse=True)[:top_n]
    top_function_names = [f[0] for f in top_functions]

    # Create a submatrix with only the top functions
    sub_matrix = adj_matrix.loc[top_function_names, top_function_names]

    # Apply logarithmic scaling
    sub_matrix = np.log1p(sub_matrix)

    # Apply threshold
    sub_matrix[sub_matrix < threshold] = 0

    # Sort functions by their communities
    community_sorted_funcs = sorted(top_function_names, key=lambda x: communities[x])
    sub_matrix = sub_matrix.loc[community_sorted_funcs, community_sorted_funcs]

    plt.figure(figsize=(20, 16))

    # Create the heatmap
    g = sns.heatmap(sub_matrix, cmap="YlGnBu", annot=False, square=True, cbar_kws={'label': 'Log(Co-occurrences + 1)'})

    # Rotate x-axis labels for better readability
    plt.setp(g.get_xticklabels(), rotation=45, ha='right')
    plt.setp(g.get_yticklabels(), rotation=0)

    # Increase font size
    plt.tick_params(axis='both', which='major', labelsize=8)

    # Add title
    plt.title("Enhanced Function Co-occurrence Heatmap (Top {} Functions, Excluding __init__)".format(top_n),
              fontsize=16)

    # Add note about scaling and threshold
    plt.figtext(0.5, 0.01,
                f"Note: Values are log-scaled and thresholded at {threshold}. __init__ functions are excluded.",
                ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def analyze_function_complexity(G: nx.Graph) -> pd.DataFrame:
    """Analyze function complexity based on graph metrics."""
    complexity = pd.DataFrame({
        'Degree': dict(G.degree()),
        'Betweenness': nx.betweenness_centrality(G),
        'Closeness': nx.closeness_centrality(G)
    })
    complexity['Complexity Score'] = complexity.mean(axis=1)
    return complexity.sort_values('Complexity Score', ascending=False)


def create_interactive_graph(G: nx.Graph, communities: Dict[str, int], output_file: str):
    """Create an interactive graph visualization."""
    net = _require('pyvis', Network)(notebook=True, height="750px", width="100%", bgcolor="#222222", font_color="white")
    net.from_nx(G)
    for node in net.nodes:
        node["group"] = communities[node["id"]]
        node["size"] = G.degree(node["id"]) * 3
        node["title"] = f"Function: {node['id']}<br>Community: {node['group']}<br>Connections: {G.degree(node['id'])}"
    net.show_buttons(filter_=['physics'])
    net.save_graph(output_file)


def analyze_function_usage(function_counts: Dict[str, Dict[str, int]]) -> pd.DataFrame:
    """Analyze function usage across conversations."""
    function_usage = defaultdict(int)
    for functions in function_counts.values():
        for function in functions:
            function_usage[function] += 1

    df = pd.DataFrame.from_dict(function_usage, orient='index', columns=['Count'])
    df = df.sort_values('Count', ascending=False)
    return df


def print_top_functions(df: pd.DataFrame, n: int = 10):
    """Print the top N most used functions."""
    print(f"\nTop {n} Most Used Functions:")
    print(df.head(n).to_string())


def ai_analyze_functions(G: nx.Graph, communities: Dict[str, int], usage_df: pd.DataFrame,
                         complexity_df: pd.DataFrame) -> str:
    """Use AI to analyze function relationships and provide insights."""
    context = f"""
    Graph Summary (excluding __init__ functions):
    - Number of nodes (functions): {G.number_of_nodes()}
    - Number of edges (connections): {G.number_of_edges()}
    - Number of communities: {len(set(communities.values()))}

    Top 5 Most Used Functions:
    {usage_df.head().to_string()}

    Top 5 Most Complex Functions:
    {complexity_df.head().to_string()}

    Please analyze this data and provide insights on:
    1. The overall structure of the codebase based on function relationships, noting that __init__ functions have been excluded.
    2. Potential areas for refactoring or optimization.
    3. Any interesting patterns or clusters of functions.
    4. Recommendations for improving code organization or architecture.
    5. How the exclusion of __init__ functions might affect our understanding of the codebase structure.
    """

    message = _get_anthropic_client().messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=4000,
        messages=[
            {"role": "user", "content": context}
        ]
    )

    return message.content[0].text


def interactive_ai_query(G: nx.Graph, communities: Dict[str, int], usage_df: pd.DataFrame, complexity_df: pd.DataFrame):
    """Allow interactive querying of the AI about the function analysis."""
    print("\nYou can now ask questions about the function analysis.")
    print("Type 'exit' to quit the interactive mode.")

    while True:
        query = input("\nEnter your question: ")
        if query.lower() == 'exit':
            break

        context = f"""
        Based on the function analysis data:
        - Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges
        - There are {len(set(communities.values()))} communities
        - Top used function: {usage_df.index[0]}
        - Most complex function: {complexity_df.index[0]}

        User question: {query}

        Please provide a concise and informative answer.
        """

        message = _get_anthropic_client().messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=4000,
            messages=[
                {"role": "user", "content": context}
            ]
        )

        print("\nAI Response:", message.content[0].text)


def main():
    input_file = 'function_counts.csv'
    heatmap_file = 'enhanced_heatmap.png'
    interactive_graph_file = 'function_graph.html'

    start_time = time.time()
    function_counts = load_function_counts(input_file)
    print(f"Loaded data from {len(function_counts)} conversations")

    G = build_function_graph(function_counts)
    print(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    communities = analyze_function_communities(G)
    print(f"Detected {len(set(communities.values()))} communities")

    create_enhanced_heatmap(G, communities, heatmap_file)
    print(f"Created enhanced heatmap: {heatmap_file}")

    create_interactive_graph(G, communities, interactive_graph_file)
    print(f"Created interactive graph: {interactive_graph_file}")

    usage_df = analyze_function_usage(function_counts)
    print("\nTop 10 Most Used Functions:")
    print(usage_df.head(10).to_string())

    complexity_df = analyze_function_complexity(G)
    print("\nTop 10 Most Complex Functions:")
    print(complexity_df.head(10).to_string())

    print("\nPerforming AI analysis...")
    ai_insights = ai_analyze_functions(G, communities, usage_df, complexity_df)
    print("\nAI Insights:")
    print(ai_insights)

    interactive_ai_query(G, communities, usage_df, complexity_df)

    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
