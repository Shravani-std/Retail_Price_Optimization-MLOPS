import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Nodes
G.add_node("Pipeline\n(tags: Tutorial Example,\nExperiment[cascade])")
G.add_node("Step 1\n(tags: Experiment)")
G.add_node("Step 2\n(tags: Experiment)")
G.add_node("Step 3\n(tags: Experiment)")
G.add_node("Artifact A\n(tags: Experiment)")
G.add_node("Artifact B\n(tags: Experiment)")
G.add_node("Run Metadata\n(tags: Experiment)")

# Edges to represent flow
G.add_edges_from([
    ("Pipeline\n(tags: Tutorial Example,\nExperiment[cascade])", "Step 1\n(tags: Experiment)"),
    ("Pipeline\n(tags: Tutorial Example,\nExperiment[cascade])", "Step 2\n(tags: Experiment)"),
    ("Pipeline\n(tags: Tutorial Example,\nExperiment[cascade])", "Step 3\n(tags: Experiment)"),
    ("Step 1\n(tags: Experiment)", "Artifact A\n(tags: Experiment)"),
    ("Step 2\n(tags: Experiment)", "Artifact B\n(tags: Experiment)"),
    ("Pipeline\n(tags: Tutorial Example,\nExperiment[cascade])", "Run Metadata\n(tags: Experiment)")
])

# Layout
pos = nx.spring_layout(G, seed=42)

# Draw
plt.figure(figsize=(12,8))
nx.draw(G, pos, with_labels=True, node_size=4000, node_color="lightblue", font_size=9, font_weight="bold", arrows=True)
plt.title("Tag Cascade Visualization", fontsize=14, fontweight="bold")
plt.show()
