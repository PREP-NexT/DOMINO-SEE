Network Analysis
===============

This guide explains how to analyze networks generated from event data using dominosee.

Introduction
------------

Once you've generated networks from your hydroclimatic event data, dominosee provides various tools to analyze these networks to uncover patterns, identify key nodes, and understand the structure of interconnected extreme events.

Basic Network Metrics
--------------------

dominosee enables you to calculate various network metrics to understand the properties of your event networks:

.. code-block:: python

    import dominosee as ds
    import matplotlib.pyplot as plt
    
    # Assuming you have a network generated as shown in the network_generation guide
    # Calculate basic network metrics
    
    # Node degree (number of connections per node)
    # Note: this is example code and should be adjusted based on your actual API
    degrees = ds.calculate_degree(network)
    
    # Plot degree distribution
    plt.figure(figsize=(8, 6))
    plt.hist(degrees, bins=10, alpha=0.7)
    plt.title('Node Degree Distribution')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    
    # Calculate network density
    density = ds.calculate_density(network)
    print(f"Network density: {density:.4f}")
    
    # Calculate clustering coefficient
    clustering = ds.calculate_clustering(network)
    print(f"Average clustering coefficient: {clustering.mean():.4f}")

Centrality Measures
------------------

Centrality measures help identify the most important nodes (locations) in your event network:

.. code-block:: python

    # Calculate different centrality measures
    # Note: this is example code and should be adjusted based on your actual API
    
    # Degree centrality
    degree_centrality = ds.calculate_degree_centrality(network)
    
    # Betweenness centrality
    betweenness_centrality = ds.calculate_betweenness_centrality(network)
    
    # Eigenvector centrality
    eigenvector_centrality = ds.calculate_eigenvector_centrality(network)
    
    # Identify top nodes by centrality
    top_nodes = ds.get_top_nodes(eigenvector_centrality, n=5)
    print("Top 5 nodes by eigenvector centrality:")
    for node, value in top_nodes.items():
        print(f"  {node}: {value:.4f}")

Community Detection
------------------

Finding communities in your network can reveal groups of locations that experience similar patterns of extreme events:

.. code-block:: python

    # Detect communities in the network
    # Note: this is example code and should be adjusted based on your actual API
    communities = ds.detect_communities(network, method='louvain')
    
    # Get the number of communities
    n_communities = len(set(communities.values()))
    print(f"Number of communities detected: {n_communities}")
    
    # Visualize communities
    ds.plot_network_communities(network, communities)

Network Comparison
-----------------

Compare networks across different event types or time periods to understand how patterns of extreme events change:

.. code-block:: python

    # Assuming you have multiple networks for different event types
    # Note: this is example code and should be adjusted based on your actual API
    
    # Calculate similarity between networks
    similarity = ds.calculate_network_similarity(network1, network2)
    print(f"Network similarity: {similarity:.4f}")
    
    # Compare network metrics across different networks
    metrics_comparison = ds.compare_network_metrics([network1, network2, network3])
    
    # Plot comparison of key metrics
    metrics_comparison.plot.bar()
    plt.title('Comparison of Network Metrics')
    plt.grid(True)
    plt.show()

Temporal Network Analysis
------------------------

Analyze how your event networks evolve over time:

.. code-block:: python

    # Generate time-windowed networks
    # Note: this is example code and should be adjusted based on your actual API
    time_windows = [(0, 30), (30, 60), (60, 90)]
    temporal_networks = []
    
    for start, end in time_windows:
        # Select data for this time window
        window_data = event_da.isel(time=slice(start, end))
        
        # Generate network for this window
        window_network = ds.create_network(window_data, threshold=0.5)
        temporal_networks.append(window_network)
    
    # Analyze network evolution
    evolution = ds.analyze_network_evolution(temporal_networks)
    
    # Plot evolution of key metrics over time
    ds.plot_network_evolution(evolution)

Advanced Statistical Analysis
---------------------------

Perform statistical tests on your networks to validate findings:

.. code-block:: python

    # Generate null model networks for comparison
    # Note: this is example code and should be adjusted based on your actual API
    null_models = ds.generate_null_models(network, n_models=100)
    
    # Compare real network to null models
    significance = ds.test_network_significance(network, null_models)
    
    # Print significance results
    for metric, p_value in significance.items():
        print(f"{metric}: p-value = {p_value:.4f}")
