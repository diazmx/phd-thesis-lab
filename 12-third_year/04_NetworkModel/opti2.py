# Función para preparar los datos de las métricas y frentes de Pareto
def prepare_parallel_coordinates_data(pareto_fronts, graph_names):
    data = []
    id_to_name = {i: graph_names[i] for i in range(len(graph_names))}

    for level, front in enumerate(pareto_fronts):
        for graph_id, metrics in front:
            row = {
                "ID": graph_id,
                "Pareto Level": level + 1,
                **metrics
            }
            data.append(row)

    return pd.DataFrame(data), id_to_name

# Función para visualizar las métricas en coordenadas paralelas
def visualize_parallel_coordinates(df, id_to_name):
    # Configurar el estilo de Seaborn
    sns.set_theme(style="whitegrid")

    # Crear figura
    plt.figure(figsize=(15, 8))

    # Lista de colores para cada frente de Pareto
    colors = sns.color_palette("tab10", len(df["Pareto Level"].unique()))

    # Trazar líneas por cada grafo
    for idx, row in df.iterrows():
        metrics = row.drop(["ID", "Pareto Level"])
        pareto_level = int(row["Pareto Level"])  # Convertir nivel de Pareto a entero
        plt.plot(metrics.index, metrics.values, label=f"ID {row['ID']}", color=colors[pareto_level - 1], linewidth=1.5)

    # Ajustar etiquetas y leyendas
    plt.title("Parallel Coordinates Plot: Metrics Across Pareto Levels", fontsize=14)
    plt.xlabel("Metrics", fontsize=12)
    plt.ylabel("Metric Value", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True)

    # Leyenda de niveles de Pareto
    for level in df["Pareto Level"].unique():
        plt.plot([], [], color=colors[int(level) - 1], label=f"Pareto Level {int(level)}")
    plt.legend(title="Pareto Level", loc="upper left", bbox_to_anchor=(1, 1))

    # Mostrar gráfica
    plt.tight_layout()
    plt.show()

    # Mostrar tabla de identificadores
    print("\nGraph Identifiers:")
    for graph_id, graph_name in id_to_name.items():
        print(f"ID {graph_id}: {graph_name}")
