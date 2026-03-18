import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ===============================
# 1. Cargar dataset
# ===============================

input_file = "rule_usage.csv"

df = pd.read_csv(input_file)

# Validación básica
required_cols = {"rule_id", "granted_accesses", "num_conditions"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"El archivo debe contener las columnas: {required_cols}")

# ===============================
# 2. Crear carpeta de salida
# ===============================

output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# Guardar copia del dataset (opcional)
df.to_csv(os.path.join(output_dir, "rules_dataset_copy.csv"), index=False)

# ===============================
# 3. Histograma
# ===============================

plt.figure()
plt.hist(df["granted_accesses"], bins=30)
plt.title("Distribution of Granted Accesses")
plt.xlabel("Granted Accesses")
plt.ylabel("Frequency")
plt.savefig(os.path.join(output_dir, "histogram.png"))
plt.close()

# ===============================
# 4. Histograma en log
# ===============================

plt.figure()
plt.hist(np.log1p(df["granted_accesses"]), bins=30)
plt.title("Log Distribution of Granted Accesses")
plt.xlabel("log(1 + granted_accesses)")
plt.ylabel("Frequency")
plt.savefig(os.path.join(output_dir, "log_histogram.png"))
plt.close()

# ===============================
# 5. Long-tail plot
# ===============================

sorted_values = df["granted_accesses"].sort_values(ascending=False).values

plt.figure()
plt.plot(sorted_values)
plt.title("Long-tail Distribution of Granted Accesses")
plt.xlabel("Rule Rank")
plt.ylabel("Granted Accesses")
plt.savefig(os.path.join(output_dir, "long_tail.png"))
plt.close()

# ===============================
# 6. Boxplot
# ===============================

plt.figure()
plt.boxplot(df["granted_accesses"], vert=True)
plt.title("Boxplot of Granted Accesses")
plt.ylabel("Granted Accesses")
plt.savefig(os.path.join(output_dir, "boxplot.png"))
plt.close()

# ===============================
# 7. Resumen estadístico (útil para análisis)
# ===============================

summary = df["granted_accesses"].describe()
summary.to_csv(os.path.join(output_dir, "summary_stats.csv"))

# ===============================
# 8. Mensaje final
# ===============================

import pandas as pd
import matplotlib.pyplot as plt
import os

# ===============================
# 1. Cargar datos
# ===============================

df = pd.read_csv("rule_usage.csv")

# Ordenar por rule_id (importante)
df = df.sort_values(by="rule_id")

# ===============================
# 2. Crear carpeta
# ===============================

output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# ===============================
# 3. Gráfica: Rule ID vs Granted Accesses
# ===============================

plt.figure()
plt.plot(df["rule_id"], df["granted_accesses"], marker='o', linestyle='-')
plt.title("Granted Accesses per Rule")
plt.xlabel("Rule ID")
plt.ylabel("Granted Accesses")
plt.savefig(os.path.join(output_dir, "rule_vs_accesses.png"))
plt.close()

print("Proceso completado correctamente.")
print(f"Archivo leído: {input_file}")
print(f"Resultados en carpeta: {output_dir}/")