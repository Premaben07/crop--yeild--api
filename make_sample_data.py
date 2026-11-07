"""Creates SYNTHETIC data so the pipeline and CI can run end to end.
Replace data/crop_data.csv with your real dataset for the actual model."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
n = 500
df = pd.DataFrame({
    "rainfall": rng.uniform(300, 1500, n),
    "temperature": rng.uniform(15, 40, n),
    "soil_ph": rng.uniform(5, 8, n),
    "nitrogen": rng.uniform(20, 150, n),
})
df["yield"] = (0.004*df.rainfall - 0.05*(df.temperature-27)**2 + 0.02*df.nitrogen
               - 0.3*(df.soil_ph-6.5)**2 + 5 + rng.normal(0, 0.4, n))
df.to_csv("data/crop_data.csv", index=False)
print("Wrote data/crop_data.csv", df.shape)
