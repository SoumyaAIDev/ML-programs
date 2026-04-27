import pandas as pd
import numpy as np

np.random.seed(42)

data = []

for _ in range(200):
    interest = np.random.uniform(10, 50)
    late_fee = np.random.uniform(100, 1500)
    annual_fee = np.random.uniform(0, 5000)
    billing_cycle = np.random.randint(20, 60)
    min_payment = np.random.uniform(2, 20)
    disclosure = np.random.choice([0, 1])  

    
    compliant = 1
    if interest > 40:
        compliant = 0
    if late_fee > 1000:
        compliant = 0
    if min_payment < 5:
        compliant = 0
    if disclosure == 0:
        compliant = 0

    data.append([
        interest, late_fee, annual_fee,
        billing_cycle, min_payment,
        disclosure, compliant
    ])

df = pd.DataFrame(data, columns=[
    "interest_rate", "late_fee", "annual_fee",
    "billing_cycle", "min_payment",
    "disclosure", "label"
])

df.to_csv("dataset.csv", index=False)

print("Dataset created!")