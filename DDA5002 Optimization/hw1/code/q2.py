import coptpy as cp
from coptpy import COPT

# Create COPT environment
env = cp.Envr()

# Create COPT model
model = env.createModel("q2")

# Add variables
m = model.addVar(lb=0, name="m")
b = model.addVar(lb=0, name="b")
y = [model.addVar(lb=0, name=f"y_{i}") for i in range(5)]

p = [10, 8, 13, 15, 9]
d = [60, 55, 75, 80, 64]

# Add constraints
for i in range(5):
    model.addConstr(d[i] - (m * p[i] + b) <= y[i])
    model.addConstr(d[i] - (m * p[i] + b) >= -y[i])

# Set objective function
model.setObjective(sum(y), sense=COPT.MINIMIZE)

# Set parameter
model.setParam(COPT.Param.TimeLimit, 10.0)

# Solve the model
model.solve()

# Analyze solution
if model.status == COPT.OPTIMAL:
    print(f"Objective value: {model.objval:.6f}")
    print(f"Variable solution: m = {m.x:.4f}, b = {b.x:.4f}")
