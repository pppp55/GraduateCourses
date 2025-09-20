import coptpy as cp
from coptpy import COPT

# Create COPT environment
env = cp.Envr()

# Create COPT model
model = env.createModel("q2")

# Add variables
vertex = ["s", "m1", "m2", "m3", "a1", "b1", "c1", "a2", "b2", "c2", "a3", "b3", "c3","t"]
edge = dict()
for v in vertex:
    edge[v] = dict()
edge["s"]["m1"] = [8, model.addVar(lb=0, ub=180, name="s_m1")]
edge["s"]["m2"] = [10, model.addVar(lb=0, ub=200, name="s_m2")]
edge["s"]["m3"] = [11, model.addVar(lb=0, ub=150, name="s_m3")]
edge["m1"]["a1"] = [4, model.addVar(lb=0, name="m1_a1")]
edge["m1"]["b1"] = [6, model.addVar(lb=0, name="m1_b1")]
edge["m1"]["c1"] = [8, model.addVar(lb=0, name="m1_c1")]
edge["m1"]["m2"] = [1, model.addVar(lb=0, ub=100, name="m1_m2")]
edge["m2"]["a2"] = [4, model.addVar(lb=0, name="m2_a2")]
edge["m2"]["b2"] = [6, model.addVar(lb=0, name="m2_b2")]
edge["m2"]["c2"] = [8, model.addVar(lb=0, name="m2_c2")]
edge["m2"]["m3"] = [1, model.addVar(lb=0, ub=100, name="m2_m3")]
edge["m3"]["a3"] = [4, model.addVar(lb=0, name="m3_a3")]
edge["m3"]["b3"] = [6, model.addVar(lb=0, name="m3_b3")]
edge["m3"]["c3"] = [8, model.addVar(lb=0, name="m3_c3")]
edge["a1"]["t"] = [-15, model.addVar(lb=0, ub=50, name="a1_t")]
edge["b1"]["t"] = [-20, model.addVar(lb=0, ub=100, name="b1_t")]
edge["c1"]["t"] = [-13, model.addVar(lb=0, ub=75, name="c1_t")]
edge["a2"]["t"] = [-19, model.addVar(lb=0, ub=75, name="a2_t")]
edge["b2"]["t"] = [-16, model.addVar(lb=0, ub=150, name="b2_t")]
edge["c2"]["t"] = [-21, model.addVar(lb=0, ub=75, name="c2_t")]
edge["a3"]["t"] = [-15, model.addVar(lb=0, ub=20, name="a3_t")]
edge["b3"]["t"] = [-18, model.addVar(lb=0, ub=80, name="b3_t")]
edge["c3"]["t"] = [-18, model.addVar(lb=0, ub=50, name="c3_t")]
edge["s"]["t"] = [0, model.addVar(lb=0, name="s_t")]

bs = model.addVar(lb=0, ub=530, name="bs")

# Add constraints
for v in vertex:
    model.addConstr(sum(edge[v][j][1] for j in vertex if j in edge[v]) - sum(edge[i][v][1] for i in vertex if v in edge[i]) == (bs if v == "s" else -bs if v == "t" else 0))

# Set objective function
model.setObjective(sum(edge[v1][v2][0] * edge[v1][v2][1] for v1 in vertex for v2 in vertex if v2 in edge[v1]), sense=COPT.MINIMIZE)

# Set parameter
model.setParam(COPT.Param.TimeLimit, 10.0)

# Solve the model
model.solve()

# Analyze solution
if model.status == COPT.OPTIMAL:
    print(f"Objective value: {model.objval:.6f}")
    allvars = model.getVars()
    print("Variable solution:")
    for var in allvars:
        print(f" {var.name} = {var.x}")
