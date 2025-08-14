import bagel as bg
from bagel.oracles import OpenMMMDOracle

# Create the chain from a real PDB file and chain ID
chain = bg.Chain.from_pdb("example_protein.pdb", "A")  # Update filename and chain ID as needed

# Instantiate the OpenMMMDOracle
oracle = OpenMMMDOracle(n_steps=1000, report_interval=100)  # Short MD for test

# Run the oracle
result = oracle.predict([chain])

# Print results
print("SASA:", result.sasa)
print("Gyration:", result.gyration)
print("RMSD:", result.rmsd)
print("RMSF:", result.rmsf)
print("Trajectory shape:", result.trajectory.shape)