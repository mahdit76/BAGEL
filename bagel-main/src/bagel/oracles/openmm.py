from .base import Oracle, OracleResult
from typing import List, Type
from ..chain import Chain
from biotite.structure import AtomArray
import numpy as np
import tempfile
import os
from typing import Dict
class OpenMMMDResult(OracleResult):
    """
    Stores statistics from OpenMM MD simulation and analysis.
    """
    input_chains: List[Chain]
    structure: AtomArray  # final structure
    trajectory: np.ndarray  # shape (n_frames, n_atoms, 3)
    sasa: float
    gyration: float
    rmsd: float
    rmsf: np.ndarray

    model_config = {"arbitrary_types_allowed": True}

    def save_attributes(self, filepath):
        np.savetxt(str(filepath.with_suffix('.sasa')), [self.sasa], fmt='%.6f', header='SASA (A^2)')
        np.savetxt(str(filepath.with_suffix('.gyration')), [self.gyration], fmt='%.6f', header='Gyration (A)')
        np.savetxt(str(filepath.with_suffix('.rmsd')), [self.rmsd], fmt='%.6f', header='RMSD (A)')
        np.savetxt(str(filepath.with_suffix('.rmsf')), self.rmsf, fmt='%.6f', header='RMSF (A)')


class OpenMMMDOracle(Oracle):
    """
    Oracle that runs OpenMM MD, analyzes SASA, Gyration, RMSD, RMSF, and returns results for BAGEL.
    """
    result_class: Type[OpenMMMDResult] = OpenMMMDResult

    def __init__(self, forcefield: str = 'amber14-all.xml', water_model: str = 'amber14/tip3p.xml', n_steps: int = 10000, report_interval: int = 100, **kwargs):
        self.forcefield = forcefield
        self.water_model = water_model
        self.n_steps = n_steps
        self.report_interval = report_interval
        self.kwargs = kwargs

    def predict(self, chains: List[Chain]) -> OpenMMMDResult:
        if openmm is None or app is None or unit is None:
            raise ImportError('OpenMM is not installed. Please install openmm and openmm.app.')
        from biotite.structure.io.pdb import PDBFile
        import tempfile
        # Convert chains to AtomArray (assume single chain for now)
        atom_arrays = [chain.to_atom_array() for chain in chains]
        atom_array = atom_arrays[0] if len(atom_arrays) == 1 else atom_arrays[0].stack(atom_arrays)
        pdb_path = None
        try:
            fd, pdb_path = tempfile.mkstemp(suffix=".pdb")
            os.close(fd)  # Close the file descriptor immediately
            from biotite.structure.io.pdb import PDBFile as BiotitePDBFile
            # Filter out non-standard residues (keep only standard amino acids)
            from biotite.structure import filter_amino_acids
            mask = filter_amino_acids(atom_array)
            filtered_array = atom_array[mask]
            with open(pdb_path, "w") as pdb_file:
                pdb_obj = BiotitePDBFile()
                pdb_obj.set_structure(filtered_array)
                pdb_obj.write(pdb_file)
            # Now the file is closed and can be safely read by OpenMM
            pdb = app.PDBFile(pdb_path)
            forcefield = app.ForceField(self.forcefield, self.water_model)
            # Add missing hydrogens using Modeller
            modeller = app.Modeller(pdb.topology, pdb.positions)
            modeller.addHydrogens(forcefield)
            system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
            integrator = openmm.LangevinIntegrator(300*unit.kelvin, 1.0/unit.picoseconds, 0.002*unit.picoseconds)
            simulation = app.Simulation(modeller.topology, system, integrator)
            simulation.context.setPositions(modeller.positions)

            # Set up reporters for trajectory
            import mdtraj as md
            with tempfile.TemporaryDirectory() as tmpdir:
                dcd_path = os.path.join(tmpdir, 'traj.dcd')
                simulation.reporters.append(app.DCDReporter(dcd_path, self.report_interval))
                # Use a temp file for StateDataReporter
                with tempfile.NamedTemporaryFile(mode="w", suffix=".log") as state_log:
                    simulation.reporters.append(app.StateDataReporter(state_log, self.report_interval, step=True, potentialEnergy=True, temperature=True))
                    simulation.minimizeEnergy()
                    simulation.step(self.n_steps)

                # Ensure all reporters are closed before loading trajectory
                del simulation
                import gc
                gc.collect()

                # Now load trajectory and analyze
                traj = md.load(dcd_path, top=pdb_path)
                sasa = float(traj.shrake_rupley().mean())
                gyration = float(traj.radius_of_gyration().mean())
                rmsd = float(traj.rmsd(traj, 0).mean())
                rmsf = traj.rmsf(traj, 0)
                # Final structure
                final_xyz = traj.xyz[-1] * 10  # MDTraj uses nm, convert to Angstrom
                minimized = atom_array.copy()
                minimized.coord = final_xyz
                return OpenMMMDResult(
                    input_chains=chains,
                    structure=minimized,
                    trajectory=traj.xyz * 10,  # shape (n_frames, n_atoms, 3), in Angstrom
                    sasa=sasa,
                    gyration=gyration,
                    rmsd=rmsd,
                    rmsf=rmsf,
                )
        finally:
            if pdb_path is not None and os.path.exists(pdb_path):
                os.remove(pdb_path)
"""
OpenMM Oracle for BAGEL

MIT License

Copyright (c) 2025
"""

from .base import Oracle, OracleResult
from ..chain import Chain
from biotite.structure import AtomArray
from typing import List, Any, Type
import numpy as np
from pydantic import field_validator

# OpenMM imports (assume openmm and openmm.app are installed)
try:
    import openmm
    import openmm.app as app
    import openmm.unit as unit
except ImportError:
    openmm = None
    app = None
    unit = None

import logging
logger = logging.getLogger(__name__)

class OpenMMResult(OracleResult):
    """
    Stores statistics from the OpenMM simulation.
    """
    input_chains: List[Chain]
    structure: AtomArray  # minimized structure
    energy: float         # minimized potential energy (kJ/mol)

    model_config = {"arbitrary_types_allowed": True}

    @field_validator('energy')
    def validate_energy(cls, v):
        assert isinstance(v, float), 'Energy must be a float.'
        return v

    def save_attributes(self, filepath):
        # Optionally save energy and structure
        np.savetxt(str(filepath.with_suffix('.energy')), [self.energy], fmt='%.6f', header='energy (kJ/mol)')

class OpenMMOracle(Oracle):
    """
    Oracle that uses OpenMM to minimize a structure and return energy and coordinates.
    """
    result_class: Type[OpenMMResult] = OpenMMResult

    def __init__(self, forcefield: str = 'amber14-all.xml', water_model: str = 'amber14/tip3p.xml', **kwargs):
        self.forcefield = forcefield
        self.water_model = water_model
        self.kwargs = kwargs

    def predict(self, chains: List[Chain]) -> OpenMMResult:
        if openmm is None or app is None or unit is None:
            raise ImportError('OpenMM is not installed. Please install openmm and openmm.app.')
        # Convert chains to PDB string (using biotite)
        from biotite.structure.io.pdb import PDBFile
        import tempfile
        # Create AtomArray from chains (assume single chain for now)
        # TODO: support multimers
        atom_arrays = [chain.to_atom_array() for chain in chains]
        atom_array = atom_arrays[0] if len(atom_arrays) == 1 else atom_arrays[0].stack(atom_arrays)
        with tempfile.NamedTemporaryFile("w+", suffix=".pdb") as pdb_buf:
            PDBFile.write(pdb_buf, atom_array)
            pdb_buf.seek(0)
            pdb = app.PDBFile(pdb_buf)
            forcefield = app.ForceField(self.forcefield, self.water_model)
            system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
            integrator = openmm.LangevinIntegrator(300*unit.kelvin, 1.0/unit.picoseconds, 0.002*unit.picoseconds)
            simulation = app.Simulation(pdb.topology, system, integrator)
            simulation.context.setPositions(pdb.positions)
            # Minimize energy
            simulation.minimizeEnergy()
            state = simulation.context.getState(getPositions=True, getEnergy=True)
            positions = state.getPositions(asNumpy=True)
            energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            # Convert positions back to AtomArray
            coords = np.array([[a.x, a.y, a.z] for a in positions])
            minimized = atom_array.copy()
            minimized.coord = coords
            return OpenMMResult(input_chains=chains, structure=minimized, energy=energy)
