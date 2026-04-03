"""
Density-field reconstruction utilities.

Main functions
--------------
* `compute_reconstruction`: Run reconstruction for cutsky geometry.
* `compute_box_reconstruction`: Run reconstruction for periodix boxes.
"""

import logging

import jax

from .tools import compute_fkp_effective_redshift
from .spectrum2_tools import prepare_jaxpower_particles


logger = logging.getLogger('reconstruction')


def compute_reconstruction(get_data_randoms, mattrs=None, mode='recsym', bias=2.0, smoothing_radius=15.):
    """
    Compute density field reconstruction using :mod:`jaxrecon`.

    Parameters
    ----------
    get_data_randoms : callable
        Functions that return dict of 'data', 'randoms' catalogs.
        See :func:`prepare_jaxpower_particles` for details.
    mattrs : dict, optional
        Mesh attributes to define the :class:`jaxpower.ParticleField` objects,
        'boxsize', 'meshsize' or 'cellsize', 'boxcenter'. If ``None``, default attributes are used.
    mode : {'recsym', 'reciso'}, optional
        Reconstruction mode. 'recsym' removes large-scale RSD from randoms, 'reciso' does not.
    bias : float, optional
        Linear bias of the tracer.
    smoothing_radius : float, optional
        Smoothing radius in Mpc/h for the density field.

    Returns
    -------
    data_positions_rec : np.ndarray
        Reconstructed data positions.
    randoms_positions_rec : np.ndarray
        Reconstructed randoms positions.
    """
    # Import reconstruction and density estimation tools from jaxrecon
    from jaxpower import create_sharding_mesh, FKPField
    from jaxrecon.zeldovich import IterativeFFTReconstruction, estimate_particle_delta

    # Use provided mesh attributes or empty dict as fallback
    mattrs = mattrs or {}
    # Set up distributed computation mesh across JAX devices
    with create_sharding_mesh(meshsize=mattrs.get('meshsize', None)):
        # Load and prepare particle catalogs (data and randoms) with reverse index for later exchange
        particles = prepare_jaxpower_particles(get_data_randoms, mattrs=mattrs, return_inverse=True)[0]

        # Create FKP field: delta(r) = n_data(r) - alpha * n_random(r), where alpha = N_data / N_random
        fkp = FKPField(particles['data'], particles['randoms'])
        # Clear particle data from memory
        del particles

        # Estimate overdensity field by smoothing FKP field with Gaussian kernel
        # This provides the initial density perturbation for reconstruction
        delta = estimate_particle_delta(fkp, smoothing_radius=smoothing_radius)

        # Load fiducial cosmology to compute growth rate
        # Growth rate f(z) = d ln D / d ln a is needed for Zeldovich reconstruction
        from cosmoprimo.fiducial import DESI
        cosmo = DESI()
        # Compute effective growth rate on the grid; order=1 is interpolation order
        growth_rate = compute_fkp_effective_redshift(fkp.data, order=1, cellsize=None, func_of_z=cosmo.growth_rate)

        # Run iterative FFT-based Zeldovich reconstruction
        # JIT-compile for performance; static args don't change across calls; donate_argnums for memory efficiency
        recon = jax.jit(IterativeFFTReconstruction, static_argnames=['los', 'halo_add', 'niterations'], donate_argnums=[0])(
            delta,
            growth_rate=growth_rate,  # Growth rate parameter for Zeldovich displacement field
            bias=bias,                # Linear bias to relate density to displacement
            los=None,                 # None = local line-of-sight; can also be 'x', 'y', 'z' or 3-vector
            halo_add=0                # Halo size in cells for distributed computation (0 = no halos needed)
        )

        # Extract reconstructed positions for data particles
        # Applies Zeldovich displacement field to move particles back to initial positions
        data_positions_rec = recon.read_shifted_positions(fkp.data.positions)

        # Validate reconstruction mode (recsym or reciso)
        assert mode in ['recsym', 'reciso']
        # RecSym mode: subtract RSD from randoms (use full displacement including RSD)
        # RecIso mode: use only isotropic displacement (no RSD removal) for randoms
        kwargs = {}
        if mode == 'reciso':
            # Use only isotropic displacement field (not including line-of-sight component)
            kwargs['field'] = 'disp'

        # Extract reconstructed positions for random particles
        randoms_positions_rec = recon.read_shifted_positions(fkp.randoms.positions, **kwargs)
        if jax.process_index() == 0:
            logger.info('Reconstruction finished.')

        # Reverse the MPI distribution to gather particles back to original process ranks
        # (data and randoms were distributed across processes for parallel reconstruction)
        data_positions_rec = fkp.data.exchange_inverse(data_positions_rec)
        randoms_positions_rec = fkp.randoms.exchange_inverse(randoms_positions_rec)
        if jax.process_index() == 0:
            logger.info('Exchange finished.')

    return data_positions_rec, randoms_positions_rec



def compute_box_reconstruction(get_data, mattrs=None, mode='recsym', zsnap=None, bias=2.0, smoothing_radius=15., nran=10, los='z'):
    """
    Compute density field reconstruction in box using :mod:`jaxrecon`.

    Parameters
    ----------
    get_data : callable
        Functions that return dict of 'data' catalogs.
        See :func:`prepare_jaxpower_particles` for details.
    mattrs : dict, optional
        Mesh attributes to define the :class:`jaxpower.ParticleField` objects,
        'boxsize', 'meshsize' or 'cellsize', 'boxcenter'.
    mode : {'recsym', 'reciso'}, optional
        Reconstruction mode. 'recsym' removes large-scale RSD from randoms, 'reciso' does not.
    zsnap : float, optional
        Snapshot redshift, used to estimate the growth rate.
    bias : float, optional
        Linear bias of the tracer.
    smoothing_radius : float, optional
        Smoothing radius in Mpc/h for the density field.
    nran : int, optional
        Number of randoms per data particle to generate for reconstruction.
    los : str or array-like, optional
        Line-of-sight specification ('x', 'y', 'z',).

    Returns
    -------
    data_positions_rec : np.ndarray
        Reconstructed data positions.
    randoms_positions_rec : np.ndarray
        Reconstructed randoms positions.
    """
    # Import reconstruction tools and uniform random particle generator
    from jaxpower import create_sharding_mesh, ParticleField, generate_uniform_particles
    from jaxrecon.zeldovich import IterativeFFTReconstruction, estimate_particle_delta

    # Use provided mesh attributes or empty dict as fallback
    mattrs = mattrs or {}
    # Set up distributed computation mesh across JAX devices
    with create_sharding_mesh(meshsize=mattrs.get('meshsize', None)):
        # Load and prepare data particles with reverse index for later exchange
        # No randoms provided (periodic box case)
        data = prepare_jaxpower_particles(get_data, mattrs=mattrs, return_inverse=True)[0]['data']
        # Extract mesh attributes from loaded data (boxsize, cellsize, meshsize, etc.)
        mattrs = data.attrs

        # Estimate overdensity field from data particles
        # Density is computed as particle counts on mesh, smoothed with Gaussian kernel
        delta = estimate_particle_delta(data, smoothing_radius=smoothing_radius)

        # Load fiducial cosmology (DESI)
        from cosmoprimo.fiducial import DESI
        cosmo = DESI()
        # Get growth rate f(z) = d ln D / d ln a at snapshot redshift
        growth_rate = cosmo.growth_rate(zsnap)

        # Run iterative FFT-based Zeldovich reconstruction on density field
        recon = jax.jit(IterativeFFTReconstruction, static_argnames=['los', 'halo_add', 'niterations'], donate_argnums=[0])(
            delta,
            growth_rate=growth_rate,  # Growth rate at snapshot epoch
            bias=bias,                # Linear bias to relate density perturbation to displacement
            los=los,                  # Line-of-sight direction: 'x', 'y', or 'z' (fixed axis for periodic box)
            halo_add=0                # Halo size in cells (0 = no halos needed for periodic box)
        )

        # Extract reconstructed positions for data particles
        data_positions_rec = recon.read_shifted_positions(data.positions)

        # Generate uniform random particles to represent the underlying matter distribution
        # nran * data.size total randoms for accurate density estimation
        # seed=(42, 'index') uses reproducible seeding based on process index
        # exchange=True distributes randoms across MPI processes for parallel processing
        # return_inverse=True prepares for later gathering back to original ranks
        randoms = generate_uniform_particles(
            mattrs,
            size=nran * data.size,           # Total number of random particles to generate
            seed=(42, 'index'),              # Reproducible seed for random number generator
            exchange=True,                   # Distribute across processes via MPI
            backend='mpi',                   # Use MPI for distributed generation
            return_inverse=True              # Store inverse mapping for exchange_inverse
        )

        # Validate reconstruction mode (recsym or reciso)
        assert mode in ['recsym', 'reciso']
        # RecSym mode: subtract RSD from randoms (use full displacement including RSD)
        # RecIso mode: use only isotropic displacement (no RSD removal) for randoms
        kwargs = {}
        if mode == 'reciso':
            # Use only isotropic displacement field (excludes line-of-sight component from RSD)
            kwargs['field'] = 'disp'

        # Extract reconstructed positions for random particles
        randoms_positions_rec = recon.read_shifted_positions(randoms.positions, **kwargs)
        if jax.process_index() == 0:
            logger.info('Reconstruction finished.')

        # Reverse the MPI distribution to gather particles back to original process ranks
        # This ensures data and randoms are on the same processes as input
        data_positions_rec = data.exchange_inverse(data_positions_rec)
        randoms_positions_rec = randoms.exchange_inverse(randoms_positions_rec)
        if jax.process_index() == 0:
            logger.info('Exchange finished.')

    return data_positions_rec, randoms_positions_rec