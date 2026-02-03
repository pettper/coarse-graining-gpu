# GPU Accelerated Coarse Graining using JAX

## Required input data
np = number of particles
ng = number of gridpoints
nc = number of contacts

We need a gridpoints buffer:
gridpoints, size ng x 3.

We need the following buffers with data for the granular particles:
Particle positions, size np x 3.
Particle velocity, size np x 3.
Particle displacements, size np x 3. (This can be derived from the particle positions if a particle id to index buffer is also stored.)
Particle masses, size np.

We need the following buffers with data for the particle contacts:
Contact force in a local frame defined by a normal vector and two tangent vectors, size nc x 3.
Contact positions, size nc x 3.
Contact normal vectors, size nc x 3.
Contact tangent vector u, size nc x 3.
Contact tangent vector v, size nc x 3.

The diameter of the particles also needs to be known. The current implementation does only support particles of the same diameter. However, introducing a small random variation is not expected to give incorrect results. It is only used to determine the distance vector between two particles involved in a contact, and it therefore only affects the calculation of the contact stress. If one also stores the ID-buffers for the particles involved in a contact, then one could exactly calculate the distance vector using the position buffer.

If the implementation were to be extended, then it should also include:
Contact ID particle 1, size nc x 3.
Contact ID particle 2, size nc x 3. 
