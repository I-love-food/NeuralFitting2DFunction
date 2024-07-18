from sampler import *
from scipy.stats import qmc

rng = np.random.default_rng()
engine = qmc.PoissonDisk(d=2, radius=0.01, seed=rng)
samples = engine.fill_space()

span = [[-5, 5], [-5, 5]]
samples[:, 0] = span[0][0] + (span[0][1] - span[0][0]) * samples[:, 0]
samples[:, 1] = span[1][0] + (span[1][1] - span[1][0]) * samples[:, 1]

# span = [[-1, 1], [-1, 1]]
# gen = poisson_disk(r=0.02, span=span)
# samples = gen.sample()
# span[0][0] -= 0.01
# span[0][1] += 0.01
# span[1][0] -= 0.01
# span[1][1] += 0.01
# horizontal = list(
#     map(
#         lambda x1_x2: [x1_x2[0], span[1][0], x1_x2[1], span[1][1]],
#         zip(
#             np.linspace(span[0][0], span[0][1], 50, endpoint=True),
#             np.linspace(span[0][0], span[0][1], 50, endpoint=True),
#         ),
#     )
# )
# vertical = list(
#     map(
#         lambda y1_y2: [span[0][0], y1_y2[0], span[0][1], y1_y2[1]],
#         zip(
#             np.linspace(span[1][0] + 0.01, span[1][1], 50, endpoint=False),
#             np.linspace(span[1][0] + 0.01, span[1][1], 50, endpoint=False),
#         ),
#     )
# )
# horizontal = np.array(horizontal).reshape(-1, 2)
# vertical = np.array(vertical).reshape(-1, 2)
# samples = np.concatenate((samples, horizontal, vertical), axis=0)

plt.scatter(samples[:, 0], samples[:, 1])
plt.show()
np.save(f"samples/{len(samples)}-{span}", samples)
