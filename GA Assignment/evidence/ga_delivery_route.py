import random

# Depot A and delivery stops B..G
LOCATIONS = ["A", "B", "C", "D", "E", "F", "G"]
DELIVERY_STOPS = ["B", "C", "D", "E", "F", "G"]
LOC_TO_IDX = {name: i for i, name in enumerate(LOCATIONS)}

# Symmetric distance matrix in kilometres (from/to A, B, C, D, E, F, G)
DIST_MATRIX = [
    [0, 4, 8, 6, 7, 9, 5],  # A
    [4, 0, 5, 7, 3, 8, 6],  # B
    [8, 5, 0, 4, 6, 3, 7],  # C
    [6, 7, 4, 0, 5, 6, 8],  # D
    [7, 3, 6, 5, 0, 4, 6],  # E
    [9, 8, 3, 6, 4, 0, 5],  # F
    [5, 6, 7, 8, 6, 5, 0],  # G
]


def route_distance(stops):
    """Total round-trip distance: A -> stops -> A."""
    tour = ["A"] + list(stops) + ["A"]
    total = 0
    for i in range(len(tour) - 1):
        r = LOC_TO_IDX[tour[i]]
        c = LOC_TO_IDX[tour[i + 1]]
        total += DIST_MATRIX[r][c]
    return total


def init_population(pop_size, rng):
    """Create initial valid permutation routes over B..G."""
    pop = []
    for _ in range(pop_size):
        chrom = DELIVERY_STOPS.copy()
        rng.shuffle(chrom)
        pop.append(chrom)
    return pop


def tournament_select(pop, fitness, rng, k=3):
    """Pick the shortest route among k randomly chosen candidates."""
    contestants = rng.sample(range(len(pop)), k=min(k, len(pop)))
    best_idx = min(contestants, key=lambda i: fitness[i])
    return pop[best_idx].copy()


def order_crossover(p1, p2, rng):
    """Order crossover: preserves order without duplicate or missing stops."""
    n = len(p1)
    a, b = sorted(rng.sample(range(n), 2))
    child = [None] * n
    child[a : b + 1] = p1[a : b + 1]
    used = set(child[a : b + 1])

    w = (b + 1) % n
    r = (b + 1) % n
    while None in child:
        gene = p2[r]
        if gene not in used:
            child[w] = gene
            used.add(gene)
            w = (w + 1) % n
        r = (r + 1) % n
    return child


def swap_mutate(chrom, rng, mut_rate=0.25):
    """Swap two stops in the route with probability mut_rate."""
    out = chrom.copy()
    if rng.random() < mut_rate:
        i, j = rng.sample(range(len(out)), 2)
        out[i], out[j] = out[j], out[i]
    return out


def run_ga(pop_size=80, generations=120, k=3, cx_rate=0.85, mut_rate=0.25, n_elite=2, seed=42):
    rng = random.Random(seed)
    pop = init_population(pop_size, rng)

    best_stops = None
    best_dist = float("inf")

    for _ in range(generations):
        dists = [route_distance(chrom) for chrom in pop]
        ranked = sorted(range(pop_size), key=lambda i: dists[i])

        if dists[ranked[0]] < best_dist:
            best_dist = dists[ranked[0]]
            best_stops = pop[ranked[0]].copy()

        # Elitism: keep top routes unchanged
        next_pop = [pop[i].copy() for i in ranked[:n_elite]]

        # Selection, crossover, and mutation
        while len(next_pop) < pop_size:
            parent1 = tournament_select(pop, dists, rng, k=k)
            parent2 = tournament_select(pop, dists, rng, k=k)

            if rng.random() < cx_rate:
                child1 = order_crossover(parent1, parent2, rng)
                child2 = order_crossover(parent2, parent1, rng)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            child1 = swap_mutate(child1, rng, mut_rate)
            next_pop.append(child1)
            if len(next_pop) < pop_size:
                child2 = swap_mutate(child2, rng, mut_rate)
                next_pop.append(child2)

        pop = next_pop

    final_dists = [route_distance(chrom) for chrom in pop]
    best_final = min(range(pop_size), key=lambda i: final_dists[i])
    if final_dists[best_final] < best_dist:
        best_dist = final_dists[best_final]
        best_stops = pop[best_final].copy()

    return " -> ".join(["A"] + best_stops + ["A"]), best_dist


if __name__ == "__main__":
    best_route, best_km = run_ga(seed=42)
    print("Best route:", best_route)
    print("Total distance:", best_km, "km")