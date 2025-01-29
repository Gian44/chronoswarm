from artificial_bee_colony3 import ArtificialBeeColony
from initialize_population import assign_courses
from config import *
from ctt_parser import read_ctt_file
import copy

# Read .ctt file
filename = INPUT
courses, rooms, unavailability_constraints, curricula, days, periods_per_day = read_ctt_file(filename)

# Create initial solutions
solution_set = [copy.deepcopy(assign_courses()) for _ in range(20)]

# Initialize Artificial Bee Colony
abc = ArtificialBeeColony(
    solution_set=solution_set,
    maximum_cycles=100,
    limit=500,
    courses=courses,
    rooms=rooms,
    unavailability_constraints=unavailability_constraints,
    curricula=curricula,
    days=days,
    periods_per_day=periods_per_day,
)

# Run the ABC algorithm
for cycle in range(abc.maximum_cycles):
    best_solution = abc.cycle()
    print(f"Cycle {cycle + 1}: Best Fitness = {abc.evaluate_fitness(best_solution)}")
