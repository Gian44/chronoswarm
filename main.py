from artificial_bee_colony3 import MultiSwarmABC
import pandas as pd
from config import *
import time


def save_output(schedule, csv_path, out_path):
    # Convert the schedule to DataFrame for CSV
    df = pd.DataFrame(schedule)
    df.to_csv(csv_path, index=False)
    
    # Write the output to .out format
    with open(out_path, 'w') as f:
        for entry in schedule:
            line = f"{entry['course_id']} {entry['room_id']} {entry['day']} {entry['period']}\n"
            f.write(line)

NUM_SWARMS = 4
POPULATION_SIZE = 5
MAX_ITERATIONS = 3000
LIMIT = 100

multi_swarm = MultiSwarmABC(NUM_SWARMS, POPULATION_SIZE, MAX_ITERATIONS, LIMIT)

start_time = time.time()

best_solutions, best_fitness = multi_swarm.run()

best = min(fitness for fitness in best_fitness)

end_time = time.time()
elapsed_time = end_time - start_time

multi_swarm.get_fitness_per_swarm()
print("Best fitness:", best)

solution = multi_swarm.get_global_best(best_solutions, best_fitness)
save_output(solution, "mnt/data/comp01.csv", OUTPUT)
print(f"\nOptimization completed in {elapsed_time:.2f} seconds.")