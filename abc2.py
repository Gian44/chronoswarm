from artificial_bee_colony6 import *
from config import *
import pandas as pd
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

def get_global_best(best_solutions):
        bs = best_solutions
        input_course_list = list(courses.keys())

        # Convert solution to a list of dictionaries
        best_solution = []
        for day, periods in bs.items():
            for period, rooms in periods.items():
                for room, course_id in rooms.items():
                    if course_id != -1:  # Skip empty slots
                        best_solution.append({                  
                            "day": day,
                            "period": period,
                            "room_id": room,
                            "course_id": course_id
                        })

        # ✅ Sort by input_course_list order
        best_solution.sort(key=lambda x: input_course_list.index(x["course_id"]) if x["course_id"] in input_course_list else -1)

        return best_solution

def main():
    population_size = POPULATION_SIZE
    limit = LIMIT
    max_iterations = MAX_ITERATIONS
    global moves

    swarm = ABCSwarm(population_size, limit)
    start_time = time.time()
    end_time = start_time + TIME
    iteration = 0

    while time.time() < end_time:
        moves += swarm.run_cycle()
        iteration += 1
        print(f"Iteration {iteration}: Best Fitness = {swarm.get_fitness()}")

    print(swarm.fitness_set, swarm.employed_bee_fitness)
    best_solution = swarm.global_best_solution
    best_fitness = swarm.get_fitness()
    elapsed_time = time.time() - start_time

    print("Best Fitness Found:", best_fitness)
    solution = get_global_best(best_solution)
    save_output(solution, "mnt/data/comp01.csv", OUTPUT)
    print(moves)
    print(f"\nOptimization completed in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main()