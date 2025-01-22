import itertools
import math
import random
import time
import copy
from artificial_bee_colony import *
from deap import base, creator, tools
from initialize_population2 import assign_courses
from functools import partial
from model import *




# Initialize globals for course and room data, which will be set in main()
courses = []
rooms = []
curricula = []
data = []
room_map = {}
reverse_room_map = {}

# Define the fitness and particle classes
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Particle", dict, fitness=creator.FitnessMin, speed=dict, best=None, bestfit=None, is_quantum=False)
creator.create("Swarm", list, best=None, bestfit=None)

# Generate a particle (schedule) using Graph-Based Heuristic
def generate(pclass):
    schedule = None
    
    # Keep trying to assign courses until a valid schedule is generated
    while schedule is None:
        schedule = assign_courses()  # Generate the initial timetable
    
    return pclass(copy.deepcopy(schedule))


def evaluate_schedule(solution, rooms, courses, curricula, constraints):
    """
    Evaluates the fitness of a solution by calculating the total penalty
    for soft constraints.
    """
    total_penalty = (
        room_capacity_cost(solution) +
        room_stability_cost(solution) +
        curriculum_compactness_cost(solution) +
        minimum_working_days_cost(solution)
    )
    return (total_penalty,)  # Return as a tuple

toolbox = base.Toolbox()

def updateParticle(data, swarm, constraints, courses, curricula, rooms, days, periods):
    """
    Update particle using the Artificial Bee Colony (ABC) algorithm.

    Args:
        data: Input data required for evaluation.
        swarm: Current swarm of particles.
        constraints: Hard constraints for feasibility.
        courses, curricula, rooms, days, periods: Problem-specific data.

    Returns:
        Updated swarm after applying the ABC algorithm.
    """
    # Convert the swarm into a solution_set for ABC processing
    current_solution_set = [particle for particle in swarm]

    # Initialize ABC for the current swarm, retaining prior state
    abc(current_solution_set, maximum_cycles_param=1000, limit_param=100, retain_state=True)

    # Perform one cycle of ABC
    cycle_abc()
    
    # Update the swarm with the new solution_set and fitness values
    for i, solution in enumerate(current_solution_set):
        swarm[i] = solution
        swarm[i].fitness.values = toolbox.evaluate(solution)  # Recalculate fitness
        swarm[i].bestfit = creator.FitnessMin(solution.fitness.values) if swarm[i].bestfit is None else swarm[i].bestfit

        # Update particle's personal best
        if swarm[i].best is None or swarm[i].fitness.values < swarm[i].bestfit.values:
            swarm[i].best = toolbox.clone(swarm[i])
            swarm[i].bestfit.values = swarm[i].fitness.values

    # Update the swarm's global best
    for particle in swarm:
        if swarm.best is None or particle.fitness.values < swarm.bestfit.values:
            swarm.best = toolbox.clone(particle)
            swarm.bestfit.values = particle.fitness.values

    return swarm

def is_feasible(schedule, constraints, courses, curricula):
    """
    Check if the entire schedule adheres to all HARD constraints.
    
    Args:
        schedule (list): The schedule (particle) to check.
        constraints (list): Hard constraints.
        courses (list): Course details, including number of students and teachers.
        curricula (list): Curricula details, including associated courses.

    Returns:
        bool: True if the schedule satisfies all HARD constraints, False otherwise.
    """
    # Track room assignments by day, period, and room ID (hashmap)
    room_assignments = {}
    # Track course assignments by course ID (hashmap)
    course_assignments = {}
    # Track teacher conflicts (using sets to track course/day/period)
    teacher_conflicts = {}

    # Initialize dictionaries for room and course assignments
    for entry in schedule:
        day = entry["day"]
        period = entry["period"]
        room = entry["room_id"]
        course_id = entry["course_id"]

        # Initialize room assignments for the day and period
        if (day, period) not in room_assignments:
            room_assignments[(day, period)] = {}
        
        # Check if the room is already assigned
        if room in room_assignments[(day, period)]:
            return False  # H2 violation

        # Assign room to the current course at the given day and period
        room_assignments[(day, period)][room] = course_id

        # Track course assignments for the course ID
        if course_id not in course_assignments:
            course_assignments[course_id] = set()
        
        if (day, period) in course_assignments[course_id]:
            return False  # H1 violation
        course_assignments[course_id].add((day, period))

        # Check for teacher conflict (ensure one teacher isn't assigned to multiple courses at the same time)
        teacher = get_teacher(courses, course_id)
        if teacher:
            if (teacher, day, period) in teacher_conflicts:
                return False  # Teacher conflict
            teacher_conflicts[(teacher, day, period)] = course_id

    # Curriculum conflict: Check if courses from the same curriculum are scheduled at the same time
    for curriculum in curricula:
        curriculum_courses = curriculum["courses"]
        curriculum_assignments = [
            entry for entry in schedule if entry["course_id"] in curriculum_courses
        ]

        # Group assignments by day and period
        assignments_by_day_period = {}
        for entry in curriculum_assignments:
            day_period = (entry["day"], entry["period"])
            if day_period not in assignments_by_day_period:
                assignments_by_day_period[day_period] = []
            assignments_by_day_period[day_period].append(entry["course_id"])

        # Check if more than one course from the curriculum is scheduled in the same time slot
        for day_period, courses_in_slot in assignments_by_day_period.items():
            if len(courses_in_slot) > 1:  # More than one course in the same slot
                return False  # H3 Violation

    # Unavailability Constraints
    for constraint in constraints:
        for entry in schedule:
            if (
                constraint["course"] == entry["course_id"]
                and constraint["day"] == entry["day"]
                and constraint["period"] == entry["period"]
            ):
                return False  # H4 violation

    # If no conflicts were found, the schedule is feasible
    return True

def get_teacher(courses, course_id):
    """
    Retrieve the teacher for a given course ID.
    """
    for course in courses:
        if course["id"] == course_id:
            return course["teacher"]
    return None

def get_room_capacity(room_id, rooms):
    """
    Retrieve the capacity of a room by its ID.

    Args:
        room_id (str): The ID of the room.
        rooms (list): The list of room dictionaries.

    Returns:
        int: The capacity of the room.

    Raises:
        ValueError: If the room ID is not found.
    """
    for room in rooms:
        if room["id"] == room_id:
            return room["capacity"]
    raise ValueError(f"Room ID {room_id} not found in rooms list")

def calculate_distance(p1, p2, days, periods, rooms, room_map):
    """
    Calculate the distance between two timetable solutions.
    The distance is based on the day, period, and room differences for the same course.

    Args:
        p1 (dict): The first timetable solution.
        p2 (dict): The second timetable solution.
        days (int): Total number of days.
        periods (int): Total number of periods per day.
        rooms (list): List of all room IDs.
        room_map (dict): Mapping of room IDs to numeric indices.

    Returns:
        float: The distance between the two solutions.
    """
    distance = 0

    # Iterate over all courses
    for course_id in courses:
        slots1 = []  # Collect all slots for course_id in p1
        slots2 = []  # Collect all slots for course_id in p2

        # Find all assignments for the course in p1
        for day, periods_dict in p1.items():
            for period, room_dict in periods_dict.items():
                for room_id, assigned_course in room_dict.items():
                    if assigned_course == course_id['id']:
                        slots1.append({"day": day, "period": period, "room_id": room_id})

        # Find all assignments for the course in p2
        for day, periods_dict in p2.items():
            for period, room_dict in periods_dict.items():
                for room_id, assigned_course in room_dict.items():
                    if assigned_course == course_id['id']:
                        slots2.append({"day": day, "period": period, "room_id": room_id})

        # Match up assignments and calculate distances
        # If the number of slots differ, we match the shorter one
        for slot1, slot2 in zip(slots1, slots2):
            distance += (
                ((slot1["day"] - slot2["day"]) / days) ** 2 +
                ((slot1["period"] - slot2["period"]) / periods) ** 2 +
                ((room_map[slot1["room_id"]] - room_map[slot2["room_id"]]) / len(rooms)) ** 2
            )

    return math.sqrt(distance)

def convertQuantum(swarm, rcloud, centre, constraints, courses, curricula, rooms, days, periods):
    for part in swarm:
        if part == swarm.best:
            continue

        for day, periods_dict in centre.items():
            for period, rooms_dict in periods_dict.items():
                for room_id, best_course in rooms_dict.items():
                    if best_course == -1:
                        continue  # Skip unassigned slots

                    # Gaussian movement
                    new_day = (day + round(rcloud * random.gauss(0, 1))) % days
                    new_period = (period + round(rcloud * random.gauss(0, 1))) % periods
                    new_room_index = (room_map[room_id] + round(rcloud * random.gauss(0, 1))) % len(rooms)
                    new_room = reverse_room_map[new_room_index]

                    # Attempt to assign in new positions
                    original_entry = part.get(new_day, {}).get(new_period, {}).get(new_room, -1)
                    part[new_day][new_period][new_room] = best_course
                    
                    # Check feasibility
                    if not is_feasible(part, constraints, courses, curricula):
                        part[new_day][new_period][new_room] = original_entry


# Main loop to simulate the Multi-Swarm Particle Swarm Optimization
def main(data, max_iterations=500, verbose=True):
    global courses, rooms, curricula, room_map, reverse_room_map
    courses = data["courses"]
    rooms = data["rooms"]
    curricula = data["curricula"]
    constraints = data["constraints"]
    days = data["num_days"]
    periods = data["periods_per_day"]
    room_map = {room['id']: i for i, room in enumerate(rooms)}
    reverse_room_map = {i: room['id'] for i, room in enumerate(rooms)}
    lectures = 0

    course_order = [course['id'] for course in courses]
    for course in courses:
        lectures += course["num_lectures"]

    toolbox.register("particle", generate, creator.Particle)
    toolbox.register("swarm", tools.initRepeat, creator.Swarm, toolbox.particle)
    toolbox.register(
        "evaluate",
        partial(evaluate_schedule, rooms=rooms, courses=courses, curricula=curricula, constraints=constraints)
    )

    NSWARMS = 1
    NPARTICLES = 20
    NEXCESS = 0
    RCLOUD = 1
    NDIM = 3
    BOUNDS = len(rooms) * days * periods

    
    start_time = time.time()
    population = [toolbox.swarm(n=NPARTICLES) for _ in range(NSWARMS)]
    chi, c1, c2 = 0.729, 1, 1

    # Initialization
    best_global_fitness = float('inf')
    global_best_particle = None
    best_global_particle_idx = None
    initial_fitness_values = []
    last_global_best_update = -1
    init_flags = [False] * NSWARMS  # Init flags for randomization markers

    for swarm in population:
        swarm.best = None
        swarm.bestfit = creator.FitnessMin((float('inf'),))
        swarm.no_improvement_iters = 0

        for i, part in enumerate(swarm):
            part.fitness.values = toolbox.evaluate(part)
            part.best = toolbox.clone(part)
            part.bestfit = creator.FitnessMin(part.fitness.values)
            if swarm.best is None or part.fitness.values <= swarm.bestfit.values:
                swarm.best = toolbox.clone(part)
                swarm.bestfit.values = part.fitness.values

        if global_best_particle is None or swarm.bestfit.values[0] <= best_global_fitness:
            best_global_fitness = swarm.bestfit.values[0]
            global_best_particle = toolbox.clone(swarm.best)

    for swarm in population:
        for part in swarm:
            fitness = part.fitness.values[0]
            initial_fitness_values.append(fitness)

    print("\nInitial Fitness Values Before Optimization:")
    for i, fitness in enumerate(initial_fitness_values):
        print(f"Particle {i + 1}: Fitness = {fitness:.2f}")

    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}/{max_iterations}")
        rexcl = (BOUNDS / len(population))** (1.0 / NDIM)
        #print("Rexcl: ", rexcl) 
        print("Total Swarms: ", len(population))

        # Anti-Convergence
        print("Anti-Convergence check")
        all_converged = True
        worst_swarm_idx = None
        worst_swarm_fitness = float('-inf') # Initial fitness

        for i, swarm in enumerate(population):
            for p1, p2 in itertools.combinations(swarm, 2):
                distance = calculate_distance(p1, p2, days, periods, rooms, room_map)

                if distance > 2 * rexcl:
                    all_converged = False
                    #print("Not all have converged yet")
                    break

            if all_converged and swarm.bestfit.values[0] > worst_swarm_fitness:
                worst_swarm_fitness = swarm.bestfit.values[0]
                worst_swarm_idx = i
                swarm.bestfit.values[0]
                print("Index: ", worst_swarm_idx)
                print("Fitness: ", worst_swarm_fitness)
        
        # Adding swarms or reinitializing worse swarm
        if all_converged and len(population) < NSWARMS + NEXCESS:
            new_swarm = toolbox.swarm(n=NPARTICLES)
            new_swarm.best = None
            new_swarm.bestfit = creator.FitnessMin((float('inf'),))
            new_swarm.no_improvement_iters = 0

            # Log the fitness of the new swarm
            print("\nNew Swarm Added:")
            for part in new_swarm:
                part.fitness.values = toolbox.evaluate(part)
                part.best = toolbox.clone(part)
                part.bestfit = creator.FitnessMin(part.fitness.values)
                if new_swarm.best is None or part.fitness <= new_swarm.bestfit:
                    new_swarm.best = toolbox.clone(part)
                    new_swarm.bestfit.values = part.fitness.values
                # Log the fitness
                print(f"New Particle Fitness: {part.fitness.values[0]:.2f}")
                initial_fitness_values.append(part.fitness.values[0])

            population.append(new_swarm)
            init_flags.append(False)

        elif all_converged and worst_swarm_idx is not None:
            print(f"Randomizing worst swarm: {worst_swarm_idx}")
            init_flags[worst_swarm_idx] = True 

        # Exclusion
        print("Exclusion check")
        reinit_swarms = set()
        for s1, s2 in itertools.combinations(range(len(population)), 2):
            if population[s1].best and population[s2].best and not (s1 in reinit_swarms or s2 in reinit_swarms):
                distance = calculate_distance(
                    population[s1].best, population[s2].best, days, periods, rooms, room_map
                )
                if distance < rexcl:
                    reinit_swarms.add(s1 if population[s1].bestfit <= population[s2].bestfit else s2)

        for s in reinit_swarms:
            print(f"Reinitializing swarm: {s}")
            init_flags[s] = True

        # Update and Randomize Particles
        print("Update and Randomize")
        for i, swarm in enumerate(population):
            # if init_flags[i]:
            #     #print(f"Swarm: {i+1}")
            #     convertQuantum(swarm, RCLOUD, swarm.best, constraints, courses, curricula, rooms, days, periods)
            #     init_flags[i] = False
            #     for j, part in enumerate(swarm):
            #         #print("Particle "+ str((NPARTICLES*i)+(j+1)) + " (Fitness: "+ str(part.fitness.values[0]) + ")")
            #         if swarm.best is None or part.fitness.values < swarm.bestfit.values:
            #             swarm.best = toolbox.clone(part)
            #             swarm.bestfit.values = part.fitness.values
            #     print(f"Swarm has been reinitialized. Swarm bestfit is now {swarm.bestfit}.")
            # else:
            updateParticle(data, swarm, part.best, swarm.best, chi, c1, c2, constraints)

        best_particle = None
        best_fitness_in_population = float('inf')
        for swarm in population:
            if swarm.bestfit.values[0] <= best_fitness_in_population:
                best_particle = swarm.best
                best_fitness_in_population = best_particle.fitness.values[0]

        print("Best fitness: ", best_fitness_in_population)
        if best_fitness_in_population <= best_global_fitness and best_particle != global_best_particle:
            for swarm_idx, swarm in enumerate(population):
                for particle_idx, particle in enumerate(swarm):
                    if particle.fitness.values[0] <= best_global_fitness:
                        best_global_fitness = particle.fitness.values[0]
                        global_best_particle = toolbox.clone(particle)
                        best_global_particle_idx = (swarm_idx, particle_idx)
                        last_global_best_update = iteration
                        print("##############GLOBAL BEST FITNESS UPDATED##############")
                        print(f"Global best updated at iteration {iteration + 1} by swarm {swarm_idx + 1}, particle {particle_idx + 1}")

        # Stop if the fitness meets the target of 0 or less
        if best_global_fitness <= 0:
            print(f"\nStopping early as target fitness of 0 was reached: {best_global_fitness}")
            break
        print("Best global fitness: ", best_global_fitness)

    # End the timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    # Determine the best particle across all swarms based on bestfit values if available
    valid_swarms = [swarm for swarm in population if swarm.best is not None and swarm.bestfit is not None]

    print("\nOptimization Completed.")

    print("\nInitial Fitness Values Before Optimization:")
    for i, fitness in enumerate(initial_fitness_values):
        print(f"Solution {i + 1}: Fitness = {fitness:.2f}")

    print("\nFitness Values After Optimization:")
    for i, swarm in enumerate(population):
        for j, part in enumerate(swarm):
            print("Solution "+ str((NPARTICLES*i)+(j+1)) + ": " + "Fitness: "+ str(part.fitness.values[0]))

    particle_origin = (NPARTICLES*best_global_particle_idx[0]) + best_global_particle_idx[1] + 1
    if valid_swarms:
        # Find the swarm with the minimum bestfit value
        best_swarm = min(valid_swarms, key=lambda s: s.bestfit.values[0])
        final_best_schedule = best_swarm.best
        print("\nFinal Best Solution Found (Fitness):", best_swarm.bestfit.values[0])
        print(f"\nOptimization completed in {elapsed_time:.2f} seconds.")
    else:
        # Fallback if no swarm has a valid best (highly unlikely with this setup)
        final_best_schedule = None
        print("\nNo solution found.")

    print(f"The last global best was updated at iteration {last_global_best_update + 1}")
    if best_global_particle_idx:
        print(f"\nBest solution found by particle: ", particle_origin)
    else:
        print("\nNo valid best solution found.")

    best_solution = []

    for day, periods in final_best_schedule.items():
        for period, rooms in periods.items():
            for room, course_id in rooms.items():
                if course_id != -1:  # Skip empty slots
                    best_solution.append({
                        "day": day,
                        "period": period,
                        "room_id": room,
                        "course_id": course_id
                    })

    return best_solution