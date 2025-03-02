import random
import copy
import itertools
import math
from ctt_parser import read_ctt_file
from initialize_population2 import assign_courses as initialize_solution
from model import *
from config import *

# Load problem data
filename = INPUT  # Replace with your .ctt file name
courses, rooms, unavailability_constraints, curricula, days, periods_per_day = read_ctt_file(filename)
periods_rooms = periods_per_day * len(rooms)
days_periods_rooms = periods_rooms * days
room_map = {room: i for i, room in enumerate(rooms)}
reverse_room_map = {i: room for room, i in room_map.items()}
total_lectures = sum(course['lectures'] for course in courses.values())

class PSOSwarm:
    """Represents a single swarm in the Multi-Swarm Particle Swarm Optimization (MSPSO)."""

    def __init__(self, population_size, limit, num_swarms):
        self.population_size = population_size
        self.limit = limit
        self.num_swarms = num_swarms
        self.solution_set = [self.produce_solution() for _ in range(population_size)]
        self.fitness_set = [self.evaluate_fitness(sol) for sol in self.solution_set]
        self.depth_set = [self.evaluate_fitness(sol) for sol in self.solution_set]
        print(self.fitness_set)
        self.employed_bee = [0] * population_size
        self.global_best_solution = self.get_best_solution()
        self.reinitialize_flag = False  

    def produce_solution(self):
        """Generate an initial feasible solution."""
        schedule = None
        while schedule is None:
            schedule = initialize_solution()

        return copy.deepcopy(schedule)

    def evaluate_fitness(self, solution):
        """Compute the fitness of a given solution."""
        return eval_fitness(solution)
    def evaluate_scout_fitness(self, solution):
        """Compute the fitness of a given solution."""
        return (
            room_capacity_cost(solution)
            + room_stability_cost(solution)
        )

    def get_best_solution(self):
        """Retrieve the best solution in the swarm."""
        best_index = min(range(len(self.solution_set)), key=lambda i: self.fitness_set[i])
        return self.solution_set[best_index]
    
    def get_worst_solution(self):
        worst = -1
        worst_solution = -1
        for solution in range(len(self.solution_set)):
            fitness = self.fitness_set[solution]
            if fitness >= worst: 
                worst = fitness
                worst_solution = solution
        return worst_solution

    def calculate_probability(self):
        """Calculate selection probabilities for onlooker bees."""
        total_fitness = sum(1 / (fit + 1) for fit in self.fitness_set)
        return total_fitness

    def employed_bee_phase(self):
        """Perform solution improvement for employed bees."""
        for bee in range(self.population_size):
            self.update(1, bee)

    def update(self, bee_type, bee_index):
        """Perform solution mutation for a given bee."""
        if bee_type == 1:
            index = bee_index
            solution = copy.deepcopy(self.solution_set[bee_index])
        else:
            index = self.onlooker_bee[bee_index]
            solution = copy.deepcopy(self.solution_set[self.onlooker_bee[bee_index]])

        course = -1
        c1, c2, chi = 1, 1, 0.729
        r1, r2 = random.random(), random.random()
        courses_index = list(courses.keys())
        courses_index.append(-1)
        neighbor_index = random.randint(0, len(self.solution_set)-1)  # Get index
        neighbor1 = self.global_best_solution #self.solution_set[neighbor_index]  # Get corresponding solution
        neighbor2 = random.choice(self.solution_set) #global_best_solution
        best_neighbor = self.global_best_solution
        day, period, room = 0, 0, "-1"
        
        while course == -1:
            day = random.randint(0, days - 1)
            period = random.randint(0, periods_per_day - 1)
            room = list(rooms.keys())[random.randint(0, len(rooms) - 1)]
            course = solution[day][period][room]
            
        neighbor_course = neighbor1[day][period][room]

        course_value = courses_index.index(course)
        neighbor_value = courses_index.index(neighbor_course)
        pb_value = courses_index.index(course)

        velocity = chi * ((c1 * r1 * (pb_value - course_value)) + (c2 * r2 * (neighbor_value - course_value)))
        position = (course_value + round(velocity)) % (len(courses_index))

        new_solution = copy.deepcopy(solution)
        conflict = False
        swapped1 = False
        swapped2 = False
        cslots = []
        slot_index = 0
        least_difference = float("inf")
        i = 0 

        #print("Course: " + str(course) + " Neighbor Course: " + str(courses_index[position]))

        for r in rooms:
            #print(str(new_solution[day][period][r]) + " and " + str(courses_index[position]))
            if new_solution[day][period][r] == courses_index[position] and room != r:
                new_solution[day][period][room] = courses_index[position]
                new_solution[day][period][r] = course
                #print("Solution Swapped (Before swap 1)")
                swapped1 = True
                swapped2 = True
                break
        
        if not swapped1:
            if courses_index[position] != -1:
                for target_course in get_assigned_courses_by_period(day, period, new_solution):
                    if has_conflict(courses_index[position], target_course, new_solution):
                        conflict = True
                        break

            if (courses_index[position] in unavailability_constraints and (day, period) in unavailability_constraints[courses_index[position]]) or conflict:
                #print("unavailable or conflict")
                for d in range (days):
                    for p in range (periods_per_day):
                        for r in rooms:
                            #print(str(new_solution[d][p][r]) + " and " + str(courses_index[position]))
                            if solution[d][p][r] == courses_index[position]:
                                cslots.append([d, p, r])

                #print("cslots", cslots)
                cslot = random.choice(cslots)
                available_slots = get_available_slots(courses_index[position], solution)
                if available_slots:
                    slot = random.choice(available_slots)
                    new_solution[cslot[0]][cslot[1]][cslot[2]] = -1
                    new_solution[slot[0]][slot[1]][slot[2]] = courses_index[position]

            elif courses_index[position] == -1:
                #print("Is -1")
                available_slots = get_available_slots(course, solution)
                if available_slots:
                    slot = random.choice(available_slots)
                    new_solution[day][period][room] = -1
                    new_solution[slot[0]][slot[1]][slot[2]] = course
            else:       
                if not swapped1:
                    emp_slots = []
                    for r in rooms:
                        if new_solution[day][period][r] == -1:
                            emp_slots.append([day, period, r])

                    for d in range (days):
                        for p in range (periods_per_day):
                            for r in rooms:
                                #print(str(new_solution[d][p][r]) + " and " + str(courses_index[position]))
                                if new_solution[d][p][r] == courses_index[position]:
                                    cslots.append([d, p, r])

                    #print("cslots", cslots)
                    cslot = random.choice(cslots)
                    
                    if emp_slots:
                        emp_slot = random.choice(emp_slots)
                        new_solution[cslot[0]][cslot[1]][cslot[2]] = -1
                        new_solution[day][period][room] = courses_index[position]
                        new_solution[emp_slot[0]][emp_slot[1]][emp_slot[2]] = course
                        #print("Solution Swapped (Before swap 2)")
                        swapped2 = True

                if not swapped2:
                    for d in range (days):
                        for p in range (periods_per_day):
                            for r in rooms:
                                #print(str(new_solution[d][p][r]) + " and " + str(courses_index[position]))
                                if new_solution[d][p][r] == courses_index[position]:
                                    cslots.append([d, p, r])

                    #print("cslots", cslots)
                    cslot = random.choice(cslots)
                    available_slots = get_available_slots(course, solution, cslot)
                    if available_slots:
                        slot = random.choice(available_slots)
                        new_solution[cslot[0]][cslot[1]][cslot[2]] = -1
                        new_solution[day][period][room] = courses_index[position]
                        new_solution[slot[0]][slot[1]][slot[2]] = course
                        #print("Solution Swapped (in swap 3)")
                    
        new_fitness = self.evaluate_fitness(new_solution)

        if new_fitness <= self.fitness_set[index] and solution != new_solution:
            self.depth_set[index] = self.fitness_set[index]
            self.fitness_set[index] = new_fitness
            self.solution_set[index] = new_solution
        # print(self.stagnation)

    def run_cycle(self):
        """Run a full PSO cycle."""
        self.employed_bee_phase()
        self.global_best_solution = self.get_best_solution()

    def get_fitness(self):
        """Return the best fitness of the swarm."""
        return self.evaluate_fitness(self.global_best_solution)

class MultiSwarmABC:
    """Manages multiple ABC swarms running in parallel with exclusion and anti-convergence mechanisms."""

    def __init__(self, num_swarms, population_size, max_iterations, limit, rcloud):
        self.num_swarms = num_swarms
        self.max_iterations = max_iterations
        self.target_fitness = 0 
        self.rcloud = rcloud
        self.swarms = [PSOSwarm(population_size, limit, num_swarms) for _ in range(num_swarms)]
        self.global_best_solution = None
        self.global_best_fitness = float("inf")
        self.reinitialize_flag = False  

    def calculate_distance(self, p1, p2):

        """Calculate the distance between two timetable solutions."""
        distance = 0

        for course_id in courses:
            slots1 = []
            slots2 = []

            # Extract course assignments from both solutions
            for day, periods_dict in p1.items():
                for period, room_dict in periods_dict.items():
                    for room_id, assigned_course in room_dict.items():
                        if assigned_course == course_id:
                            slots1.append({"day": day, "period": period, "room_id": room_id})

            for day, periods_dict in p2.items():
                for period, room_dict in periods_dict.items():
                    for room_id, assigned_course in room_dict.items():
                        if assigned_course == course_id:
                            slots2.append({"day": day, "period": period, "room_id": room_id})

            # Compute the difference in schedule assignments
            for slot1, slot2 in zip(slots1, slots2):
                distance += (
                    ((slot1["day"] - slot2["day"]) / days) ** 2 +
                    ((slot1["period"] - slot2["period"]) / periods_per_day) ** 2 +
                    ((room_map[slot1["room_id"]] - room_map[slot2["room_id"]]) / len(rooms)) ** 2
                )
        #print(math.sqrt(distance))
        return math.sqrt(distance)

    def convertQuantum(self, swarm: PSOSwarm):
        """Quantum reinitialization of a swarm using Gaussian movement."""

        centre = swarm.global_best_solution  # Use the best solution as the center
        new_solution_set = []  # Store new solutions

        for part in swarm.solution_set:
            new_solution = copy.deepcopy(part)  # Copy solution before modification

            for day, periods_dict in centre.items():
                for period, rooms_dict in periods_dict.items():
                    for room_id, best_course in rooms_dict.items():
                        if best_course == -1:
                            continue  # Skip unassigned slots

                        # Apply Gaussian movement
                        new_day = (day + round(self.rcloud * random.gauss(0, 1))) % days
                        new_period = (period + round(self.rcloud * random.gauss(0, 1))) % periods_per_day
                        new_room_index = (room_map[room_id] + round(self.rcloud * random.gauss(0, 1))) % len(rooms)
                        new_room = reverse_room_map[new_room_index]

                        # Attempt to assign in new positions
                        original_entry = part.get(new_day, {}).get(new_period, {}).get(new_room, -1)
                        part[new_day][new_period][new_room] = best_course

                        # Validate feasibility
                        if not is_feasible(part, unavailability_constraints, courses, curricula):
                            part[new_day][new_period][new_room] = original_entry  # Undo if infeasible

            # Ensure new solutions are assigned
            new_solution_set.append(new_solution)

        # ✅ Overwrite entire swarm with new solutions
        swarm.solution_set = new_solution_set
        swarm.fitness_set = [swarm.evaluate_fitness(sol) for sol in swarm.solution_set]

        swarm.reinitialize_flag = False  # Reset flag after quantum reinitialization


    def run(self):
        """Run all swarms for the specified number of iterations with exclusion and anti-convergence."""
        # ✅ Compute convergence & exclusion radii first
        rexcl = (days_periods_rooms / len(self.swarms)) ** (1 / 3)
        rconv = rexcl
        worst_fitness = 0
        print("Rexcl:", rexcl)
        for cycle in range(self.max_iterations):
            # ✅ **Step 1: Anti-Convergence**
            for swarm in self.swarms:
                all_converged = all(
                    self.calculate_distance(swarm.solution_set[i], swarm.solution_set[j]) > 2 * rconv
                    for i, j in itertools.combinations(range(len(swarm.solution_set)), 2)
                )
            # 🔹 If all solutions in the swarm have converged → Tag swarm for reinitialization
            if all_converged:
                for i, swarm in enumerate(self.swarms):
                    fitness = swarm.evaluate_fitness(swarm.get_best_solution())
                    if fitness >= worst_fitness:
                        print(f"Tagging swarm {i+1} for quantum reinitialization due to stagnation.")
                        worst_swarm = swarm
                        worst_fitness = fitness
                worst_swarm.reinitialize_flag = True  # Mark for quantum reinitialization

            # ✅ **Step 2: Exclusion**
            for i, j in itertools.combinations(range(len(self.swarms)), 2):
                best_i = self.swarms[i].global_best_solution
                best_j = self.swarms[j].global_best_solution
                
                distance = self.calculate_distance(best_i, best_j)

                if distance < rexcl:
                    worse_swarm_idx = i if self.swarms[i].get_fitness() > self.swarms[j].get_fitness() else j
                    print(f"Tagging swarm {worse_swarm_idx} for quantum reinitialization due to exclusion.")
                    self.swarms[worse_swarm_idx].reinitialize_flag = True  # Tag swarm for reinitialization

            # ✅ **Step 3: Run ABC or Quantum Reinitialization**
            for swarm in self.swarms:
                if swarm.reinitialize_flag:
                    self.convertQuantum(swarm)  # Quantum Reinitialization
                else:
                    swarm.run_cycle()  # Standard ABC Update

            # ✅ **Step 4: Update Global Best**
            best_fitness = [swarm.get_fitness() for swarm in self.swarms]
            for i, fitness in enumerate(best_fitness):
                if fitness <= self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_solution = self.swarms[i].global_best_solution

            print(f"Iteration {cycle+1}: Global Best Fitness: {self.global_best_fitness}")

            # ✅ **Step 5: Check Stopping Condition**
            if self.global_best_fitness <= self.target_fitness:
                print(f"Stopping early, reached target fitness: {self.global_best_fitness}")
                break

            for swarm in self.swarms:
                best = swarm.global_best_solution
                print(str(swarm.fitness_set) + str(swarm.evaluate_fitness(best)))
        
        best_solutions = [swarm.global_best_solution for swarm in self.swarms]
        best_fitness = [swarm.evaluate_fitness(sol) for sol in best_solutions]
        return best_solutions, best_fitness
    
    def get_fitness_per_swarm(self):
        for i, swarm in enumerate(self.swarms):
            print(f"Swarm {i+1}: " + str(swarm.fitness_set))

    def get_global_best(self, best_solutions, best_fitness):
        bs = None
        bf = float("inf")

        for i in range(len(best_solutions)):
            if best_fitness[i] <= bf :
                bs = best_solutions[i]
                bf = best_fitness[i]
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
            
        return best_solution


# Utility Functions
def is_feasible(schedule, constraints, courses, curricula):
        """
        Check if the entire schedule adheres to all HARD constraints.
        
        Args:
            schedule (dict): The schedule (solution) to check.
            constraints (dict): Unavailability constraints.
            courses (dict): Course details, including number of students and teachers.
            curricula (dict): Curricula details, including associated courses.

        Returns:
            bool: True if the schedule satisfies all HARD constraints, False otherwise.
        """
        for day, periods in schedule.items():
            for period, rooms in periods.items():
                for room, course_id in rooms.items():
                    if course_id == -1:
                        continue  # Skip empty slots

                    # Check for unavailability constraints
                    if course_id in constraints and (day, period) in constraints[course_id]:
                        return False

                    # Check if courses have conflicting teachers
                    teacher = courses[course_id]["teacher"]
                    for other_course in get_assigned_courses_by_period(day, period, schedule):
                        if courses[other_course]["teacher"] == teacher:
                            return False  # Teacher conflict

                    # Check curriculum constraints
                    for curriculum in curricula:
                        if course_id in curricula[curriculum]:
                            for other_course in get_assigned_courses_by_period(day, period, schedule):
                                if other_course in curricula[curriculum] and other_course != course_id:
                                    return False  # Curriculum conflict

        return True  # If no constraints are violated, the schedule is feasible

def get_available_slots(course, timetable, constraint_period=[-1, -1, -1]):
    """Retrieve available time slots for a given course."""
    available_slots = []
    for day in timetable:
        for period in timetable[day]:
            has_conflict = (day == constraint_period[0] and period == constraint_period[1])
            for target_course in get_assigned_courses_by_period(day, period, timetable):
                if has_conflict(course, target_course, timetable):
                    has_conflict = True
                    break
            if not has_conflict:
                for room in timetable[day][period]:
                    if timetable[day][period][room] == -1:
                        available_slots.append([day, period, room])
    return available_slots

def getRoomIndex(room):
    cnt = 0
    for r in rooms:
        cnt+=1
        if room == r:
            return cnt
    return -1 
def get_available_slots(course, timetable, constraint_period=[-1,-1,-1]):
    available_slots = []
    for day in timetable:
        for period in timetable[day]:
            hasConflict = False
            if day == constraint_period[0] and period == constraint_period[1]: hasConflict = True
            for target_course in get_assigned_courses_by_period(day, period, timetable):
                if has_conflict(course, target_course, timetable):
                    hasConflict = True
                    break
            if hasConflict != True:
                for room in timetable[day][period]:
                    slot = [day, period, room]
                    isValid = True
                    if course in unavailability_constraints and (day, period) in unavailability_constraints[course]:
                        isValid = False
                    if timetable[day][period][room] == -1 and isValid:
                        available_slots.append(slot)
    return available_slots

def get_swappable_slots(course_slot, timetable):
    available_slots = []
    for day in timetable:
        for period in timetable[day]:
            for room in timetable[day][period]:
                swappable_course = timetable[day][period][room]
                hasSwappableConflict = False
                isSwappableValid = True
                hasConflict = False
                if swappable_course == -1: hasConflict = True
                if day == course_slot[1] and period == course_slot[2]: hasConflict = True
                for target_course in get_assigned_courses_by_period(day, period, timetable):
                    if has_conflict(course_slot[0], target_course, timetable):
                        hasConflict = True
                        break
                if hasConflict != True:
                    
                    slot = [day, period, room]
                    isValid = True
                    if course_slot[0] in unavailability_constraints and (day, period) in unavailability_constraints[course_slot[0]]:
                        isValid = False
                    for target_course in get_assigned_courses_by_period(course_slot[1], course_slot[2], timetable):
                        if has_conflict(swappable_course, target_course, timetable) and target_course != course_slot[0]:
                            hasSwappableConflict  = True
                    if swappable_course in unavailability_constraints and (course_slot[1], course_slot[2]) in unavailability_constraints[swappable_course]:
                        isSwappableValid = False
                    if isValid and isSwappableValid and not hasSwappableConflict:
                        available_slots.append(slot)
    return available_slots

def has_conflict(course1, course2, timetable):
    # Check if courses have the same teacher
    if courses[course1]['teacher'] == courses[course2]['teacher']:
        return True

    # Check if courses are in the same curriculum
    for curriculum_id, course_list in curricula.items():
        if course1 in course_list and course2 in course_list:
            return True
        
    return False
        
def get_assigned_courses_by_period(day, period, timetable):
    courses = []
    for room in timetable[day][period]:
        if timetable[day][period][room] != -1:
            courses.append(timetable[day][period][room])
    return courses

def get_available_period(course, timetable, constraint_period=[-1,-1,-1]):
    available_slots = []
    for day in timetable:
        for period in timetable[day]:
            hasConflict = False
            if day == constraint_period[0] and period == constraint_period[1]: hasConflict = True
            for target_course in get_assigned_courses_by_period(day, period, timetable):
                if has_conflict(course, target_course, timetable):
                    hasConflict = True
                    break
            if hasConflict != True:
                slot = [day, period, constraint_period[2]]
                isValid = True
                if course in unavailability_constraints and (day, period) in unavailability_constraints[course]:
                    isValid = False
                if timetable[day][period][constraint_period[2]] == -1 and isValid:
                    available_slots.append(slot)
    return available_slots

def get_available_slots_different_period_room(course, timetable, constraint_period=[-1,-1,-1]):
    available_slots = []
    for day in timetable:
        for period in timetable[day]:
            hasConflict = False
            
            if day == constraint_period[0] and period == constraint_period[1]: hasConflict = True
            if not hasConflict:
                for target_course in get_assigned_courses_by_period(day, period, timetable):
                    if has_conflict(course, target_course, timetable):
                        hasConflict = True
                        break
            if not hasConflict:
                for room in timetable[day][period]:
                    slot = [day, period, room]
                    isValid = True
                    if (course in unavailability_constraints and (day, period) in unavailability_constraints[course]) or room == constraint_period[2]:
                        isValid = False
                    if timetable[day][period][room] == -1 and isValid:
                        available_slots.append(slot)
    return available_slots


def get_courses_with_mwd_violations(timetable):
    violating_course_assignment = {}
    violating_courses = []
    course_assignments = {}
    course_days = {}
    # Iterate through the timetable
    for day in timetable:
        for period in timetable[day]:
            for room in timetable[day][period]:
                course = timetable[day][period][room]
                if course != -1:  # Ignore unassigned slots
                    if course not in course_days:
                        course_days[course] = set()  # Initialize a set for unique rooms
                        course_assignments[course] = set()
                    course_assignments[course].add((day, period, room))
                    if day not in course_days[course]:
                        course_days[course].add(day)
                        
    for course in course_days:
        if courses[course]['min_days'] > (len(course_days[course])):
            violating_courses.append(course)
            violating_course_assignment[course] = course_assignments[course]
    return violating_courses, violating_course_assignment