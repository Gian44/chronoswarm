import random
import copy
from ctt_parser import read_ctt_file
from initialize_population import assign_courses as initialize_solution
from model import *
from config import *

# Load problem data
filename = INPUT  # Replace with your .ctt file name
courses, rooms, unavailability_constraints, curricula, days, periods_per_day = read_ctt_file(filename)
periods_rooms = periods_per_day * len(rooms)
days_periods_rooms = periods_rooms * days

print(days_periods_rooms)


class ABCSwarm:
    """Represents a single swarm in the Artificial Bee Colony (ABC) algorithm."""
    
    def __init__(self, population_size, limit, num_swarms):
        self.population_size = population_size
        self.limit = limit
        self.num_swarms = num_swarms
        self.solution_set = [self.produce_solution() for _ in range(population_size)]
        self.fitness_set = [self.evaluate_fitness(sol) for sol in self.solution_set]
        print(self.fitness_set)
        self.onlooker_bee = [0] * population_size
        self.employed_bee = [0] * population_size
        self.stagnation = [0] * population_size
        self.abandoned = []
        self.global_best_solution = self.get_best_solution()

    def produce_solution(self):
        """Generate an initial feasible solution."""
        return copy.deepcopy(initialize_solution())

    def evaluate_fitness(self, solution):
        """Compute the fitness of a given solution."""
        return (
            room_capacity_cost(solution)
            + room_stability_cost(solution)
            + curriculum_compactness_cost(solution)
            + minimum_working_days_cost(solution)
        )

    def get_best_solution(self):
        """Retrieve the best solution in the swarm."""
        best_index = min(range(len(self.solution_set)), key=lambda i: self.fitness_set[i])
        return self.solution_set[best_index]

    def calculate_probability(self):
        """Calculate the selection probability for onlooker bees."""
        total_fitness = sum(1 / (fit + 1) for fit in self.fitness_set)
        probabilities = [(1 / (fit + 1)) / total_fitness for fit in self.fitness_set]
        return probabilities

    def employed_bee_phase(self):
        """Perform solution improvement for employed bees."""
        for bee in range(self.population_size):
            self.update(1, bee)

    def onlooker_bee_phase(self):
        """Move onlooker bees based on fitness probabilities."""
        probabilities = self.calculate_probability()

        for bee in range(self.population_size):
            position = random.randint(0, int(sum(probabilities) * 1000))
            for solution in range(len(self.solution_set)):
                position -= (1/(self.fitness_set[solution] +1)) * 1000
                if position <= 0:
                    self.onlooker_bee[bee] = solution
                    break

        for _ in range(3):
            for bee in range(self.population_size):
                self.update(2, bee)
                if self.stagnation[bee] >= self.limit and bee not in self.abandoned:
                    self.abandoned.append(bee)

    def scout_bee_phase(self):
        """Replace abandoned solutions with new ones."""
        if self.abandoned:
            for bee in self.abandoned:
                self.solution_set[bee] = self.produce_solution()
                self.stagnation[bee] = 0
            self.abandoned.clear()

    def stagnate(self):
        """Increase stagnation counter for each solution."""
        for i in range(self.population_size):
            self.stagnation[i] += 1

    def update(self, bee_type, bee_index):
        """Perform solution mutation for a given bee."""
        if bee_type == 1:
            index = bee_index
            solution = copy.deepcopy(self.solution_set[bee_index])
        else:
            index = self.onlooker_bee[bee_index]
            solution = copy.deepcopy(self.solution_set[self.onlooker_bee[bee_index]])

        course = -1
        lecture_num = 0
        cell = 0
        endSearch = False
        neighbor = random.choice(self.solution_set) #global_best_solution
        neighbor_cell = 0
        day, period, room = 0, 0, "-1"
        
        while course == -1:
            day = random.randint(0, days - 1)
            period = random.randint(0, periods_per_day - 1)
            room = list(rooms.keys())[random.randint(0, len(rooms) - 1)]
            course = solution[day][period][room]

        for d in range (days):
            for p in range (periods_per_day):
                for r in rooms:
                    cell+= 1
                    if solution[d][p][r] == course:
                        lecture_num += 1
                    if d == day and p == period and r == room:
                        endSearch = True
                        break
                    

                if endSearch: break
            if endSearch: break
        
        lect_count = lecture_num
        endSearch = False

        for d in range (days):
            for p in range (periods_per_day):
                for r in rooms:
                    neighbor_cell+= 1
                    if neighbor[d][p][r] == course:
                        lect_count -= 1
                    if lect_count == 0:
                        endSearch = True
                        break
                if endSearch: break
            if endSearch: break

        if bee_type == 1:    
            neighborhood_search_value = (round(cell + round(random.random() * (cell - neighbor_cell)))%(days_periods_rooms)) + 1
            
        elif bee_type == 2:
            r = 0.1 * (days_periods_rooms/self.num_swarms)
            neighborhood_search_value = (round(cell + round(random.random() * (r)))%(days_periods_rooms)) + 1
        
        slot_index = 0
        least_difference = float("inf")
        i = 0 

        if random.random() > 0.7:
            # MOVE
            available_slots = get_available_slots(course, solution, [day, period, room])
            if available_slots:
                for slot in available_slots:
                    new_cell = (slot[0] * (periods_rooms) )+ (slot[1] * len(rooms)) + getRoomIndex(slot[2])
                    if least_difference >= abs(new_cell - neighborhood_search_value):
                        slot_index = i
                        least_difference = abs(new_cell - neighborhood_search_value)
                    i += 1
                solution[day][period][room] = -1
                slot = available_slots[slot_index]
                solution[slot[0]][slot[1]][slot[2]] = course
        else:
            # SWAP
            swappable_slots = get_swappable_slots([course, day, period, room], solution)
            if swappable_slots:
                for slot in swappable_slots:
                    new_cell = (slot[0] * (periods_rooms) )+ (slot[1] * len(rooms)) + getRoomIndex(slot[2])
                    if least_difference >= abs(new_cell - neighborhood_search_value):
                        slot_index = i
                        least_difference = abs(new_cell - neighborhood_search_value)
                    i += 1
                slot = swappable_slots[slot_index]
                solution[day][period][room] = solution[slot[0]][slot[1]][slot[2]]
                solution[slot[0]][slot[1]][slot[2]] = course

        new_fitness = self.evaluate_fitness(solution)
        if new_fitness <= self.fitness_set[index]:
            self.fitness_set[index] = new_fitness
            self.solution_set[index] = solution
            self.stagnation[index] = 0 

    def run_cycle(self):
        """Run a full ABC cycle (employed bee, onlooker bee, scout bee, stagnation update)."""
        self.employed_bee_phase()
        self.onlooker_bee_phase()
        self.scout_bee_phase()
        self.stagnate()
        self.global_best_solution = self.get_best_solution()


class MultiSwarmABC:
    """Manages multiple ABC swarms running in parallel."""

    def __init__(self, num_swarms, population_size, max_iterations, limit):
        self.num_swarms = num_swarms
        self.max_iterations = max_iterations
        self.swarms = [ABCSwarm(population_size, limit, num_swarms) for _ in range(num_swarms)]

    def run(self):
        """Run all swarms for the specified number of iterations."""
        for cycle in range(self.max_iterations):
            for swarm in self.swarms:
                swarm.run_cycle()
            # Get best solutions from each swarm
            best_solutions = [swarm.global_best_solution for swarm in self.swarms]
            best_fitness = [swarm.evaluate_fitness(sol) for sol in best_solutions]
            print(f"Iteration {cycle+1}:")
            for i, fitness in enumerate(best_fitness):
                print(f"Swarm {i + 1} Best Fitness: {fitness}")

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
        print(bf)
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
        
def get_assigned_courses_by_period(day, period, timetable):
    courses = []
    for room in timetable[day][period]:
        if timetable[day][period][room] != -1:
            courses.append(timetable[day][period][room])
    return courses