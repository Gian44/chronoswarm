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

class ABCSwarm:
    """Represents a single swarm in the Multi-Swarm Artificial Bee Colony (MSABC)."""

    def __init__(self, population_size, limit, num_swarms):
        self.population_size = population_size
        self.limit = limit
        self.num_swarms = num_swarms
        self.solution_set = [self.produce_solution() for _ in range(population_size)]
        self.fitness_set = [self.evaluate_fitness(sol) for sol in self.solution_set]
        self.depth_set = [self.evaluate_fitness(sol) for sol in self.solution_set]
        print(self.fitness_set)
        self.onlooker_bee = []
        self.employed_bee = [0] * population_size
        self.stagnation = [0] * population_size
        self.abandoned = []
        self.global_best_solution = self.get_best_solution()
        self.reinitialize_flag = False  

        # NMGD 
        self.beta = 0.05
        self.alpha = 0.05
        self.gamma = 0.05
        self.gd_iteration = 50

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
        """Retrieve the worst solution in the swarm."""
        worst = -1
        worst_solution = -1
        for solution in range(len(self.solution_set)):
            fitness = self.fitness_set[solution]
            if fitness >= worst: 
                worst = fitness
                worst_solution = solution
        return self.solution_set[worst_solution]
    
    def get_second_worst(self):
        worst = self.evaluate_fitness(self.get_worst_solution())
        best = self.evaluate_fitness(self.get_best_solution())
        second_solution = -1
        for solution in range(len(self.solution_set)):
            fitness = self.fitness_set[solution]
            if fitness > best and fitness < worst: 
                best = fitness
                second_solution = solution
        return self.solution_set[second_solution]

    def calculate_probability(self):
        """Calculate selection probabilities for onlooker bees."""
        total_fitness = sum(1 / (fit + 1) for fit in self.fitness_set)
        return total_fitness

    def employed_bee_phase(self):
        """Perform solution improvement for employed bees."""
        for bee in range(self.population_size):
            self.update(1, bee)
    
    def nelder_mead_values(self):
        # print(self.fitness_set)
        # print("Best: ", eval_fitness(self.get_best_solution()))
        # print("2nd Worst: ", self.evaluate_fitness (self.get_second_worst()))
        # print("Worst: ", self.evaluate_fitness(self.get_worst_solution()))
        nm_values = []

        c_cent = (self.get_fitness() + self.evaluate_fitness (self.get_second_worst())+ self.evaluate_fitness(self.get_worst_solution()))/3
        ec = c_cent - (self.beta * c_cent)
        r = ec - (self.alpha * c_cent)
        e = r - (self.gamma * c_cent)

        ec_r = (ec - r) / 3
        ec1 = ec-ec_r
        ec2 = ec1-ec_r
        ec3 = ec2-ec_r

        r_e = (r - e) / 3
        r1 = r-r_e
        r2 = r1-r_e
        r3 = r2-r_e

        nm_values.append(ec)
        nm_values.append(ec1)
        nm_values.append(ec2)
        nm_values.append(ec3)
        nm_values.append(r)
        nm_values.append(r1)
        nm_values.append(r2)
        nm_values.append(r3)
        nm_values.append(e)

        return nm_values
    
    def calculate_dr(self, fitness, nm_values, iteration):
        dr_values = []
        for nm_value in nm_values:
            dr = (fitness - nm_value) / iteration
            dr_values.append(dr)
        return dr_values


    def onlooker_bee_phase(self):
        """Move onlooker bees based on fitness probabilities."""
        # probabilities = self.calculate_probability()
        # probability_set = []
        # onlooker = []
        # iteration_count = []
        # i_base = 1
        # alpha = 50
        nm_values = self.nelder_mead_values()
        #print(nm_values)
        
        for solution in range(len(self.solution_set)):
            sol_i, best = self.solution_set[solution], self.solution_set[solution]
            sol_best = self.fitness_set[solution]
            level = self.fitness_set[solution]
            decay_rates = self.calculate_dr(sol_best, nm_values, self.gd_iteration)
            #print(decay_rates)
            for _ in range(self.gd_iteration):
                valid_indices = [i for i in range(len(nm_values)) if nm_values[i] <= sol_best]
                if valid_indices:
                    index_q = min(valid_indices, key=lambda i: abs(sol_best - nm_values[i]))  
                else:
                    index_q = nm_values.index(min(nm_values))  
                beta = abs(decay_rates[index_q]) # decay rate
                # if beta < 0:
                #     print("negative decay: ", beta)
                new_solution = copy.deepcopy(sol_i)
                self.update_gd(new_solution)
                fitness = self.evaluate_fitness(new_solution)
                if fitness <= sol_best:
                    if fitness < sol_best:
                        self.stagnation[solution] = 0
                    sol_i = new_solution
                    best = new_solution
                    sol_best = fitness
                elif fitness < level:
                    sol_i = new_solution
                level -= beta

            self.solution_set[solution] = best
            self.fitness_set[solution] = sol_best

            if self.stagnation[solution] >= self.limit and solution not in self.abandoned:
                self.abandoned.append(solution)
                print("ABANDON")    


        # for solution in range(len(self.solution_set)):
        #     probability_frac = 1/(self.fitness_set[solution]+1)/probabilities
        #     # print(probability_frac)
        #     probability = round(((1/(self.fitness_set[solution]+1))/probabilities) * (self.population_size*2))
        #     # print(probability)
        #     probability_set.append(probability)
        #     for probability in range(probability_set[solution]):
        #         self.onlooker_bee.append(copy.deepcopy(self.solution_set[solution]))
        #         onlooker.append(solution)
        #         iteration_count.append(round(i_base + (alpha * probability_frac)))
                
        # for bee in range(len(self.onlooker_bee)):
        #     for _ in range(iteration_count[bee]):
        #         self.update2(2, bee, onlooker[bee])
        #     fitness = self.evaluate_fitness(self.onlooker_bee[bee])
        #     if fitness <= self.fitness_set[onlooker[bee]]:
        #         self.solution_set[onlooker[bee]] = self.onlooker_bee[bee]
        #         self.fitness_set[onlooker[bee]] = fitness

        # for bee in range(len(self.solution_set)):
        #     if self.stagnation[bee] >= self.limit and bee not in self.abandoned:
        #             self.abandoned.append(bee)
        #             print("ABANDON")

        # self.onlooker_bee.clear()

    def scout_bee_phase(self):
        if self.abandoned:
            for pos in range(len(self.abandoned)):
                bee = self.abandoned[pos]
                self.scout(bee)
                self.stagnation[bee] = 0
            self.abandoned.clear()

    def scout(self, bee):
        """Replace abandoned solutions with new ones using 30% modification while satisfying room capacity and stability."""
        print(f"Bee {bee} abandoned and replaced with new solution")
        print(f"Fitness Now: {self.fitness_set[bee]}")

        solution = self.solution_set[bee]
        lectures_to_modify = math.ceil(0.3 * days_periods_rooms)  # 30% modification
        fitness = self.evaluate_fitness(solution)
        scout_fitness = self.evaluate_scout_fitness(solution)
        modified_count = 0
        attempts = 0
        new_solution = copy.deepcopy(solution)

        while modified_count < lectures_to_modify and attempts < 1000:
            course = -1
            day, period, room = -1, -1, "-1"

            while course == -1:
                day = random.randint(0, days - 1)
                period = random.randint(0, periods_per_day - 1)
                room = random.choice(list(rooms.keys()))
                course = new_solution[day][period][room]

            if self.solution_set[bee] != self.get_best_solution():
                
                available_slots = get_available_slots(course, new_solution, [day, period, room])
                valid_slots = available_slots

                if valid_slots:
                    chosen_slot = random.choice(valid_slots)
                    new_solution[day][period][room] = -1
                    new_solution[chosen_slot[0]][chosen_slot[1]][chosen_slot[2]] = course
                    if self.evaluate_scout_fitness(new_solution) > scout_fitness:
                        new_solution[day][period][room] = new_solution[chosen_slot[0]][chosen_slot[1]][chosen_slot[2]]
                        new_solution[chosen_slot[0]][chosen_slot[1]][chosen_slot[2]] = -1
                    else:
                        modified_count += 1
            else:
                
                available_slots = get_available_slots(course, new_solution, [day, period, room])
                valid_slots = available_slots

                if valid_slots:
                    chosen_slot = random.choice(valid_slots)
                    new_solution[day][period][room] = -1
                    new_solution[chosen_slot[0]][chosen_slot[1]][chosen_slot[2]] = course
                    if self.evaluate_fitness(new_solution) > fitness:
                        new_solution[day][period][room] = new_solution[chosen_slot[0]][chosen_slot[1]][chosen_slot[2]]
                        new_solution[chosen_slot[0]][chosen_slot[1]][chosen_slot[2]] = -1
                    else:
                        modified_count += 1
     
            attempts += 1

        if self.solution_set[bee] != self.get_best_solution():
            print("Modified")
            new_fitness = self.evaluate_scout_fitness(new_solution)
            if new_fitness <= scout_fitness:
                self.solution_set[bee] = new_solution
                self.fitness_set[bee] = self.evaluate_fitness(new_solution)
                print(f"Fitness After: {self.fitness_set[bee]}")
            else:
                print("No scout found")
        else:
            print("Modified 2")
            new_fitness = self.evaluate_fitness(new_solution)
            self.solution_set[bee] = new_solution
            self.fitness_set[bee] = new_fitness
            print(f"Fitness After: {self.fitness_set[bee]}")

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

        # fitness_i = self.fitness_set[bee_index]
        # fitness_j = self.fitness_set[neighbor_index]
        # w = 0

        # if fitness_i > fitness_j:
        #     w = ((fitness_i * course_value) - (fitness_j * neighbor_value)) / (fitness_i + fitness_j)
        # else:
        #     w = ((fitness_i * course_value) + (fitness_j * neighbor_value)) / (fitness_i + fitness_j)

        neighborhood_search_value = (round(course_value + round(random.random() * (course_value - neighbor_value)))%(len(courses_index)))
        #neighborhood_search_value = (round(cell + round(random.random() * (best_cell - cell + neighbor_cell1 - neighbor_cell2)))%(days_periods_rooms)) + 1
        #neighborhood_search_value = (round(w + round(random.random() * (cell - neighbor_cell1)))%(days_periods_rooms)) + 1
        #neighborhood_search_value = (round(w + round(random.random() * (course_value - neighbor_value)))%(len(courses_index)))

        #neighborhood_search_value = (round(cell + round((random.random() * (neighbor_cell1 - cell)) + (random.uniform(0, 1.5) * (best_cell-cell))))%(days_periods_rooms)) + 1

        new_solution = copy.deepcopy(solution)
        conflict = False
        swapped1 = False
        swapped2 = False
        cslots = []

        #print("Course: " + str(course) + " Neighbor Course: " + str(courses_index[neighborhood_search_value]))

        for r in rooms:
            #print(str(new_solution[day][period][r]) + " and " + str(courses_index[neighborhood_search_value]))
            if new_solution[day][period][r] == courses_index[neighborhood_search_value] and room != r:
                new_solution[day][period][room] = courses_index[neighborhood_search_value]
                new_solution[day][period][r] = course
                #print("Solution Swapped (Before swap 1)")
                swapped1 = True
                swapped2 = True
                break
        
        if not swapped1:
            if courses_index[neighborhood_search_value] != -1:
                for target_course in get_assigned_courses_by_period(day, period, new_solution):
                    if has_conflict(courses_index[neighborhood_search_value], target_course, new_solution):
                        conflict = True
                        break

            if (courses_index[neighborhood_search_value] in unavailability_constraints and (day, period) in unavailability_constraints[courses_index[neighborhood_search_value]]) or conflict:
                #print("unavailable or conflict")
                for d in range (days):
                    for p in range (periods_per_day):
                        for r in rooms:
                            #print(str(new_solution[d][p][r]) + " and " + str(courses_index[neighborhood_search_value]))
                            if solution[d][p][r] == courses_index[neighborhood_search_value]:
                                cslots.append([d, p, r])

                #print("cslots", cslots)
                cslot = random.choice(cslots)
                available_slots = get_available_slots(courses_index[neighborhood_search_value], solution)
                if available_slots:
                    slot = random.choice(available_slots)
                    new_solution[cslot[0]][cslot[1]][cslot[2]] = -1
                    new_solution[slot[0]][slot[1]][slot[2]] = courses_index[neighborhood_search_value]

            elif courses_index[neighborhood_search_value] == -1:
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
                                #print(str(new_solution[d][p][r]) + " and " + str(courses_index[neighborhood_search_value]))
                                if new_solution[d][p][r] == courses_index[neighborhood_search_value]:
                                    cslots.append([d, p, r])

                    #print("cslots", cslots)
                    cslot = random.choice(cslots)
                    
                    if emp_slots:
                        emp_slot = random.choice(emp_slots)
                        new_solution[cslot[0]][cslot[1]][cslot[2]] = -1
                        new_solution[day][period][room] = courses_index[neighborhood_search_value]
                        new_solution[emp_slot[0]][emp_slot[1]][emp_slot[2]] = course
                        #print("Solution Swapped (Before swap 2)")
                        swapped2 = True

                if not swapped2:
                    for d in range (days):
                        for p in range (periods_per_day):
                            for r in rooms:
                                #print(str(new_solution[d][p][r]) + " and " + str(courses_index[neighborhood_search_value]))
                                if new_solution[d][p][r] == courses_index[neighborhood_search_value]:
                                    cslots.append([d, p, r])

                    #print("cslots", cslots)
                    cslot = random.choice(cslots)
                    available_slots = get_available_slots(course, solution, cslot)
                    if available_slots:
                        slot = random.choice(available_slots)
                        new_solution[cslot[0]][cslot[1]][cslot[2]] = -1
                        new_solution[day][period][room] = courses_index[neighborhood_search_value]
                        new_solution[slot[0]][slot[1]][slot[2]] = course
                        #print("Solution Swapped (in swap 3)")
   
        new_fitness = self.evaluate_fitness(new_solution)

        if new_fitness <= self.fitness_set[index] and solution != new_solution:
            if new_fitness <= self.fitness_set[index]:
                self.stagnation[index] = 0
            self.fitness_set[index] = new_fitness
            self.solution_set[index] = new_solution
            
        # print(self.stagnation)

    def update_gd(self, solution):

        #course = list(courses.keys())[random.randint(0,len(courses)-1)]
        course = -1
        neighbor_index = random.randint(0, len(self.solution_set) -1)
        neighbor = self.solution_set[neighbor_index]
        day, period, room = 0, 0, "-1"
        
        while (course == -1):
            day = random.randint(0, days-1)
            period = random.randint(0, periods_per_day-1)
            room = list(rooms.keys())[random.randint(0,len(rooms)-1)]
            course = solution[day][period][room]

        # print(periods_rooms)
        rnd = random.randint(1, 6) 
        if rnd == 1: #N1 change day_period
            available_slots = get_available_period(course, solution, [day, period, room])
            if available_slots:
                solution[day][period][room] = -1
                slot = random.choice(available_slots)
                solution[slot[0]][slot[1]][slot[2]] = course
            
        elif rnd == 2: # N2
            available_rooms = []
            for r in solution[day][period]:
                if solution[day][period][r] == -1:
                    available_rooms.append(r)
            if available_rooms:
                solution[day][period][room] = -1
                solution[day][period][random.choice(available_rooms)] = course
                
        elif rnd == 3: #N3 Cant be the same period or room
            available_slots = get_available_slots_different_period_room(course, solution, [day, period, room])
            if available_slots:
                solution[day][period][room] = -1
                slot = random.choice(available_slots)
                solution[slot[0]][slot[1]][slot[2]] = course

        elif rnd == 4:   #N4 
            select_course = random.choice(list(courses.keys()))
            select_room = random.choice(list(rooms.keys()))
            for day in solution:
                for period in solution[day]:
                    for room in solution[day][period]:
                        if solution[day][period][room] == select_course:
                            solution[day][period][room] = solution[day][period][select_room]
                            solution[day][period][select_room] = select_course
                            
        elif rnd == 5 : #N5 Move not Swap
            violating_courses, violating_courses_assignment = get_courses_with_mwd_violations(solution)
            if violating_courses:
                violating_course = random.choice(violating_courses)
                slots = violating_courses_assignment[violating_course]
                orig_slot = random.choice(list(slots))
                available_slots = get_available_slots(violating_course, solution, [orig_slot[0], orig_slot[1], orig_slot[2]])
                if available_slots:
                    slot = random.choice(available_slots)
                    solution[orig_slot[0]][orig_slot[1]][orig_slot[2]] = -1
                    solution[slot[0]][slot[1]][slot[2]] = violating_course
                    n5 = True
                    
        elif rnd == 6: #N6 copy the move in 6
            curriculum = random.choice(list(curricula))
            violating_lectures = []
            for day in solution:
                for period in solution[day]:
                    for room in solution[day][period]:
                        course = solution[day][period][room]
                        if course in curricula[curriculum]:
                            hasUpper = False
                            hasLower = False
                            if period<(periods_per_day-1) and solution[day][period+1]:
                                for coursej in solution[day][period+1].values(): 
                                    if (coursej in curricula[curriculum]):
                                        hasUpper = True
                                        break
                            if not hasUpper and period>0 and solution[day][period-1]:
                                for coursej in solution[day][period-1].values(): 
                                    if (coursej in curricula[curriculum]):
                                        hasLower = True
                                        break
                            if (not hasUpper) and (not hasLower): 
                                violating_lectures.append((course, day, period, room))
            if  violating_lectures:
                lecture = random.choice(violating_lectures)
                course = lecture[0]
                available_period = []
                for day in solution:
                    for period in solution[day]:
                        hasConflict = False
                        for target_course in get_assigned_courses_by_period(day, period, solution):
                            if has_conflict(course, target_course, solution):
                                hasConflict = True
                                break
                        if  (not(course in unavailability_constraints and (day, period) in unavailability_constraints[course])) and not hasConflict:
                            if period<(periods_per_day-1) and solution[day][period+1]:
                                for coursej in solution[day][period+1].values(): 
                                    if (coursej in curricula[curriculum]):
                                        if solution[day][period][lecture[3]] == -1:
                                            available_period.append((day, period, lecture[3]))
                                        else:
                                            for r in solution[day][period]:
                                                if solution[day][period] == 0:
                                                    available_period.append((day, period, r))
                                        break
                            if not hasUpper and period>0 and solution[day][period-1]:
                                for coursej in solution[day][period-1].values(): 
                                    if (coursej in curricula[curriculum]):
                                        if solution[day][period][lecture[3]] == -1:
                                            available_period.append((day, period, lecture[3]))
                                        else:
                                            for r in solution[day][period]:
                                                if solution[day][period] == 0:
                                                    available_period.append((day, period, r))
                                        break
                if available_period:
                    period = random.choice(available_period)
                    solution[lecture[1]][lecture[2]][lecture[3]] = -1
                    solution[period[0]][period[1]][lecture[3]] = lecture[0]

    def update2(self, type, bee, source): #types: (1)Employed (2)Onlooker
        n5 = False
        index = 0
        if type == 1:
            index = bee
            solution = copy.deepcopy(self.solution_set[bee])
        else:
            index = source
            solution = copy.deepcopy(self.onlooker_bee[bee])
        #course = list(courses.keys())[random.randint(0,len(courses)-1)]
        course = -1
        neighbor_index = random.randint(0, len(self.solution_set) -1)
        neighbor = self.solution_set[neighbor_index]
        day, period, room = 0, 0, "-1"
        
        while (course == -1):
            day = random.randint(0, days-1)
            period = random.randint(0, periods_per_day-1)
            room = list(rooms.keys())[random.randint(0,len(rooms)-1)]
            course = solution[day][period][room]

        # print(periods_rooms)
        rnd = random.randint(1, 6) 
        if rnd == 1: #N1 change day_period
            available_slots = get_available_period(course, solution, [day, period, room])
            if available_slots:
                solution[day][period][room] = -1
                slot = random.choice(available_slots)
                solution[slot[0]][slot[1]][slot[2]] = course
            
        elif rnd == 2: # N2
            available_rooms = []
            for r in solution[day][period]:
                if solution[day][period][r] == -1:
                    available_rooms.append(r)
            if available_rooms:
                solution[day][period][room] = -1
                solution[day][period][random.choice(available_rooms)] = course
                
        elif rnd == 3: #N3 Cant be the same period or room
            available_slots = get_available_slots_different_period_room(course, solution, [day, period, room])
            if available_slots:
                solution[day][period][room] = -1
                slot = random.choice(available_slots)
                solution[slot[0]][slot[1]][slot[2]] = course

        elif rnd == 4:   #N4 
            select_course = random.choice(list(courses.keys()))
            select_room = random.choice(list(rooms.keys()))
            for day in solution:
                for period in solution[day]:
                    for room in solution[day][period]:
                        if solution[day][period][room] == select_course:
                            solution[day][period][room] = solution[day][period][select_room]
                            solution[day][period][select_room] = select_course
                            
        elif rnd == 5 : #N5 Move not Swap
            violating_courses, violating_courses_assignment = get_courses_with_mwd_violations(solution)
            if violating_courses:
                violating_course = random.choice(violating_courses)
                slots = violating_courses_assignment[violating_course]
                orig_slot = random.choice(list(slots))
                available_slots = get_available_slots(violating_course, solution, [orig_slot[0], orig_slot[1], orig_slot[2]])
                if available_slots:
                    slot = random.choice(available_slots)
                    solution[orig_slot[0]][orig_slot[1]][orig_slot[2]] = -1
                    solution[slot[0]][slot[1]][slot[2]] = violating_course
                    n5 = True
                    
        elif rnd == 6: #N6 copy the move in 6
            curriculum = random.choice(list(curricula))
            violating_lectures = []
            for day in solution:
                for period in solution[day]:
                    for room in solution[day][period]:
                        course = solution[day][period][room]
                        if course in curricula[curriculum]:
                            hasUpper = False
                            hasLower = False
                            if period<(periods_per_day-1) and solution[day][period+1]:
                                for coursej in solution[day][period+1].values(): 
                                    if (coursej in curricula[curriculum]):
                                        hasUpper = True
                                        break
                            if not hasUpper and period>0 and solution[day][period-1]:
                                for coursej in solution[day][period-1].values(): 
                                    if (coursej in curricula[curriculum]):
                                        hasLower = True
                                        break
                            if (not hasUpper) and (not hasLower): 
                                violating_lectures.append((course, day, period, room))
            if  violating_lectures:
                lecture = random.choice(violating_lectures)
                course = lecture[0]
                available_period = []
                for day in solution:
                    for period in solution[day]:
                        hasConflict = False
                        for target_course in get_assigned_courses_by_period(day, period, solution):
                            if has_conflict(course, target_course, solution):
                                hasConflict = True
                                break
                        if  (not(course in unavailability_constraints and (day, period) in unavailability_constraints[course])) and not hasConflict:
                            if period<(periods_per_day-1) and solution[day][period+1]:
                                for coursej in solution[day][period+1].values(): 
                                    if (coursej in curricula[curriculum]):
                                        if solution[day][period][lecture[3]] == -1:
                                            available_period.append((day, period, lecture[3]))
                                        else:
                                            for r in solution[day][period]:
                                                if solution[day][period] == 0:
                                                    available_period.append((day, period, r))
                                        break
                            if not hasUpper and period>0 and solution[day][period-1]:
                                for coursej in solution[day][period-1].values(): 
                                    if (coursej in curricula[curriculum]):
                                        if solution[day][period][lecture[3]] == -1:
                                            available_period.append((day, period, lecture[3]))
                                        else:
                                            for r in solution[day][period]:
                                                if solution[day][period] == 0:
                                                    available_period.append((day, period, r))
                                        break
                if available_period:
                    period = random.choice(available_period)
                    solution[lecture[1]][lecture[2]][lecture[3]] = -1
                    solution[period[0]][period[1]][lecture[3]] = lecture[0]

        new_fitness = self.evaluate_fitness(solution)
        personal_fitness = 0 

        if type == 1: personal_fitness = self.fitness_set[bee]
        else: personal_fitness = self.evaluate_fitness(self.onlooker_bee[bee])
        
        if type == 1:
            if (new_fitness <= personal_fitness):
                if (new_fitness < personal_fitness):
                    self.stagnation[index] = 0 #Reset
                self.fitness_set[index] = self.evaluate_fitness(solution)
                self.solution_set[index] = solution
                #if n5: print("Done N5")
        else:
            if (new_fitness <= personal_fitness):
                if (new_fitness < personal_fitness):
                    self.stagnation[index] = 0 #Reset
                self.onlooker_bee[bee] = solution
                #if n5: print("Done N5")

    def run_cycle(self):
        """Run a full ABC cycle."""
        self.employed_bee_phase()

        self.onlooker_bee_phase()

        self.scout_bee_phase()
        
        self.stagnate()

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
        self.swarms = [ABCSwarm(population_size, limit, num_swarms) for _ in range(num_swarms)]
        self.global_best_solution = None
        self.global_best_fitness = float("inf")

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

    def convertQuantum(self, swarm: ABCSwarm):
        """Quantum reinitialization of a swarm using Gaussian movement."""

        valid, invalid = 0,0
        centre = swarm.global_best_solution  # Use the best solution as the center
        new_solution_set = []  # Store new solutions

        courses_index = list(courses.keys())
        courses_index.append(-1)
    

        for part in swarm.solution_set:
            new_solution = copy.deepcopy(part)  # Copy solution before modification

            for day, periods_dict in centre.items():
                for period, rooms_dict in periods_dict.items():
                    for room_id, best_course in rooms_dict.items():
                        if best_course == -1:
                            continue  # Skip unassigned slots
                        
                        best_course_index = courses_index.index(best_course)
                        # Apply Gaussian movement
                        new_course_index = round(best_course_index + (self.rcloud * random.gauss(0, 1))) % len(courses_index)
                        new_course = courses_index[new_course_index]

                        cslots = []
                        conflict = False

                        if new_course == -1:
                            available_slots = get_available_slots(course, new_solution)
                            if available_slots:
                                slot = random.choice(available_slots)
                                new_solution[day][period][room_id] = -1
                                new_solution[slot[0]][slot[1]][slot[2]] = course
                        else:
                            for target_course in get_assigned_courses_by_period(day, period, new_solution):
                                if has_conflict(new_course, target_course, new_solution):
                                    conflict = True
                                    break

                            if (new_course in unavailability_constraints and (day, period) in unavailability_constraints[new_course]) or conflict:
                                #print("unavailable or conflict")
                                for d in range (days):
                                    for p in range (periods_per_day):
                                        for r in rooms:
                                            #print(str(new_solution[d][p][r]) + " and " + str(courses_index[neighborhood_search_value]))
                                            if new_solution[d][p][r] == new_course:
                                                cslots.append([d, p, r])

                                #print("cslots", cslots)
                                cslot = random.choice(cslots)
                                available_slots = get_available_slots(new_course, new_solution, cslots)
                                if available_slots:
                                    slot = random.choice(available_slots)
                                    new_solution[cslot[0]][cslot[1]][cslot[2]] = -1
                                    new_solution[slot[0]][slot[1]][slot[2]] = new_course
                            else:
                                for d in range (days):
                                    for p in range (periods_per_day):
                                        for r in rooms:
                                            #print(str(new_solution[d][p][r]) + " and " + str(courses_index[neighborhood_search_value]))
                                            if new_solution[d][p][r] == new_course:
                                                cslots.append([d, p, r])

                                #print("cslots", cslots)
                                cslot = random.choice(cslots)
                                available_slots = get_available_slots(course, new_solution, cslot)
                                if available_slots:
                                    slot = random.choice(available_slots)
                                    new_solution[cslot[0]][cslot[1]][cslot[2]] = -1
                                    new_solution[day][period][room_id] = new_course
                                    new_solution[slot[0]][slot[1]][slot[2]] = course 

                        # Validate feasibility
                        # if not is_feasible(new_solution, unavailability_constraints, courses, curricula):
                        #     new_solution[new_day][new_period][new_room] = new_entry  
                        #     new_solution[day][period][room_id] = original_entry
                        #     invalid += 1
                        #     #print("Not feasible")  
                        # else:
                        #     print("Feasible")
                        #     valid +=1
            # if new_solution == part:
            #     print("equal (No change detected in reinitialized solution!)")
            # else:
            #     print(f"not equal (Changes made, checking differences...)")
            #     # for d in range(days):
            #     #     for p in range(periods_per_day):
            #     #         for r in rooms:
            #     #             if new_solution[d][p].get(r, -1) != part[d][p].get(r, -1):
            #     #                 print(f"Changed: {part[d][p].get(r, -1)} → {new_solution[d][p].get(r, -1)} at ({d}, {p}, {r})")

            # # Ensure new solutions are assigned
            # print("Invalid: ", invalid)
            # print("Valid: ", valid)
            new_solution_set.append(new_solution)
            valid, invalid = 0, 0

        # ✅ Overwrite entire swarm with new solutions
        swarm.solution_set = new_solution_set
        swarm.fitness_set = [swarm.evaluate_fitness(sol) for sol in swarm.solution_set]

        swarm.reinitialize_flag = False  # Reset flag after quantum reinitialization


    def run(self):
        """Run all swarms for the specified number of iterations with exclusion and anti-convergence."""
        # ✅ Compute convergence & exclusion radii first
        rconv = (days_periods_rooms / len(self.swarms)) ** (1 / 3)
        rexcl = rconv
        print("Radius: ", rconv)
        worst_fitness = 0

        best_fitness = [swarm.get_fitness() for swarm in self.swarms]
        for i, fitness in enumerate(best_fitness):
            if fitness <= self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_solution = self.swarms[i].global_best_solution
        print(f"Global Best Fitness: {self.global_best_fitness}")

        all_converged = False
        for cycle in range(self.max_iterations):
            # ✅ **Step 1: Anti-Convergence**
            for swarm in self.swarms:
                swarm_converged = all(
                    self.calculate_distance(swarm.solution_set[i], swarm.solution_set[j]) <= 2 * rconv
                    for i, j in itertools.combinations(range(len(swarm.solution_set)), 2)
                )
                
                if not swarm_converged:
                    all_converged = False  # If at least one solution has not converged, update flag

            # 🔹 If all solutions in the swarm have converged → Tag swarm for reinitialization
            if all_converged:
                for i, swarm in enumerate(self.swarms):
                    fitness = swarm.evaluate_fitness(swarm.get_best_solution())
                    if fitness > worst_fitness:
                        worst_swarm = swarm
                        worst_fitness = fitness
                        num = i+1
                print(f"Tagging swarm {num} for quantum reinitialization due to convergence.")
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
                    #print("Swarm " + str(swarm) + " for reinitialization")
                    self.convertQuantum(swarm)  # Quantum Reinitialization
                else:
                    #print("Swarm " + str(swarm) + " for cycle")
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
                print(str(swarm.fitness_set) + " " + str(swarm.evaluate_fitness(best)))
        
        return self.global_best_solution, self.global_best_fitness
    
    def get_fitness_per_swarm(self):
        for i, swarm in enumerate(self.swarms):
            print(f"Swarm {i+1}: " + str(swarm.fitness_set))

    def get_global_best(self, best_solutions):
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