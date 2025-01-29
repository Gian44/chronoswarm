from ctt_parser import read_ctt_file
from initialize_population2 import assign_courses
from deap import base, creator, tools
from model import *
from config import *
import random
import copy

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Particle", dict, fitness=creator.FitnessMin, speed=dict, best=None, bestfit=None, is_quantum=False)
creator.create("Swarm", list, best=None, bestfit=None)

toolbox = base.Toolbox()

filename = INPUT  # Replace with your .ctt file name
courses, rooms, unavailability_constraints, curricula, days, periods_per_day= read_ctt_file(filename)

periods_rooms = periods_per_day * len(rooms)
days_periods_rooms = periods_rooms * days
abandoned = []
limit = 1000
maximum_cycles = 0
solution_set = [] #The set of solutions
fitness_set = [] #Marks the fitness of each solution given the index
onlooker_bee = [] #Marks the position of each onlooker bee
employed_bee = []  #Marks the position of each employed bee
stagnation = [] #Marks the iterations where the solution did not improve
total_fitness = 0
population  = 0
global_best_solution = {}

def generate(pclass):
    schedule = None
    
    # Keep trying to assign courses until a valid schedule is generated
    while schedule is None:
        schedule = assign_courses()  # Generate the initial timetable
    
    return pclass(copy.deepcopy(schedule))

toolbox.register("particle", generate, creator.Particle)

def get_abc_best_solution():
    best = float("inf")
    best_solution = {}
    for solution in range(len(solution_set)):
        fitness = evaluate_fitness(solution_set[solution])
        # print(f"{solution}: {fitness}")
        if (fitness) <= best:
            best = fitness
            best_solution = solution_set[solution]
    return (best_solution)

def get_abc_worst():
    worst = -1
    worst_solution = -1
    for solution in range(len(solution_set)):
        fitness = evaluate_fitness(solution_set[solution])
        if (fitness) >= worst: 
            worst = fitness
            worst_solution = solution
    return (worst_solution)
    
def calculate_probability():
    best = 100000
    global total_fitness
    total_fitness = 0
    for solution in range(population):
        fitness = evaluate_fitness(solution_set[solution])
        if (fitness) <= best: 
            best = fitness
        total_fitness += 1/(fitness+1)
        fitness_set[solution] = fitness

def evaluate_fitness(solution):
    return room_capacity_cost(solution) +  room_stability_cost(solution) + curriculum_compactness_cost(solution) + minimum_working_days_cost(solution)

def stagnate():
    for i in range(len(stagnation)):
        stagnation[i] = stagnation[i] + 1

def abandon(position):
    print("abandoning")
    abandoned.append(position)

def employed_bee_phase():
    for bee in range(len(employed_bee)-1):
            swap(1, bee)

def onlooker_bee_phase():
    calculate_probability()

    for bee in range(len(onlooker_bee)):  #Randomly move each bee based on fitness probability
        position = random.randint(0,int(total_fitness * 1000)) 
        for solution in range(len(solution_set)):
            position -= (1/(fitness_set[solution] +1)) * 1000
            if position <= 0: 
                # print(solution)
                onlooker_bee[bee] = solution
                break
            
    for _ in range(3):
        for bee in range(len(onlooker_bee)):
            swap(2, bee)
            if(stagnation[bee] == limit) and (bee not in abandoned): 
                abandon(bee)

def scout_bee_phase():
    if abandoned:
        for bee in abandoned:
            solution_set[bee] = toolbox.particle()
            stagnation[bee] = 0
            abandoned.pop()
            
def swap(type, bee): #types: (1)Employed (2)Onlooker
    index = 0
    if type == 1:
        index = bee
        solution = copy.deepcopy(solution_set[bee])
    else:
        index = onlooker_bee[bee]
        solution = copy.deepcopy(solution_set[onlooker_bee[bee]])
    #course = list(courses.keys())[random.randint(0,len(courses)-1)]
    course = -1
    lecture_num = 0
    cell = 0
    endSearch = False
    neighbor = random.choice(solution_set)#global_best_solution
    
    neighbor_cell = 0
    while (course == -1):
        day = random.randint(0, days-1)
        period = random.randint(0, periods_per_day-1)
        room = list(rooms.keys())[random.randint(0,len(rooms)-1)]
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
        
    neighborhood_search_value = (round(cell + round(random.random() * (cell - neighbor_cell)))%(days_periods_rooms)) + 1
        
    #print(str(cell) + ", " + str(neighbor_cell) + "\n")
    #print(periods_rooms)
    if get_available_slots(course, solution, [day, period, room]):
        
        
        #print (neighborhood_search_value)
        available_slots = get_available_slots(course, solution, [day, period, room])
        slot_index = 0
        least_difference = float("inf")
        i = 0 
        for slot in available_slots:
            new_cell = (slot[0] * (periods_rooms) )+ (slot[1] * len(rooms)) + getRoomIndex(slot[2])
            if least_difference >= abs(new_cell - neighborhood_search_value):
                slot_index = i
                least_difference = abs(new_cell - neighborhood_search_value)
            i += 1
        solution[day][period][room] = -1
        slot = available_slots[slot_index]
        solution[slot[0]][slot[1]][slot[2]] = course
        if (type==1 and evaluate_fitness(solution) <= fitness_set[bee]) or ( type == 2 and evaluate_fitness(solution) <= fitness_set[onlooker_bee[bee]]):
            fitness_set[index] = evaluate_fitness(solution)
            solution_set[index] = solution
            stagnation[index] = 0 #Reset
    elif get_swappable_slots(course, solution, [day, period, room]):
        available_slots = get_swappable_slots(course, solution, [day, period, room])
        slot_index = 0
        least_difference = float("inf")
        i = 0 
        for slot in available_slots:
            new_cell = (slot[0] * (periods_rooms) )+ (slot[1] * len(rooms)) + getRoomIndex(slot[2])
            if least_difference >= abs(new_cell - neighborhood_search_value):
                slot_index = i
                least_difference = abs(new_cell - neighborhood_search_value)
            i += 1
        slot = available_slots[slot_index]
        solution[day][period][room] = solution[slot[0]][slot[1]][slot[2]]
        solution[slot[0]][slot[1]][slot[2]] = course
        if (type==1 and evaluate_fitness(solution) <= fitness_set[bee]) or ( type == 2 and evaluate_fitness(solution) <= fitness_set[onlooker_bee[bee]]):
            fitness_set[index] = evaluate_fitness(solution)
            solution_set[index] = solution
            stagnation[index] = 0 #Reset
                    
    

def produce_solution():
    return assign_courses()

#*********Utils**********#        

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

def get_swappable_slots(course, timetable, constraint_period=[-1,-1,-1]):
    available_slots = []
    for day in timetable:
        for period in timetable[day]:
            hasConflict = 0
            conflict_course = ""
            if day == constraint_period[0] and period == constraint_period[1]: hasConflict = True
            for target_course in get_assigned_courses_by_period(day, period, timetable):
                if has_conflict(course, target_course, timetable):
                    conflict_course = target_course
                    hasConflict += 1
            if hasConflict <= 1:
                for room in timetable[day][period]:
                    if has_conflict == 0 or timetable[day][period][room] == conflict_course: 
                        slot = [day, period, room]
                        isValid = True
                        if course in unavailability_constraints and (day, period) in unavailability_constraints[course]:
                            isValid = False
                            break
                        if timetable[day][period][room] != -1 and isValid:
                            if get_available_slots(timetable[day][period][room], timetable, [day, period, room]):
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
    
def abc(solution_set_param, maximum_cycles_param, limit_param, retain_state=False):
    """
    Initialize or reuse data for the Artificial Bee Colony algorithm.

    Args:
        solution_set_param: The current swarm of solutions.
        maximum_cycles_param: Maximum cycles for the ABC algorithm.
        limit_param: Limit for stagnation.
        retain_state: If True, retain existing data (fitness_set, onlooker_bee, etc.).
    """
    global solution_set, maximum_cycles, limit
    global fitness_set, onlooker_bee, employed_bee, stagnation

    # Retain solution set and parameters
    solution_set = solution_set_param
    maximum_cycles = maximum_cycles_param
    limit = limit_param

    # Initialize or retain data for fitness and bee positions
    if not retain_state or not fitness_set:  # Initialize only if state retention is off or no prior state exists
        fitness_set = [0] * len(solution_set)
        onlooker_bee = [""] * len(solution_set)
        employed_bee = [""] * len(solution_set)
        stagnation = [0] * len(solution_set)

    # Compute initial fitness if fitness_set is still empty
    if all(f == 0 for f in fitness_set):
        for i, solution in enumerate(solution_set):
            fitness_set[i] = evaluate_fitness(solution)

def cycle_abc():

    employed_bee_phase()
    onlooker_bee_phase()
    scout_bee_phase()
    stagnate()

    global global_best_solution
    global_best_solution = get_abc_best_solution()
    return global_best_solution

    

def kem_abc(best_solution):
    solution_set[get_abc_worst()] = best_solution