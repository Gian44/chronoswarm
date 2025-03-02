from ctt_parser import read_ctt_file
from initialize_population import initialize_solution
from model import * 
import random
import copy
import threading

filename = 'input.ctt'  # Replace with your .ctt file name
courses, rooms, unavailability_constraints, curricula, days, periods_per_day= read_ctt_file(filename)
periods_rooms = periods_per_day * len(rooms)
days_periods_rooms = periods_rooms * days
abandoned = []
limit = 200
maximum_cycles = 0
solution_set = [] #The set of solutions
fitness_set = [] #Marks the fitness of each solution given the index
depth_set = []
onlooker_bee = [] #Marks the position of each onlooker bee
employed_bee = []  #Marks the position of each employed bee
stagnation = [] #Marks the iterations where the solution did not improve
total_fitness = 0
population  = 0
global_best_solution = {}

total_lectures = 0

for course in courses:
    total_lectures += courses[course]['lectures']


def get_abc_best_solution():
    global global_best_solution
    best = evaluate_fitness(global_best_solution)
    for solution in range(len(solution_set)):
        fitness = evaluate_fitness(solution_set[solution])
        # print(f"{solution}: {fitness}")
        if (fitness) <= best:
            best = fitness
            global_best_solution = solution_set[solution]
    
    return (global_best_solution)

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
    return eval_fitness(solution)

def stagnate():
    for i in range(len(stagnation)):
        stagnation[i] = stagnation[i] + 1

def abandon(position):
    print("abandoning " + str(position))
    abandoned.append(position)

def employed_bee_phase():
    threads = []
    def employed_bee_int(bee):
        update(1, bee)
        # local_search(bee, 50)
    for bee in range(len(employed_bee)):
        thread = threading.Thread(target=employed_bee_int(bee))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

def onlooker_bee_phase():
    calculate_probability()
    threads = []
    def onlooker_bee_int(bee):
        swap(2, bee)
        if(stagnation[bee] >= limit) and (bee not in abandoned): 
            abandon(bee)
    for bee in range(len(onlooker_bee)):  #Randomly move each bee based on fitness probability
        position = random.randint(0,int(total_fitness * 1000))
        for solution in range(len(solution_set)):
            position -= (1/(fitness_set[solution] +1)) * 1000
            if position <= 0: 
                onlooker_bee[bee] = solution
                break
    for _ in range(10):
        
        for bee in onlooker_bee:
            thread = threading.Thread(target=onlooker_bee_int(bee))
            threads.append(thread)
            thread.start()

    for thread in threads:
        thread.join()

    for bee in range(len(employed_bee)):
        if(stagnation[bee] >= limit) and (bee not in abandoned): 
            abandon(bee)
            

def scout_bee_phase():
    if abandoned:
        for pos in range(len(abandoned)):
            bee = abandoned[pos]
            print(bee)
            scout(bee)
            stagnation[bee] = 0
        abandoned.clear()
            

def scout(bee):
    worst = evaluate_fitness(solution_set[get_abc_worst()])
    orig_min_days = minimum_working_days_cost(solution_set[bee])
    orig_curr_comp = curriculum_compactness_cost(solution_set[bee])
    for _ in range(1000):
        solution = copy.deepcopy(solution_set[bee])
        for _ in range(3):
            course = -1
            while (course == -1):
                day = random.randint(0, days-1)
                period = random.randint(0, periods_per_day-1)
                room = list(rooms.keys())[random.randint(0,len(rooms)-1)]
                course = solution[day][period][room]
            available_slots = get_available_slots(course, solution, [day, period, room])
            if available_slots:
                rnd = random.randint(0,len(available_slots)-1)
                solution[day][period][room] = -1
                slot = available_slots[rnd]
                solution[slot[0]][slot[1]][slot[2]] = course
        if evaluate_fitness(solution) <= depth_set[bee] and (orig_curr_comp >= curriculum_compactness_cost(solution) or orig_min_days >= minimum_working_days_cost(solution)):
            print("["+ str(bee)+"] Improved from "+ str(fitness_set) + " to "+ str(evaluate_fitness(solution)))
            solution_set[bee] = solution
            break


# def scout(bee):
#     for _ in range (round(days_periods_rooms * .3)):
#         solution = copy.deepcopy(solution_set[bee])
#         course = -1
#         while (course == -1):
#             day = random.randint(0, days-1)
#             period = random.randint(0, periods_per_day-1)
#             room = list(rooms.keys())[random.randint(0,len(rooms)-1)]
#             course = solution[day][period][room]
#         if get_available_slots(course, solution, [day, period, room]):
#             available_slots = get_available_slots(course, solution, [day, period, room])
#             rnd = random.randint(0,len(available_slots)-1)
#             solution[day][period][room] = -1
#             slot = available_slots[rnd]
#             solution[slot[0]][slot[1]][slot[2]] = course
#             solution_set[bee] = solution

def local_search(bee, iter):
    for _ in range(iter):
        isComplete = False
        day, period, room, course = 0, 0, "-1", -1
        day2, period2, room2, course2 = 0, 0, "-1", -1
        timetable = copy.deepcopy(solution_set[bee])
        while (course == -1):
            day = random.randint(0, days-1)
            period = random.randint(0, periods_per_day-1)
            room = list(rooms.keys())[random.randint(0,len(rooms)-1)]
            course = timetable[day][period][room]
        while (course2 == -1):
            day2 = random.randint(0, days-1)
            period2 = random.randint(0, periods_per_day-1)
            room2 = list(rooms.keys())[random.randint(0,len(rooms)-1)]
            course2 = timetable[day2][period2][room2]
       
        course_slot = [course2, day2, period2, room2]
        # print (str(course_slot)  + " " + str([course, day, period, room]))
        swappable_course = course
        hasSwappableConflict = False
        isSwappableValid = True
        hasConflict = False
        if swappable_course == -1: hasConflict = True
        if day == course_slot[1] and period == course_slot[2]: hasConflict = True
        for target_course in get_assigned_courses_by_period(day, period, timetable):
            if has_conflict(course_slot[0], target_course, timetable):
                # print(str(_) +" " + str(course_slot[0]) + " " + str(target_course))
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
                timetable[day][period][room] = course2
                timetable[day2][period2][room2] = course
                if evaluate_fitness(timetable) < fitness_set[bee]:
                    fitness_set[bee] = evaluate_fitness(timetable)
                    solution_set[bee] = timetable
                    stagnation[bee] = 0 #Reset
                    isComplete =  True
                    break
            if isComplete: break

def update(bee_type, bee_index):
    """Perform solution mutation for a given bee."""
    if bee_type == 1:
        index = bee_index
        solution = copy.deepcopy(solution_set[bee_index])
    else:
        index = onlooker_bee[bee_index]
        solution = copy.deepcopy(solution_set[onlooker_bee[bee_index]])

    course = -1
    lecture_num = 0
    cell = 0
    endSearch = False
    courses_index = list(courses.keys())
    courses_index.append(-1)
    neighbor_index = random.randint(0, len(solution_set)-1)  # Get index
    neighbor1 = solution_set[neighbor_index]  # Get corresponding solution
    neighbor2 = random.choice(solution_set) #global_best_solution
    best_neighbor = global_best_solution
    neighbor_cell1 = 0
    neighbor_cell2 = 0
    best_cell = 0
    day, period, room = 0, 0, "-1"
    
    while course == -1:
        day = random.randint(0, days - 1)
        period = random.randint(0, periods_per_day - 1)
        room = list(rooms.keys())[random.randint(0, len(rooms) - 1)]
        course = solution[day][period][room]
        
    neighbor_course = neighbor1[day][period][room]

    course_value = courses_index.index(course)
    neighbor_value = courses_index.index(neighbor_course)

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
                neighbor_cell1+= 1
                if neighbor1[d][p][r] == course:
                    lect_count -= 1
                if lect_count == 0:
                    endSearch = True
                    break
            if endSearch: break
        if endSearch: break

    lect_count = lecture_num
    endSearch = False

    for d in range (days):
        for p in range (periods_per_day):
            for r in rooms:
                neighbor_cell2+= 1
                if neighbor2[d][p][r] == course:
                    lect_count -= 1
                if lect_count == 0:
                    endSearch = True
                    break
            if endSearch: break
        if endSearch: break

    lect_count = lecture_num
    endSearch = False

    for d in range (days):
        for p in range (periods_per_day):
            for r in rooms:
                best_cell+= 1
                if best_neighbor[d][p][r] == course:
                    lect_count -= 1
                if lect_count == 0:
                    endSearch = True
                    break
            if endSearch: break
        if endSearch: break

    fitness_i = fitness_set[bee_index]
    fitness_j = fitness_set[neighbor_index]
    w = 0

    # if fitness_i > fitness_j:
    #     w = ((fitness_i * cell) - (fitness_j * neighbor_cell1)) / (fitness_i + fitness_j)
    # else:
    #     w = ((fitness_i * cell) + (fitness_j * neighbor_cell1)) / (fitness_i + fitness_j)

    if fitness_i > fitness_j:
        w = ((fitness_i * course_value) - (fitness_j * neighbor_value)) / (fitness_i + fitness_j)
    else:
        w = ((fitness_i * course_value) + (fitness_j * neighbor_value)) / (fitness_i + fitness_j)

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
    slot_index = 0
    least_difference = float("inf")
    i = 0 

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
                
    new_fitness = evaluate_fitness(new_solution)

    if new_fitness <= fitness_set[index] and solution != new_solution:
        depth_set[index] = fitness_set[index]
        fitness_set[index] = new_fitness
        solution_set[index] = new_solution
        stagnation[index] = 0
    # print(self.stagnation)

def swap(type, bee): #types: (1)Employed (2)Onlooker
    n5 = False
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
    neighbor_index = random.randint(0, len(solution_set) -1)
    neighbor = solution_set[neighbor_index]
    day, period, room = 0, 0, "-1"
    
    neighbor_cell = 0
    while (course == -1):
        day = random.randint(0, days-1)
        period = random.randint(0, periods_per_day-1)
        room = list(rooms.keys())[random.randint(0,len(rooms)-1)]
        course = solution[day][period][room]

    # print(periods_rooms)
    rnd = random.random()
    if rnd > 0.82: #N1 change day_period
        available_slots = get_available_period(course, solution, [day, period, room])
        if available_slots:
            solution[day][period][room] = -1
            slot = random.choice(available_slots)
            solution[slot[0]][slot[1]][slot[2]] = course
        
    elif rnd > 0.66: # N2
        available_rooms = []
        for r in solution[day][period]:
            if solution[day][period][r] == -1:
                available_rooms.append(r)
        if available_rooms:
            solution[day][period][room] = -1
            solution[day][period][random.choice(available_rooms)] = course
            
    elif rnd > 0.50: #N3 Cant be the same period or room
        available_slots = get_available_slots_different_period_room(course, solution, [day, period, room])
        if available_slots:
            solution[day][period][room] = -1
            slot = random.choice(available_slots)
            solution[slot[0]][slot[1]][slot[2]] = course

    elif rnd > 0.33:   #N4 
        select_course = random.choice(list(courses.keys()))
        select_room = random.choice(list(rooms.keys()))
        for day in solution:
            for period in solution[day]:
                for room in solution[day][period]:
                    if solution[day][period][room] == select_course:
                        solution[day][period][room] = solution[day][period][select_room]
                        solution[day][period][select_room] = select_course

    elif rnd >0.16: #N5 Move not Swap
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
                
    else: #N6 copy the move in 6
        available_slots = get_available_slots(course, solution, [day, period, room])
        if available_slots:
            solution[day][period][room] = -1
            slot = random.choice(available_slots)
            solution[slot[0]][slot[1]][slot[2]] = course

    new_fitness = evaluate_fitness(solution)
    personal_fitness = 0 
    if type == 1: personal_fitness = fitness_set[bee]
    else: personal_fitness = fitness_set[onlooker_bee[bee]]

    if (new_fitness <= personal_fitness):
                if (new_fitness < personal_fitness):
                    depth_set[index] = fitness_set[index]
                    stagnation[index] = 0 #Reset
                fitness_set[index] = evaluate_fitness(solution)
                solution_set[index] = solution
                #if n5: print("Done N5")
                
                

def produce_solution():
    return initialize_solution(1, False)[0]

#*********Utils**********#        

def getRoomIndex(room):
    cnt = 0
    for r in rooms:
        cnt+=1
        if room == r:
            return cnt
    return -1

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

def get_available_slots(course, timetable, constraint_period=[-1,-1,-1]):
    available_slots = []
    for day in timetable:
        for period in timetable[day]:
            hasConflict = False
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

def abc(solution_set_param, maximum_cycles_param, limit_param):

    global solution_set
    solution_set = solution_set_param

    global global_best_solution 
    best_solution = {}
    best = 1000000
    for solution in range(len(solution_set)):
        fitness = evaluate_fitness(solution_set[solution])
        # print(f"{solution}: {fitness}")
        if (fitness) <= best:
            best = fitness
            best_solution = solution_set[solution]
    global_best_solution = best_solution
    
    global maximum_cycles
    maximum_cycles = maximum_cycles_param
    global limit
    limit = limit_param

    global population
    population = len(solution_set_param)

    for _ in range(len(solution_set)):
        fitness_set.append(evaluate_fitness(solution_set[_]))
        depth_set.append(fitness_set[_])
        onlooker_bee.append("")
        employed_bee.append("")
        stagnation.append(0)

def cycle_abc():

    employed_bee_phase()
    onlooker_bee_phase()
    scout_bee_phase()
    print(str(fitness_set))
    stagnate()

    global global_best_solution
    global_best_solution = get_abc_best_solution()
    return global_best_solution

    

def kem_abc(best_solution):
    solution_set[get_abc_worst()] = best_solution