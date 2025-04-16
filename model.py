from ctt_parser import read_ctt_file
from config import *

filename = INPUT  # Replace with your .ctt file name
courses, rooms, unavailability_constraints, curricula, days, periods_per_day = read_ctt_file(filename)

co_curr_set = {}
course_curr_set = {}
for course in courses:
    co_curr_set[course] = set()
    course_curr_set[course] = set()
    co_curr_set[course].add(course)
    for curriculum in curricula:
        if course in curricula[curriculum]:
            course_curr_set[course].add(curriculum)
            for coursej in curricula[curriculum]:
                #print (coursej)
                if coursej not in co_curr_set[course]:
                    co_curr_set[course].add(coursej)
#print(co_curr_set)

s1 = 1 # Weight of Room Capacity constraint
s2 = 1 # Weight of Room Stability constraint
s3 = 2 # Weight of Curriculum Compactness constraint
s4 = 5 # Weight of Minimum Working Days constraint

def room_capacity_cost(timetable): #Evaluate the cost on S1 (Room Capacity)
    cost = 0
    for day in timetable:
        for period in timetable[day]:
            for room in timetable[day][period]:
                if timetable[day][period][room] != -1: #For all courses assigned
                    course = timetable[day][period][room]
                    if (courses[course]["students"] > rooms[room]):
                        cost += s1 * courses[course]["students"] - rooms[room]
    return cost

def room_stability_cost(timetable):
    cost = 0
    course_rooms = {}
    # Iterate through the timetable
    for day in timetable:
        for period in timetable[day]:
            for room in timetable[day][period]:
                course = timetable[day][period][room]
                if course != -1:  # Ignore unassigned slots
                    if course not in course_rooms:
                        course_rooms[course] = set()  # Initialize a set for unique rooms
                    if room not in course_rooms[course]:
                        course_rooms[course].add(room)
    for course in course_rooms:
        cost += s2 * (len(course_rooms[course])-1)
    return cost

def curriculum_compactness_cost(timetable):
    cost = 0
    for day in timetable:
        for period in timetable[day]:
            for room in timetable[day][period]:
                course = timetable[day][period][room]
                if course != -1:
                    for curr in course_curr_set[course]:
                        hasUpper = False
                        hasLower = False
                        if period<(periods_per_day-1) and timetable[day][period+1]:
                            for coursej in timetable[day][period+1].values(): 
                                if (coursej in curricula[curr]):
                                    hasUpper = True
                                    break
                        if not hasUpper and period>0 and timetable[day][period-1]:
                            for coursej in timetable[day][period-1].values(): 
                                if (coursej in curricula[curr]):
                                    hasLower = True
                                    break
                        if (not hasUpper) and (not hasLower): 
                            cost += s3 * 1
    return cost

def minimum_working_days_cost(timetable):
    cost = 0
    course_days = {}
    # Iterate through the timetable
    for day in timetable:
        for period in timetable[day]:
            for room in timetable[day][period]:
                course = timetable[day][period][room]
                if course != -1:  # Ignore unassigned slots
                    if course not in course_days:
                        course_days[course] = set()  # Initialize a set for unique rooms
                    if day not in course_days[course]:
                        course_days[course].add(day)
    for course in course_days:
        if courses[course]['min_days'] > (len(course_days[course])):
            cost += s4 * (courses[course]['min_days'] - (len(course_days[course])))
    return cost

def eval_fitness(timetable):
    cost = 0
    course_rooms = {}
    course_days = {}
    for day in timetable:
        for period in timetable[day]:
            for room in timetable[day][period]:
                if timetable[day][period][room] != -1: #For all courses assigned
                    course = timetable[day][period][room]

                    #ROOM CAPACITY
                    if (courses[course]["students"] > rooms[room]): 
                        cost += s1 * courses[course]["students"] - rooms[room]

                    #********************#
                    
                    # ROOM STABILITY #
                    if course not in course_rooms: 
                        course_rooms[course] = set()  # Initialize a set for unique rooms
                    if room not in course_rooms[course]:
                        course_rooms[course].add(room)

                    #********************#

                    #CURRICULUM COMPACTNESS
                    for curr in course_curr_set[course]:
                        hasUpper = False
                        hasLower = False
                        if period<(periods_per_day-1) and timetable[day][period+1]:
                            for coursej in timetable[day][period+1].values(): 
                                if (coursej in curricula[curr]):
                                    hasUpper = True
                                    break
                        if not hasUpper and period>0 and timetable[day][period-1]:
                            for coursej in timetable[day][period-1].values(): 
                                if (coursej in curricula[curr]):
                                    hasLower = True
                                    break
                        if (not hasUpper) and (not hasLower): 
                            cost += s3 * 1

                    #********************#

                    #MINIMUM WORKING DAYS
                    if course not in course_days:
                        course_days[course] = set()  # Initialize a set for unique rooms
                    if day not in course_days[course]:
                        course_days[course].add(day)

                    #********************#

    for course in course_rooms:
        cost += s2 * (len(course_rooms[course])-1)
    
    for course in course_days:
        if courses[course]['min_days'] > (len(course_days[course])):
            cost += s4 * (courses[course]['min_days'] - (len(course_days[course])))


    return cost
