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

def get_room_capacity_cost(course, room):
    students = courses[course]['students']
    capacity = rooms[room]
    return max(0, students - capacity)  # Each extra student = 1 penalty

def get_room_stability_cost(course, solution):
    used_rooms = set()
    for day in solution:
        for period in solution[day]:
            for room, c in solution[day][period].items():
                if c == course:
                    used_rooms.add(room)
    return max(0, len(used_rooms) - 1)  # Each extra room used = 1 penalty

def get_min_working_days_cost(course, solution):
    """Calculate the minimum working days cost for a specific course."""
    days_used = set()
    
    # Iterate through the solution to gather all the days the course is assigned to
    for day in solution:
        for period in solution[day]:
            for room, assigned_course in solution[day][period].items():
                if assigned_course == course:
                    days_used.add(day)
    
    min_days_required = courses[course]['min_days']
    actual_working_days = len(days_used)

    # Calculate the cost for missing working days
    if actual_working_days < min_days_required:
        return (min_days_required - actual_working_days) * s4
    else:
        return 0


def get_curriculum_compactness_cost(course, day, period, solution):
    curriculum_penalty = 0
    for curriculum, course_list in curricula.items():
        if course not in course_list:
            continue
        has_upper = any(solution[day].get(period+1, {}).get(r, -1) in course_list for r in rooms)
        has_lower = any(solution[day].get(period-1, {}).get(r, -1) in course_list for r in rooms)
        if not has_upper and not has_lower:
            curriculum_penalty += 1 * s3  # Penalty for isolated lecture
    return curriculum_penalty


def calculate_individual_lecture_cost(course, day, period, room, timetable):
    """
    Calculate the total cost for a specific lecture based on all soft constraints.

    Args:
        course (int): The course ID
        day (int): The day the course is scheduled
        period (int): The period the course is scheduled
        room (int): The room the course is scheduled in
        timetable (dict): The timetable dictionary

    Returns:
        int: The total penalty cost for the lecture
    """
    total_cost = 0
    room_capacity = 0
    room_stability = 0
    curriculum_compactness = 0
    min_working_days = 0

    # Room Capacity Cost (S1)
    room_capacity += get_room_capacity_cost(course, room)

    # Room Stability Cost (S2)
    room_stability += get_room_stability_cost(course, timetable)

    # Curriculum Compactness Cost (S3)
    curriculum_compactness += get_curriculum_compactness_cost(course, day, period, timetable)

    # Minimum Working Days Cost (S4)
    min_working_days += get_min_working_days_cost(course, timetable)

    total_cost = room_capacity + room_stability + curriculum_compactness + min_working_days

    return room_capacity, room_stability, curriculum_compactness, min_working_days

def eval_fitness(timetable):
    """Evaluate the total cost (fitness) for the entire timetable by summing individual lecture costs."""
    total_cost = 0
    rc, rs, cc, mwd = 0, 0, 0, 0

    # Keep track of processed courses for room stability and min working days
    processed_room_stability = set()
    processed_min_working_days = set()

    for day in timetable:
        for period in timetable[day]:
            for room in timetable[day][period]:
                if timetable[day][period][room] != -1:  # For all courses assigned
                    course = timetable[day][period][room]

                    # Room Capacity cost calculation
                    rc += get_room_capacity_cost(course, room)

                    # Room Stability cost calculation (only if not processed already)
                    if course not in processed_room_stability:
                        rs += get_room_stability_cost(course, timetable)
                        processed_room_stability.add(course)

                    # Curriculum Compactness cost calculation
                    cc += get_curriculum_compactness_cost(course, day, period, timetable)

                    # Minimum Working Days cost calculation (only if not processed already)
                    if course not in processed_min_working_days:
                        mwd += get_min_working_days_cost(course, timetable)
                        processed_min_working_days.add(course)

    # Sum up all costs
    total_cost += rc + rs + cc + mwd

    # Debug prints to show individual costs
    # print(f"Room Capacity: {rc}")
    # print(f"Room Stability: {rs}")
    # print(f"Curriculum Compactness: {cc}")
    # print(f"Min Working Days: {mwd}")
    # print(f"Total fitness cost: {total_cost}")  # Debug print

    return total_cost
