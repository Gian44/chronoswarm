from ctt_parser import read_ctt_file
from config import *

filename = INPUT  # Replace with your .ctt file name
courses, rooms, unavailability_constraints, curricula, days, periods_per_day= read_ctt_file(filename)

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
    for curriculum in curricula:
        for course in curricula[curriculum]:
            for day in timetable:
                for period in timetable[day]:
                    for room in timetable[day][period]:
                        if timetable[day][period][room] != -1 and timetable[day][period][room] == course:
                            hasUpper = False
                            hasLower = False
                            for courseJ in curricula[curriculum]:
                                if  period<(periods_per_day-1) and timetable[day][period+1]:
                                    if courseJ in timetable[day][period+1].values(): 
                                        hasUpper = True
                                if period>0 and timetable[day][period-1]:
                                    if courseJ in timetable[day][period-1].values(): 
                                        hasLower = True
                            if (not hasUpper) and (not hasLower): cost += s3 * 1
    return cost

def minimum_working_days_cost(timetable):
    cost = 0
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

