import random
from ctt_parser import read_ctt_file
from config import INPUT

# Function to generate the conflict matrix
def generate_conflict_matrix(courses, curricula):
    num_courses = len(courses)
    conflict_matrix = [[0] * num_courses for _ in range(num_courses)]

    for curriculum_id, curriculum_courses in curricula.items():
        for i in range(len(curriculum_courses)):
            for j in range(i + 1, len(curriculum_courses)):
                course_i = curriculum_courses[i]
                course_j = curriculum_courses[j]
                course_i_index = list(courses.keys()).index(course_i)
                course_j_index = list(courses.keys()).index(course_j)

                conflict_matrix[course_i_index][course_j_index] = 1
                conflict_matrix[course_j_index][course_i_index] = 1

    # Create conflict sets for each course
    conflict_sets = {}
    for i, course in enumerate(courses):
        conflict_sets[course] = set()
        for j, other_course in enumerate(courses):
            if conflict_matrix[i][j] == 1:
                conflict_sets[course].add(other_course)
    
    return conflict_sets

import random

# Function to generate random course permutation based on the number of conflicts, with randomization for ties
def generate_random_permutation(courses, conflict_matrix):
    # Generate the course IDs
    course_ids = list(courses.keys())

    # Calculate the number of conflicts for each course
    course_conflicts = []
    for course_id in course_ids:
        conflict_count = len(conflict_matrix[course_id])  # Count the number of conflicts (courses in the set)
        course_conflicts.append((course_id, conflict_count))

    # Sort courses by number of conflicts (highest first)
    # If there are ties (same number of conflicts), shuffle those tied courses randomly
    course_conflicts.sort(key=lambda x: x[1], reverse=True)
    
    # Shuffle courses that have the same number of conflicts
    grouped_courses = {}
    for course_id, conflict_count in course_conflicts:
        if conflict_count not in grouped_courses:
            grouped_courses[conflict_count] = []
        grouped_courses[conflict_count].append(course_id)

    # Randomize courses within the same conflict group
    for conflict_count in grouped_courses:
        random.shuffle(grouped_courses[conflict_count])

    # Flatten the grouped courses list into a single permutation
    permutation = []
    for conflict_count in sorted(grouped_courses.keys(), reverse=True):
        permutation.extend(grouped_courses[conflict_count])

    return permutation

# Function to generate the random timetable solution
def generate_random_schedule(courses, rooms, unavailability_constraints, curricula, periods_per_day, days):
    # Initialize the schedule matrix with nested dictionaries
    schedule = {day: {period: {room: -1 for room in rooms} for period in range(periods_per_day)} for day in range(days)}
    
    # Generate the conflict matrix and random permutation of courses
    conflict_matrix = generate_conflict_matrix(courses, curricula)
    course_permutation = generate_random_permutation(courses, conflict_matrix)

    # Track assigned lectures for each course (key = course_id, value = set of (day, period, room))
    assigned_lectures = {course_id: set() for course_id in courses}

    def get_assigned_courses_by_period(day, period, timetable):
        courses = []
        for room in timetable[day][period]:
            if timetable[day][period][room] != -1:
                courses.append(timetable[day][period][room])
        return courses
    
    def is_valid_assignment(course_id, day, period, room, schedule, constraints, courses, curricula):
        # 1. Check if the room is already occupied
        if schedule[day][period][room] != -1:
            #print("occupied")
            return False
        
        # 2. Check for unavailability constraints
        if course_id in constraints and (day, period) in constraints[course_id]:
            #print("unavailable")
            return False

        # 3. Check if courses have conflicting teachers
        teacher = courses[course_id]["teacher"]
        for other_course in get_assigned_courses_by_period(day, period, schedule):
            if courses[other_course]["teacher"] == teacher:
                return False  # Teacher conflict

        # 4. Check curriculum constraints
        for curriculum in curricula:
            if course_id in curricula[curriculum]:
                for other_course in get_assigned_courses_by_period(day, period, schedule):
                    if other_course in curricula[curriculum] and other_course != course_id:
                        #print("conflict")
                        return False  # Curriculum conflict

        return True

    def assign_course(course_id, required_lectures):
        assigned_lectures_for_course = 0
        retry_limit = 100  # Retry up to 100 times to assign a valid slot
        
        while assigned_lectures_for_course < required_lectures and retry_limit > 0:
            day = random.randint(0, days - 1)
            period = random.randint(0, periods_per_day - 1)
            room = random.choice(list(rooms))  # Randomly choose room

            if is_valid_assignment(course_id, day, period, room, schedule, unavailability_constraints, courses, curricula):

                schedule[day][period][room] = course_id
                assigned_lectures[course_id].add((day, period, room))
                assigned_lectures_for_course += 1
            else:
                retry_limit -= 1

        return assigned_lectures_for_course

    # Flag to track if scheduling is successful for all courses
    all_courses_assigned = False
    
    # Retry the scheduling if not all courses have been assigned their required lectures
    while not all_courses_assigned:
        # Clear the schedule and assigned lectures for each course
        schedule = {day: {period: {room: -1 for room in rooms} for period in range(periods_per_day)} for day in range(days)}
        assigned_lectures = {course_id: set() for course_id in courses}
        
        # Assign courses based on the random permutation order
        for course_id in course_permutation:
            required_lectures = courses[course_id]['lectures']
            assign_course(course_id, required_lectures)

        # Check if all courses have been assigned the required number of lectures
        all_courses_assigned = True
        for course_id in courses:
            if len(assigned_lectures[course_id]) < courses[course_id]['lectures']:
                all_courses_assigned = False
                break

    return schedule

# Function to save the timetable to a .out file in the required format
def save_schedule_to_file(schedule, rooms, filename="timetable.out"):
    with open(filename, "w") as file:
        for day, day_schedule in schedule.items():
            for period, period_schedule in day_schedule.items():
                for room, course_id in period_schedule.items():
                    if course_id  != -1:
                        file.write(f"{course_id} {room} {day} {period}\n")

def initialize_solution():
    return generate_random_schedule(courses, rooms, unavailability_constraints, curricula, periods_per_day, days)

# Example usage:
courses, rooms, unavailability_constraints, curricula, days, periods_per_day = read_ctt_file(INPUT)

# Generate the initial random schedule using the parsed data
schedule = generate_random_schedule(courses, rooms, unavailability_constraints, curricula, periods_per_day, days)

# Save the schedule to a .out file
save_schedule_to_file(schedule, rooms)
