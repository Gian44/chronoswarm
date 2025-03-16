import os
import json
import random
import copy
import math
from ctt_parser import read_ctt_file
from initialize_population2 import assign_courses as initialize_solution
from model import *
from config import *

# Load problem data
filename = INPUT  # Replace with your actual .ctt file
courses, rooms, unavailability_constraints, curricula, days, periods_per_day = read_ctt_file(filename)

periods_rooms = periods_per_day * len(rooms)
days_periods_rooms = periods_rooms * days
room_map = {room: i for i, room in enumerate(rooms)}
reverse_room_map = {i: room for room, i in room_map.items()}

# Folder for test inputs
TEST_INPUT_FOLDER = "input"

def convert_keys_to_int(d):
    """Recursively convert dictionary keys from strings to integers where applicable."""
    if isinstance(d, dict):
        return {int(k) if k.isdigit() else k: convert_keys_to_int(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_keys_to_int(i) for i in d]
    else:
        return d

def save_test_input(file_name, data):
    """Save test input as JSON."""
    os.makedirs(TEST_INPUT_FOLDER, exist_ok=True)
    with open(os.path.join(TEST_INPUT_FOLDER, file_name), "w") as f:
        json.dump(data, f, indent=4)

def load_test_input(file_name):
    """Load test input from JSON and convert day/period keys to integers."""
    with open(os.path.join(TEST_INPUT_FOLDER, file_name), "r") as f:
        data = json.load(f)
    return convert_keys_to_int(data)

def generate_test_inputs(num_tests=5):
    """Generate test input solutions and save them if not existing."""
    if not os.path.exists(TEST_INPUT_FOLDER):
        os.makedirs(TEST_INPUT_FOLDER)
    
    test_inputs = []
    
    for i in range(num_tests):
        file_path = os.path.join(TEST_INPUT_FOLDER, f"test_{i}.json")
        
        if os.path.exists(file_path):
            print(f"Test input {file_path} already exists. Skipping generation.")
            continue
        
        print(f"Generating test input {file_path}...")
        schedule = initialize_solution()  # Generate random initial solution
        save_test_input(f"test_{i}.json", schedule)
        test_inputs.append(schedule)

    return test_inputs

def calculate_distance(p1, p2):
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
                            slots1.append({"day": int(day), "period": int(period), "room_id": room_id})

        for day, periods_dict in p2.items():
            for period, room_dict in periods_dict.items():
                for room_id, assigned_course in room_dict.items():
                    if assigned_course == course_id:
                        slots2.append({"day": int(day), "period": int(period), "room_id": room_id})

        # Compute the difference in schedule assignments
        for slot1, slot2 in zip(slots1, slots2):
            #print(slot1["day"])
            distance += (
                ((slot1["day"] - slot2["day"]) / days) ** 2 +
                ((slot1["period"] - slot2["period"]) / periods_per_day) ** 2 +
                ((room_map[slot1["room_id"]] - room_map[slot2["room_id"]]) / len(rooms)) ** 2
            )
            #print(distance)

    return math.sqrt(distance)

class MockSwarm:
    """Mock swarm for testing convertQuantum with multiple solutions."""
    def __init__(self, solutions):
        self.solution_set = [copy.deepcopy(sol) for sol in solutions]  # ✅ Store multiple solutions
        self.global_best_solution = min(self.solution_set, key=self.evaluate_fitness)  # ✅ Pick the best solution
        self.fitness_set = [self.evaluate_fitness(sol) for sol in self.solution_set]  # ✅ Compute fitness for all solutions
        self.reinitialize_flag = False
        self.rcloud = 1.5  # Adjust perturbation intensity

    def evaluate_fitness(self, solution):
        """Dummy fitness function for testing."""
        return random.randint(500, 5000)

def convertQuantum(swarm):
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
                        new_course_index = round(best_course_index + (1 * random.gauss(0, 1))) % len(courses_index)
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

def main():
    """Main function to test calculate_distance and convertQuantum."""
    test_files = [f for f in os.listdir(TEST_INPUT_FOLDER) if f.endswith(".json")]

    if not test_files:
        print("No test cases found. Generating...")
        generate_test_inputs()

    test_cases = [load_test_input(f) for f in os.listdir(TEST_INPUT_FOLDER) if f.endswith(".json")]
    #print(test_cases)

    # ✅ Initialize MockSwarm with all solutions at once
    swarm = MockSwarm(test_cases)
    rconv = (days_periods_rooms / 4) ** (1 / 3)
    print("Radius: ", rconv)
    for i, solution in enumerate(swarm.solution_set):
        original_solution = copy.deepcopy(solution)
        if i < len(swarm.solution_set)-1:
            distance = calculate_distance(original_solution, swarm.solution_set[i+1])
        else:
            distance = calculate_distance(original_solution, swarm.solution_set[0])
        print(f"\nTest Case {i+1}: Distance after Quantum Reinitialization = {distance:.4f}")

    # ✅ Test convertQuantum with all solutions in the swarm
    convertQuantum(swarm)

def get_available_slots(course, timetable, constraint_period=[-1, -1, -1]):
    """Retrieve available time slots for a given course."""
    available_slots = []
    for day in timetable:
        for period in timetable[day]:
            hasConflict = (day == constraint_period[0] and period == constraint_period[1])
            for target_course in get_assigned_courses_by_period(day, period, timetable):
                if has_conflict(course, target_course, timetable):
                    hasConflict = True
                    break
            if not hasConflict:
                for room in timetable[day][period]:
                    if timetable[day][period][room] == -1:
                        available_slots.append([day, period, room])
    return available_slots

def get_assigned_courses_by_period(day, period, timetable):
    courses = []
    for room in timetable[day][period]:
        if timetable[day][period][room] != -1:
            courses.append(timetable[day][period][room])
    return courses

def has_conflict(course1, course2, timetable):
    # Check if courses have the same teacher
    if courses[course1]['teacher'] == courses[course2]['teacher']:
        return True

    # Check if courses are in the same curriculum
    for curriculum_id, course_list in curricula.items():
        if course1 in course_list and course2 in course_list:
            return True
        
    return False

if __name__ == "__main__":
    main()
