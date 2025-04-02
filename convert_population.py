from config import *

def parse_ctt_file(ctt_file_path):
    """
    Parses the .ctt file to extract configuration data, including room details.
    """
    config = {}
    rooms = {}

    with open(ctt_file_path, 'r') as f:
        lines = f.readlines()

    # Extract configuration details from the .ctt file
    for line in lines:
        if line.startswith("Days:"):
            config['num_days'] = int(line.split(":")[1].strip())
        elif line.startswith("Periods_per_day:"):
            config['periods_per_day'] = int(line.split(":")[1].strip())
        elif line.startswith("Rooms:"):
            config['num_rooms'] = int(line.split(":")[1].strip())
        elif line.startswith("ROOMS:"):
            # Extract room details (ID and capacity)
            idx = lines.index(line) + 1
            while idx < len(lines) and lines[idx].strip():
                room_info = lines[idx].strip().split()
                room_id = room_info[0]
                room_capacity = int(room_info[1])
                rooms[room_id] = room_capacity
                idx += 1

    config['rooms'] = rooms
    return config


def parse_out_file(out_file_path, config):
    """
    Parses the .out file to create the timetable based on course-room-period assignments.
    """
    timetable = {}

    # Initialize timetable structure based on the config (Days, Periods, Rooms)
    for day in range(config['num_days']):
        timetable[day] = {}
        for period in range(config['periods_per_day']):
            timetable[day][period] = {}
            for room_id in config['rooms']:  # Use room IDs from config['rooms']
                timetable[day][period][room_id] = -1  # Initially, no course is assigned

    # Read and parse the .out file for course assignments
    with open(out_file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        # Extract course, room, day, and period info from the line
        parts = line.split()
        course_id = parts[0]
        room_id = parts[1]
        day = int(parts[2])
        period = int(parts[3])

        # Update the timetable with course assignments
        if day in timetable and period in timetable[day]:
            timetable[day][period][room_id] = course_id

    return timetable


def print_timetable(timetable):
    """
    Prints the timetable in a readable format.
    """
    for day in timetable:
        print(f"Day {day}:")
        for period in timetable[day]:
            print(f"  Period {period}:")
            for room_id, course_id in timetable[day][period].items():
                print(f"    Room {room_id}: {course_id}")

# Example usage:

def get_timetable():

    # Step 1: Parse the configuration from the .ctt file
    ctt_file_path = INPUT  # Replace with the actual .ctt file path
    config = parse_ctt_file(ctt_file_path)

    # Step 2: Parse the timetable from the .out file
    out_file_path = OUTPUT  # Replace with the actual .out file path
    timetable = parse_out_file(out_file_path, config)

    return timetable


# # Step 3: Print the timetable
# print_timetable(timetable)
