import os
import shutil

def read_ctt_file(filename):
    with open(filename, 'r') as file:
        data = file.readlines()

    # Extract data from the .ctt file
    courses = {}
    rooms = {}
    unavailability_constraints = {}
    curricula = {}
    reading_section = None
    days = 0
    periods_per_day = 0

    for line in data:
        line = line.strip()
        if not line or line == "END.":
            continue
        
        if line.startswith("Days"):
            parts = line.split()
            days = int(parts[1])
        
        if line.startswith("Periods_per_day:"):
            parts = line.split()
            periods_per_day = int(parts[1])

        if line.startswith("Name:") or line.startswith("Courses:") or line.startswith("Rooms:") or \
           line.startswith("Days:") or line.startswith("Periods_per_day:") or line.startswith("Constraints:"):
            continue

        if line == "COURSES:":
            reading_section = "COURSES"
            continue
        elif line == "ROOMS:":
            reading_section = "ROOMS"
            continue
        elif line == "UNAVAILABILITY_CONSTRAINTS:":
            reading_section = "CONSTRAINTS"
            continue
        elif line == "CURRICULA:":
            reading_section = "CURRICULA"
            continue

        if reading_section == "COURSES":
            parts = line.split()
            course_id = parts[0]
            courses[course_id] = {
                'teacher': parts[1],
                'lectures': int(parts[2]),
                'min_days': int(parts[3]),
                'students': int(parts[4])
            }
        elif reading_section == "ROOMS":
            parts = line.split()
            room_id = parts[0]
            rooms[room_id] = int(parts[1])
        elif reading_section == "CONSTRAINTS":
            parts = line.split()
            course_id = parts[0]
            day = int(parts[1])
            period = int(parts[2])
            if course_id not in unavailability_constraints:
                unavailability_constraints[course_id] = []
            unavailability_constraints[course_id].append((day, period))
        elif reading_section == "CURRICULA":
            parts = line.split()
            curriculum_id = parts[0]
            course_list = parts[2:]  # The list of courses in the curriculum
            curricula[curriculum_id] = course_list

    # Ensure the 'assigned_lectures' and 'assigned_days' keys are initialized for each course
    for course_id in courses:
        courses[course_id]['assigned_lectures'] = 0

    return courses, rooms, unavailability_constraints, curricula, days, periods_per_day

def write_out_file(timetable, filename):
    with open(filename, "w") as file:  # Open the file in write mode
        for day in timetable:
            for period in timetable[day]:
                for room in timetable[day][period]:
                    if timetable[day][period][room] != -1:
                            file.write(f"{timetable[day][period][room]} {room} {day} {period}\n")

def delete_all_files_in_output():
    output_folder = "output"
    # Check if the folder exists
    if os.path.exists(output_folder) and os.path.isdir(output_folder):
        # Loop through and delete all files
        for filename in os.listdir(output_folder):
            file_path = os.path.join(output_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove file or symlink
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directory
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

