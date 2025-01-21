room = 6
day = 5
period = 6

timeslots = day * period

slots = room * timeslots

cell =  136

cell_index = cell - 1  
room_index = (cell_index % room)+1
timeslot_index = cell_index // room

day_index = timeslot_index // period
period_index = timeslot_index % period

print("Cell: " + str(cell) + 
      " is in Day " + str(day_index) + 
      " Period " + str(period_index) + 
      " Room " + str(room_index))
