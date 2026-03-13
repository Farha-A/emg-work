import shutil
import csv
import sys

def remove_specific_instance(file_path, target_session_id, target_level, target_instance):
    temp_path = file_path + '.tmp'
    
    with open(file_path, 'r', newline='') as infile, open(temp_path, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Write header
        header = next(reader)
        writer.writerow(header)
        
        instance_count = 0
        in_target_state = False
        last_state = None
        
        for row in reader:
            if len(row) >= 3:
                session_id = row[1]
                level_number = row[2]
                
                current_state = (session_id, level_number)
                is_target = (session_id == target_session_id and level_number == target_level)
                
                if current_state != last_state:
                    # state change
                    if is_target:
                        instance_count += 1
                        in_target_state = True
                    else:
                        in_target_state = False
                    
                    last_state = current_state
                
                if is_target and instance_count == target_instance:
                    # Skip writing this row
                    continue
            
            # Write all other rows
            writer.writerow(row)
            
    # Replace original file with modified temp file
    shutil.move(temp_path, file_path)
    if instance_count >= target_instance:
        print(f"Successfully removed instance {target_instance} of level {target_level} for id {target_session_id}.")
    else:
        print(f"Did not find instance {target_instance} of level {target_level} for id {target_session_id}. Total instances found: {instance_count}")

if __name__ == "__main__":
    file_path = r"c:\University\Grad!!!!!!!!!\Data collection\Cleaning\Data\Stream\emg_stream_f.csv"
    remove_specific_instance(file_path, '912', '3', 3)
