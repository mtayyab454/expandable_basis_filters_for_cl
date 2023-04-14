import os
import re
def find_last_occurrence(file_contents, search_str):

    last_index = file_contents.rfind(search_str)
    if last_index != -1:
        return file_contents[last_index+len(search_str):]
    else:
        return None

def extract_task_values(log_contents):
    pattern = re.compile(r'Testing performance of task \d+:  \[\d+\.\d+, (\d+\.\d+), \d+\.\d+, \d+\.\d+\]')
    matches = pattern.findall(log_contents)
    if matches:
        value_list = [float(match) for match in matches]
        return value_list
    else:
        return None

def extract_from_single_line(log_contents, search_str):

    pattern = re.compile(f'{search_str}\[(.*?)\]')
    match = pattern.search(log_contents)
    if match:
        accuracy_str = match.group(1)
        # accuracy_list = [float(x) for x in accuracy_str.split(',')]
        accuracy_list = [round(float(x), 2) for x in accuracy_str.split(',')]
        return accuracy_list
    else:
        return None


def parse_logfile(file_name):

    with open(os.path.join('../logs', file_name+'.out'), 'r') as f:
        file_contents = f.read()

    file_contents = find_last_occurrence(file_contents, "Keys:  ['time', 'acc1', 'acc5', 'ce_loss']")
    task_accuracy = extract_task_values(file_contents)
    task_prediction_accuracy = extract_from_single_line(file_contents, 'Task Prediction Accuracy :  ')
    class_incrimental_accuracy = extract_from_single_line(file_contents, 'Class Incrimental Accuracy :  ')

    return task_accuracy, task_prediction_accuracy, class_incrimental_accuracy

# parse_logfile()

