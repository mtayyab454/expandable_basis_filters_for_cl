import os
def extract_between(text, start, end):
    start_indices = [i for i in range(len(text)) if text.startswith(start, i)]
    end_indices = [i for i in range(len(text)) if text.startswith(end, i)]

    substrings = []
    for start_index in start_indices:
        for end_index in end_indices:
            if start_index < end_index:
                substrings.append(text[start_index + len(start):end_index])
                break

    return substrings

def read_text_files(folder_name):
    # Get a list of all files in the folder
    file_list = os.listdir(folder_name)

    # Filter the list to keep only the text files
    text_files = [file for file in file_list if file.endswith(".out")]
    text_files = [x.replace('.out', '') for x in text_files]
    text_files.sort()
    # Read the content of each text file and store it in a dictionary
    content_dict = {}
    for text_file in text_files:
        with open(os.path.join(folder_name, text_file+'.out'), 'r') as file:
            content = file.read()
            content_dict[text_file] = content

    return text_files, content_dict

def parse_namespace(folder_name='../logs', keys=['arch', 'jobid', 'l1_group', 'l1_avg', 'l1_w', 'ortho_w', 'baseline']):
    _, file_contents = read_text_files(folder_name)
    experiments = {}

    for key, value in file_contents.items():
        experiments[key] = {k[0]: k[1] for k in [item.split('=') for item in value.split(', ')] if k[0] in keys}

    for exp in experiments:
        print(experiments[exp])

    return experiments

parse_namespace()

