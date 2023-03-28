import csv
import json
#keeps count of all the 1s (morally acceptable for each category 
#no_1, no_34, no_8, yes_1, yes_34, yes_8
categoryCounter = {
	'no_1': 0,
	'no_34': 0,
	'no_8': 0,
	'yes_1': 0,
	'yes_34': 0,
	'yes_8': 0
}
#keeps count of all the 1s (morally acceptable) for each map
#video/no_line_1_1cut.mp4, video/no_line_2_1cut.mp4, etc.
stimuliCounter = {}

#demographic information
demographicCounter = {
	'age': {},
	'gender': {}
}

with open('exp1_data.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:
    	#looking at the stimuli data for experiment 1. if it's not video-button-response, change string to what it is.
    	if row['trial_type'] == 'video-button-response':
    		#if they said it was morally acceptable
    		if row['response'] == 1:
    			categoryCounter[row['trial_category']] += 1
    			stimuliCounter[row['stimulus_name']] = my_dict.get(row['stimulus_name'], 0) + 1
        
        #get demographic information
    	elif row['trial_category'] == 'debrief':
    		continue
    		JSON_data = json.loads(row['response'])
    		#JSON_data looks like this for example:
    		#[{"name":"understand","value":"yes"},{"name":"problems","value":"No"},{"name":"age","value":"23"},{"name":"gender","value":"male"},{"name":"comments","value":"I felt since there were no rules dictating everyone obey a line, someone disrupting the line was morally justified. "}]


        line_count += 1