# IMPORTS
import pandas as pd
import os
import re

# GLOBAL VARIABLES
OUTPUT_DIRECTORY = 'exp2_no-lock-in_data'
original_csv_name = '../exp2_no-lock-in_data/exp2_no-lock-in_data.csv'
num_instructions = 3
num_stimuli = 4
stim_trial_type = 'video-button-response'

def save_csv(df, filename):
    target_filepath = os.path.join(OUTPUT_DIRECTORY,filename)
    df.to_csv(target_filepath, index=False)



def trial_category_counter(df, trial_categories):
    all_datas = []
    for trial_cat in trial_categories:
        trial_category_count_df = df[['response','trial_category']].value_counts().reset_index()
        
        # only count when response == 1
        try:
            count_1 = trial_category_count_df[
                (trial_category_count_df['trial_category'] == trial_cat) & (trial_category_count_df['response'] == '1')
            ][0].values[0]
        except:
            count_1 = 0
        
        # count for the count_total
        try:
            count_total = trial_category_count_df[
                (trial_category_count_df['trial_category'] == trial_cat)
            ].sum()[0]
        except:
            count_total = 0
        
        new_data = (trial_cat, count_1, count_total)
        all_datas.append(new_data)
    return all_datas



def stimulus_name_counter(df, stimulus_names):
    all_datas = []
    for stimulus_name in stimulus_names:
        stimulus_name_count_df = df[['response','stimulus_name']].value_counts().reset_index()
        
        # only count when response == 1 (morally acceptable)
        try:
            count_1 = stimulus_name_count_df[
                (stimulus_name_count_df['stimulus_name'] == stimulus_name) & (stimulus_name_count_df['response'] == '1')
            ][0].values[0]
        except:
            count_1 = 0
        
        # count all for count_total
        try:
            count_total = stimulus_name_count_df[
                (stimulus_name_count_df['stimulus_name'] == stimulus_name)
            ].sum()[0]
        except:
            count_total = 0
        
        new_data = (stimulus_name, count_1, count_total)
        all_datas.append(new_data)
    return all_datas


def main():
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    df = pd.read_csv(original_csv_name)


    #FUNCTION 1 
    # we only want to keep the relevant column data which are: 'run_id', 'ID', 'trial_type', 'stimulus_name', 'response', 'trial_category'

    df = df[['run_id','ID','trial_type','stimulus_name','response','trial_category']]


    #FUNCTION 2 (filtering for completed experiment data)
    unique_ids = df['ID'].unique()

    #check each unique id, if it doesn't satisfy the requirements indicating that the experiment was completed, then remove the rows with that ID
    for unique_id in unique_ids:
        specific_id_df = df[df['ID'] == unique_id]
        
        instructions_count_valid = False
        videobuttonresp_count_valid = False
        
        trial_category_term_counts = specific_id_df['trial_category'].value_counts()

        #check that the instructions and control questions were completed        
        if 'instructions' in trial_category_term_counts.keys():
            instructions_count = trial_category_term_counts['instructions']
            if (instructions_count >= num_instructions):
                instructions_count_valid = True
            
        
        trial_type_term_counts = specific_id_df['trial_type'].value_counts()
        #make sure all stimuli were completed
        if stim_trial_type in trial_type_term_counts.keys():
            vbuttonresp_count = trial_type_term_counts[stim_trial_type]
            if (vbuttonresp_count >= num_stimuli):
                videobuttonresp_count_valid = True
        
        # if either condition is not valid, then remove all rows with that ID
        if(not instructions_count_valid or not videobuttonresp_count_valid):
            df = df[df['ID'] != unique_id]



    #FUNCTION 3 - removing the video/ prefix and .mp4 suffix from the string, to make stimulus_name easier to work with
    df['stimulus_name'] = df['stimulus_name'].str.replace(r'^video/', '', regex=True)
    df['stimulus_name'] = df['stimulus_name'].str.replace(r'\.mp4$', '', regex=True)


    #OUTPUT CLEANED CSV 
    save_csv(df,'cleaned.csv')



    #FUNCTION 4 - getting the total sum of each trial category and each stimulus name
    trial_categories = df['trial_category'].unique()
    trial_category_counter_result = trial_category_counter(df, trial_categories)
    trial_category_count_df = pd.DataFrame(trial_category_counter_result, columns=['trial_type', 'count_1', 'count_total'])
    trial_category_count_df.dropna(inplace=True)
    save_csv(trial_category_count_df, 'trial_category_count.csv')

    stimulus_names = df['stimulus_name'].unique()
    stimulus_name_counter_result = stimulus_name_counter(df, stimulus_names)
    stimulus_name_count_df = pd.DataFrame(stimulus_name_counter_result, columns=['stimulus_name', 'count_1', 'count_total'])
    stimulus_name_count_df.dropna(inplace=True)
    save_csv(stimulus_name_count_df, 'stimulus_name_count.csv')



    #FUNCTION 5 - getting demographic data (age, gender) and the ID (only if participant said there were problems or didn't understand)
    # in trial_category "debrief", output these data:
    # - Count for age, gender
    # - run_id and problem value (string) if problem value chars length > 4 or contains 'Yes'/'yes'
    # - run_id and if understand value == "no"

    debrief_response = df[df['trial_category'] == 'debrief'][['run_id','response']]
    ages = []
    genders = []

    problem_value_not_no = []
    understand_value_no = []

    for key,value in debrief_response.iterrows():
        
        run_id = value['run_id']
        response = value['response']
        response = response.replace('"value":"}','"value":""}')
        response = eval(response)
        
        age = response[2]['value']
        gender = response[3]['value']
        problem_val = response[1]['value']
        understand_val = response[0]['value']
        
        ages.append(age)
        genders.append(gender)
        
        # only include problems that have characters length of at least 5, or if it contains 'Yes'/'yes'
        if(len(problem_val) > 4 or 'Yes' in problem_val or 'yes' in problem_val):
            new_tuple_row = (run_id, problem_val)
            problem_value_not_no.append(new_tuple_row)
        
        # only include understand value that contains 'no'
        if(understand_val == 'no'):
            new_tuple_row = (run_id, understand_val)
            understand_value_no.append(new_tuple_row)


    # zip ages and genders to the same dataframe
    age_gender_df = pd.DataFrame(list(zip(ages,genders)), columns=['age','gender'])


    age_count_df = age_gender_df['age'].value_counts(ascending=True).reset_index()
    age_count_df.columns = ['age','count']
    save_csv(age_count_df, 'age_count.csv')

    gender_count_df = age_gender_df['gender'].value_counts(ascending=True).reset_index()
    gender_count_df.columns = ['gender','count']
    save_csv(gender_count_df, 'gender_count.csv')

    # combine both age and gender for count
    age_gender_count_df = age_gender_df.value_counts(ascending=True).reset_index()
    age_gender_count_df.columns = ['age','gender','count']
    save_csv(age_gender_count_df, 'age_and_gender_count.csv')

    problem_value_not_no_df = pd.DataFrame(problem_value_not_no, columns=['run_id','problem_value'])
    save_csv(problem_value_not_no_df, 'problem_value_not_no.csv')

    understand_value_no_df = pd.DataFrame(understand_value_no, columns=['run_id','understand_value'])
    save_csv(understand_value_no_df, 'understand_value_no.csv')



    #FUNCTION 6 - simple CSV with just control questions, to filter manually (more easily)
    control_questions = df[df['trial_category'] == 'instructions'][['run_id','response']]
    save_csv(control_questions, 'control_questions.csv')


    #FUNCTION 7 - getting all the responses into an array. This will be helpful for the continuous slider responses

    resp_values_for_video_btn_resp = df.groupby('trial_type')['response'].apply(list)[stim_trial_type]
    resp_values_for_video_btn_resp_df = pd.DataFrame([stim_trial_type, str(resp_values_for_video_btn_resp)]).T
    resp_values_for_video_btn_resp_df.columns = ['trial_type','response']
    save_csv(resp_values_for_video_btn_resp_df, 'response-array.csv')

if __name__ == "__main__":
    main()

