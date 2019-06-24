import pandas as pd
import numpy as np

# def pre_process(submissions_path, challenges_path):
print('Loading data files...')
challenges = pd.read_csv("../machine_learning_module/data/challenges.csv")
submissions = pd.read_csv("../machine_learning_module/data/submissions.csv")

print('Modifying datasets...')
# Add a flag to determine if a hacker solved a given challenge or not
tmp = submissions.groupby(['hacker_id', 'challenge_id'], as_index=False)['solved'].agg(np.sum)
tmp['solved'] = tmp['solved'] > 0
submissions = submissions.drop('solved', axis=1).merge(tmp, on=['hacker_id', 'challenge_id'], how='left')

# Add whether or not a challenge is part of the target contest
challenges['in_target_contest'] = challenges['contest_id'] == 'c8ff662c97d345d2'
challenges.drop('contest_id', axis=1, inplace=True)

# Add submission count and solved submission count per challenge
submission_count = challenges.groupby('challenge_id', as_index=False).agg({
        'solved_submission_count': np.sum,
        'total_submissions_count': np.sum
    })
challenges = pd.merge(challenges.drop(['solved_submission_count', 'total_submissions_count'], axis=1),
                      submission_count, on='challenge_id', how='left')

# Remove duplicate entries and challenges that are not part of submissions.csv
challenges = challenges.sort_values(by=['challenge_id', 'in_target_contest'], ascending=[True, False])
challenges = challenges.drop_duplicates('challenge_id')

# Prepare data to generate the models
hackers = np.unique(submissions['hacker_id'])
recommendations = challenges[challenges['in_target_contest']]\
                    .sort_values(by=['solved_submission_count', 'total_submissions_count'], ascending=False)

print()