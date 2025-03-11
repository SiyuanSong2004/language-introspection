'''
Task: Continuation
Dataset: BLiMP + LI
Supported Methods :
- minimal_pairs
- prompting
Required Columns: prefix, good_continuation, bad_continuation
Referred to: https://github.com/jennhu/lm-task-demands/blob/d28b94b9d83a9ad855734dae44e7582029fcc13e/src/metrics/lambada.py
'''

import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
import models
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate language models using minimal pairs or prompting."
    )
    parser.add_argument("-i", "--input", type=str, 
                        default="wikipedia.csv",
                        help="Path to CSV file containing stimuli")
    parser.add_argument("-o", "--output", type=Path, default="model_output",
                        help="Path to output directory where output files will be written")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--accelerate", action='store_true')
    parser.add_argument("--instruct", action='store_true')
    parser.add_argument("--quantization", action='store_true')
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()

    print(f"Reading stimuli from {args.input}")
    df = pd.read_csv(Path('testsuites',args.input), encoding='utf-8')
    df['prediction_direct'] = pd.Series(dtype='object')
    df['prediction_prompt'] = pd.Series(dtype='object')


    # Initialize model.
    print(f"Initializing model ({args.model},")
    
    m = models.LM(
        args.model, 
        revision = args.revision,
        instruct = args.instruct,
        accelerate = args.accelerate,
        quantization = args.quantization       
    )
    # Get a model name that's safe for naming files (i.e., no "/").
    safe_model_name = args.model.split("/")[-1].lower()
    if args.revision!=None:
        safe_model_name = safe_model_name + '-' + args.revision
    #all_tokens_df = pd.read_csv('rawdata/all_tokens.csv',header=None, skiprows=1)
    #all_tokens = all_tokens_df[1].tolist()
    #print(len(all_tokens))
    # Evaluate model using minimal pairs or prompting.

    for i, row in tqdm(df.iterrows(), total=len(df.index)):
        '''
        Forced Choice Style Continuation Task
        '''

        prefix = row["prefix"]
        good_ans = row["good_continuation"]
        bad_ans = row["bad_continuation"]
        continuations = [good_ans, bad_ans]
        pre, scores = m.get_completion(prefix, continuations, separator=' ', reduction='sum', eval_mode='generation')
        df.loc[i, "direct_score_1"] = scores[0]
        df.loc[i, "direct_score_2"] = scores[1]
        df.loc[i, "direct_correct"] = scores[0] > scores[1]
        


        '''
        Prompting Style Continuation Task (Choose From Two Complete Sentences)
        '''
        prompt_templates = [
            "Which sentence is a better English sentence? 1: '[PRE] [ANS1]', 2: '[PRE] [ANS2]'. Respond with 1 or 2 as your answer.", # Prompt1: MetaQuestionSimple-Sent-Choice-fc
            "Which sentence are you more likely to produce, 1 or 2? 1: '[PRE] [ANS1]', 2: '[PRE] [ANS2]'. Respond with 1 or 2 as your answer.", # Prompt2: ProductionChoice-Sent-Choice-fc
            "Which word is a better continuation after '[PRE]', 1 or 2? 1: '[ANS1]', 2: '[ANS2]'. Respond with 1 or 2 as your answer.", # Prompt3: MetaQuestionSimple-Word-Choice-fc
            "Which word are you more likely to produce after '[PRE]', 1 or 2? 1: '[ANS1]', 2: '[ANS2]'. Respond with 1 or 2 as your answer.", # Prompt4: ProductionChoice-Word-Choice-fc
            "Would you produce the following sentence in English? [PRE] [ANS1] Respond with 1 if you [PLC1] produce it, and 2 if you [PLC2].", # Prompt5: ProductionChoice-Sent-Choice-ij-1
            "Would you produce the following sentence in English? [PRE] [ANS2] Respond with 1 if you [PLC1] produce it, and 2 if you [PLC2].", # Prompt6: ProductionChoice-Sent-Choice-ij-2
            'What word is most likely to come next in the following sentence ([ANS1], or [ANS2])? [PRE]', # Prompt7: MetaQuestionSimple_Direct
            'Here is the beginning of an English sentence: [PRE]... What word is more likely to come next: [ANS1] or [ANS2]?', # Prompt8: MetaQuestionComplex_Direct
        ]

        #prediction_prompt = "What word would you produce after the text '[PRE]'? Respond with the word you would produce."



        new_data = {}

        for idx, template in enumerate(prompt_templates):
            '''
            Forced Choice Style Question: compare the answer and the delta
            '''
            if idx < 4: # forced choice prompt1-4; choice is ['1', '2']
                prompt = template.replace("[PRE]", prefix).replace("[ANS1]", good_ans).replace("[ANS2]", bad_ans)
                prompt_reverse = template.replace("[PRE]", prefix).replace("[ANS1]", bad_ans).replace("[ANS2]", good_ans)
                input, scores = m.get_completion(prompt, ["1", "2"],separator=' ',reduction='sum', direct_response=False)
                input_reverse, scores_reverse =  m.get_completion(prompt_reverse, ["1", "2"],separator=' ',reduction='sum', direct_response=False)

            elif 4 <= idx < 6: # forced choice prompt5-6; choice is ['1', '2']
                prompt = template.replace("[PRE]", prefix).replace("[ANS1]", good_ans).replace("[ANS2]", bad_ans).replace("[PLC1]", "would").replace("[PLC2]", "would not")
                prompt_reverse = template.replace("[PRE]", prefix).replace("[ANS1]", good_ans).replace("[ANS2]", bad_ans).replace("[PLC1]", "would not").replace("[PLC2]", "would")
                input, scores =  m.get_completion(prompt, ["1", "2"],separator=' ',reduction='sum', direct_response=False)
                input_reverse, scores_reverse = m.get_completion(prompt_reverse, ["1", "2"],separator=' ',reduction='sum', direct_response=False)

            elif 6 <= idx < 8: # forced choice prompt7-8; choice is ['ANS1', 'ANS2']
                prompt = template.replace("[PRE]", prefix).replace("[ANS1]", good_ans).replace("[ANS2]", bad_ans)
                prompt_reverse = template.replace("[PRE]", prefix).replace("[ANS1]", bad_ans).replace("[ANS2]", good_ans)
                input, scores = m.get_completion(prompt,continuations,separator=' ',reduction='sum',eval_mode='prompting', direct_response=True)
                input_reverse, scores_reverse = m.get_completion(prompt_reverse,continuations,separator=' ',reduction='sum',eval_mode='prompting', direct_response=True)

        
            new_data[f'prompt{idx+1}'] = input
            new_data[f'prompt{idx+1}_reverse'] = input_reverse
            new_data[f'prompt{idx+1}_score_1'] = scores[0]
            new_data[f'prompt{idx+1}_score_2'] = scores[1]
            new_data[f'prompt{idx+1}_score_1_reverse'] = scores_reverse[0]
            new_data[f'prompt{idx+1}_score_2_reverse'] = scores_reverse[1]


            new_data[f'prompt{idx+1}_score_gram'] = (float(scores[0]) + float(scores_reverse[1])) / 2
            new_data[f'prompt{idx+1}_score_ungram'] = (float(scores[1]) + float(scores_reverse[0])) / 2


            new_data[f'prompt{idx+1}_correct'] = new_data[f'prompt{idx+1}_score_gram'] > new_data[f'prompt{idx+1}_score_ungram']



        for col, val in new_data.items():
            df.loc[i, col] = val



    output_folder = Path(args.output)
    df.to_csv(Path(output_folder, f"{safe_model_name}_{args.input}"), index=False, encoding='utf-8')
    
