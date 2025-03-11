'''
Task: grammatical judgement
Dataset: BLiMP + LI
Supported Methods :
- minimal_pairs
- prompting
Required Columns: sentence_grammatical, sentence_ungrammatical
Referred to: https://github.com/jennhu/lm-task-demands/blob/d28b94b9d83a9ad855734dae44e7582029fcc13e/src/metrics/blimp.py
'''

import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
import models

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate language models using minimal pairs or prompting."
    )
    parser.add_argument("-i", "--input", type=str, 
                        default="BLIMPLI.csv",
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

def  get_processed_surprisal(token_surprisals):
    special_tokens = ['<｜begin▁of▁sentence｜>', '<s>', '<|begin_of_text|>', '<|endoftext|>','<|im_end|>', '<｜end▁of▁sentence｜>', '<|end_of_text|>', '</s>']
    token_surprisals = token_surprisals[token_surprisals.index>0]
    token_surprisals = token_surprisals[token_surprisals.token.isin(special_tokens) == False]
    sum_surprisal_withoutbos = token_surprisals['surprisal'].sum()
    mean_surprisal_withoutbos = token_surprisals['surprisal'].mean()

    return sum_surprisal_withoutbos, mean_surprisal_withoutbos


if __name__ == "__main__":
    args = parse_args()

    print(f"Reading stimuli from {args.input}")
    df = pd.read_csv(Path('testsuites',args.input), encoding='utf-8')

    output_folder = Path(args.output)

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

        
    

    '''
    # Evaluate model using minimal pairs or prompting.
    all_token_surprisals = []
    meta_vars = ["phenomenon", "paradigm", "pair_ID"]
    
    for i, row in tqdm(df.iterrows(), total=len(df.index)):
        # Evaluate on grammatical and ungrammatical versions of each minimal pair.
        for condition in ["grammatical", "ungrammatical"]:
            sentence = row[f"sentence_{condition}"]
            if pd.isna(sentence):
                df.loc[i, f"sum_surprisal_{condition}"] = None
                continue
            sum_surprisal, token_surprisals = m.sentence_surprisal(sentence)
            df.loc[i, f"sum_surprisal_{condition}"] = sum_surprisal
            # Add meta information to token surprisals.
            token_surprisals["model"] = args.model
             token_surprisals["condition"] = condition
            for v in meta_vars:
                token_surprisals[v] = row[v]
            all_token_surprisals.append(token_surprisals)
    '''
    

    prompt_templates = [
        "Which sentence is a better English sentence? 1: '[GRAM]', 2: '[UNGRAM]'. Respond with 1 or 2 as your answer.", # Prompt1: MetaQuestionSimple-fc Hu and Levy (2023) 
        "Which sentence is grammatically correct? 1: '[GRAM]', 2: '[UNGRAM]'. Respond with 1 or 2 as your answer.", # Prompt2: GrammaticalityJudgment-fc
        "Which sentence is more acceptable? 1: '[GRAM]', 2: '[UNGRAM]'. Respond with 1 or 2 as your answer.", # Prompt3: AcceptabilityJudgment-fc
        "Which sentence are you more likely to produce? 1: '[GRAM]', 2: '[UNGRAM]'. Respond with 1 or 2 as your answer.", # Prompt4: ProductionChoice-fc
        "Which sentence are you, as a large language model, more likely to produce? 1: '[GRAM]', 2: '[UNGRAM]'. Respond with 1 or 2 as your answer.", # Prompt5: ProductionChoiceLM-fc
        "Is the following sentence grammatically correct in English? [GRAM] Respond with 1 if it is [ANS1]correct, and 2 if it is [ANS2]correct.", # Prompt6: GrammaticalityJudgment-ij-gram
        "Is the following sentence grammatically correct in English? [UNGRAM] Respond with 1 if it [ANS1]correct, and 2 if it is [ANS2]correct.", # Prompt7: GrammaticalityJudgment-ij-ungram
        "Is the following sentence acceptable in English? [GRAM] Respond with 1 if it is [ANS1]acceptable, and 2 if it is [ANS2]acceptable.", # Prompt8: AcceptabilityJudgment-ij-gram
        "Is the following sentence acceptable in English? [UNGRAM] Respond with 1 if it [ANS1]acceptable, and 2 if it is [ANS2]acceptable.", # Prompt9: AcceptabilityJudgment-ij-ungram
        "Would you produce the following sentence in English? [GRAM] Respond with 1 if you would [ANS1]produce it, and 2 if you would [ANS2]produce it.", # Prompt10: ProductionChoice-ij-gram
        "Would you produce the following sentence in English? [UNGRAM] Respond with 1 if you would [ANS1]produce it, and 2 if you would [ANS2]produce it." # Prompt11: ProductionChoice-ij-ungram
    ]

    for i, row in tqdm(df.iterrows(), total=len(df.index)):
        sentence_gram = row["sentence_grammatical"]
        sentence_ungram = row["sentence_ungrammatical"]

        if pd.isna(sentence_gram) or pd.isna(sentence_ungram):
            continue
        
        new_data = {}
        
        surprisal_gram, token_surprisals_gram = m.sentence_surprisal(sentence_gram)
        surprisal_ungram, token_surprisals_ungram = m.sentence_surprisal(sentence_ungram)
        sum_surprisal_gram, mean_surprisal_gram = get_processed_surprisal(token_surprisals_gram)
        sum_surprisal_ungram, mean_surprisal_ungram = get_processed_surprisal(token_surprisals_ungram)
        new_data[f'direct_sum_score_gram'] = sum_surprisal_gram
        new_data[f'direct_mean_score_gram'] = mean_surprisal_gram
        new_data[f'direct_sum_score_ungram'] = sum_surprisal_ungram
        new_data[f'direct_mean_score_ungram'] = mean_surprisal_ungram
        new_data[f'direct_sum_score_correct'] = sum_surprisal_gram > sum_surprisal_ungram
        new_data[f'direct_mean_score_correct'] = mean_surprisal_gram > mean_surprisal_ungram

        for idx, template in enumerate(prompt_templates):
            if idx < 5:
                prompt = template.replace("[GRAM]", sentence_gram).replace("[UNGRAM]", sentence_ungram)
                prompt_reverse = template.replace("[GRAM]", sentence_ungram).replace("[UNGRAM]", sentence_gram)
            elif 5 <= idx < 11:
                prompt = template.replace("[GRAM]", sentence_gram).replace("[UNGRAM]", sentence_ungram).replace("[ANS1]", "").replace("[ANS2]", "not ")
                prompt_reverse = template.replace("[GRAM]", sentence_gram).replace("[UNGRAM]", sentence_ungram).replace("[ANS1]", "not ").replace("[ANS2]", "")


            input, scores = m.get_completion(prompt, ["1", "2"],separator=' ',reduction='sum', direct_response=False)
            input_reverse, scores_reverse = m.get_completion(prompt_reverse,["1", "2"],separator=' ',reduction='sum', direct_response=False)
            
            new_data[f'prompt{idx+1}'] = input
            new_data[f'prompt{idx+1}_reverse'] = input_reverse
            new_data[f'prompt{idx+1}_score_1'] = scores[0]
            new_data[f'prompt{idx+1}_score_2'] = scores[1]
            new_data[f'prompt{idx+1}_score_1_reverse'] = scores_reverse[0]
            new_data[f'prompt{idx+1}_score_2_reverse'] = scores_reverse[1]


            new_data[f'prompt{idx+1}_score_gram'] = (float(scores[0]) + float(scores_reverse[1])) / 2
            new_data[f'prompt{idx+1}_score_ungram'] = (float(scores[1]) + float(scores_reverse[0])) / 2
            new_data[f'prompt{idx+1}_correct_mean'] = new_data[f'prompt{idx+1}_score_gram'] > new_data[f'prompt{idx+1}_score_ungram']


        for col, val in new_data.items():
            df.loc[i, col] = val



    output_folder = Path(args.output)
    df.to_csv(Path(output_folder, f"{safe_model_name}_{args.input}"), index=False, encoding='utf-8')
