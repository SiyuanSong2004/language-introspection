## script developed from Dr. Jennifer Hu's code: https://github.com/jennhu/lm-task-demands.git
import pandas as pd
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from minicons import scorer
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig
import torch.nn.functional as F

class LM():
    def __init__(self, 
                 model_name: str, 
                 tokenizer_name=None, 
                 revision=None, 
                 accelerate=False,
                 instruct=False,
                 quantization=False,
                 **load_kwargs):
        self.model_name = model_name
        self.safe_model_name = self.get_file_safe_model_name(model_name)
        if tokenizer_name is None:
            self.tokenizer_name = model_name
        else:
            self.tokenizer_name = tokenizer_name
        self.accelerate = accelerate
        self.revision = revision
        self.instruct = instruct

        if accelerate:
            self.device = "auto"
            self.input_device = 0
            print("Using accelerate")
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("Set device to CUDA")
            else:
                self.device = torch.device("cpu")
                print("Using CPU (CUDA unavailable)")
            self.input_device = self.device
        
        if quantization:
            print("Using quantization")
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            self.bnb_config = None

        print(
            f"Initializing tokenizer ({self.tokenizer_name}) "
            f"and model ({model_name}, revision={revision})"
        )
        tokenizer, model = self.load_tokenizer_and_model(
            self.model_name, 
            tokenizer_name=self.tokenizer_name,
            bnb_config=self.bnb_config,
            revision=revision,
            accelerate=accelerate,
            quantization = quantization,
            **load_kwargs
        )
        self.tokenizer = tokenizer
        if accelerate:
            self.model = model
        else:
            self.model = model.to(self.device)

        print("Initializing incremental LM scorer")
        if quantization:
            self.ilm_model = scorer.IncrementalLMScorer(
            self.model,
            self.device,
            quantization_config=self.bnb_config,
            tokenizer=self.tokenizer
        )   
        else:
            self.ilm_model = scorer.IncrementalLMScorer(
                self.model,
                self.device,
                tokenizer=self.tokenizer
            )

    def get_file_safe_model_name(self, model:str) -> str:
        safe_model_name = model.split("/")[1] if "/" in model else model
        return safe_model_name

    def load_tokenizer_and_model(self, model_name, tokenizer_name=None, bnb_config=None, revision=None, accelerate=False, quantization=False, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **kwargs)
        if revision is not None:
            kwargs["revision"] = revision
        if accelerate:
            if quantization:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    quantization_config=bnb_config,
                    **kwargs
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    **kwargs
                )
        else:
            if quantization:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    **kwargs
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        return tokenizer, model

    def get_reduction_fn(self, reduction="mean"):
        if reduction == "mean":
            reduction_fn = lambda x: x.mean(0).item()
        elif reduction == "sum":
            reduction_fn = lambda x: x.sum(0).item()
        elif reduction == "none":
            reduction_fn = lambda x: x
        else:
            raise ValueError("`reduction` should be 'mean', 'sum', or 'none'")
        return reduction_fn


    def sentence_surprisal(self, sentence):
        token_surprisals = self.ilm_model.token_score(
            sentence
        )[0]
        tokens, surprisal_scores = zip(*token_surprisals)
        sum_surprisal = sum(surprisal_scores)
        token_surprisal_df = pd.DataFrame({
            "token": tokens,
            "surprisal": surprisal_scores,
            "token_id": list(range(len(tokens)))
        })
        return sum_surprisal, token_surprisal_df

    def get_logprob_of_continuation(self, 
                                prefixes, 
                                continuations, 
                                separator="", 
                                reduction="mean"):
        reduction_fn = self.get_reduction_fn(reduction=reduction)
        scores = self.ilm_model.conditional_score(
            prefixes, 
            continuations, 
            separator=separator,
            reduction=reduction_fn
        )
        return scores


    
    def format_prompt(self, prompt, direct_response = False):
        '''
        Format the prompt for the Instruct model
        The default is to add the system message "You are a helpful assistant." and the assistant message "My answer is" if we do not want a direct response
        '''
        if direct_response:
            if self.instruct:
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant.",
                    },
                    {"role": "user", "content": prompt}
                ]
                formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                formatted_prompt = f'{prompt}'
        else:
            if self.instruct:
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant.",
                    },
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "My answer is"}
                ]
                formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)
            else:
                formatted_prompt = f'{prompt}\nMy answer is'
        return formatted_prompt
    
    def get_completion(self, prompt, answer_choices, separator=" ", reduction='mean', eval_mode='prompting', direct_response=False):
        if eval_mode == 'prompting':
            formatted_prompt = self.format_prompt(prompt, direct_response=direct_response)
            prefixes = [formatted_prompt] * len(answer_choices)
            continuations = answer_choices

            if self.instruct and direct_response:
                separator = ''
            #use the default reduction (mean) and the empty separator ''
            scores = self.get_logprob_of_continuation(
                prefixes,
                continuations,
                separator=separator,
                reduction=reduction
            )
            
            return formatted_prompt, scores
        elif eval_mode == 'generation':
            prefixes = [prompt] * len(answer_choices)
            continuations = answer_choices
            scores = self.get_logprob_of_continuation(
                prefixes,
                continuations,
                separator=separator,
                reduction=reduction
            )
            return prompt, scores
        else:
            raise ValueError("Please use 'prompting' or 'generation' for eval_mode")

