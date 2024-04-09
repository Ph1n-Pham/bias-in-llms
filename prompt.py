from datasets import load_dataset

from transformers import pipeline, AutoTokenizer, OpenLlamaForCausalLM, AutoModelForSeq2SeqLM
from transformers import AutoModelForSeq2SeqLM
import sentencepiece as spm

from datasets import load_dataset
import random

import argparse 
import numpy as np

import json
#import polars as pl
from dotenv import load_dotenv
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
import matplotlib.pyplot as plt
import pandas as pd
import evaluate
import torch
from tqdm import tqdm
import hf_olmo

# Utils ==================================================================
def check_cuda():
    # Check CUDA
    if torch.cuda.is_available():
        print('GPU is available')
    else:
        print('GPU is not available')

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__CUDA Device Name:', torch.cuda.get_device_name(0))
        print(
            '__CUDA Device Total Memory [GB]:',
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

    device = torch.device('cuda' if use_cuda else 'cpu')
    print('Device: ', device)

def randomize_list(lst, seed=1):
    """
    Randomizes the order of elements in a list.
    
    Args:
        lst (list): The list to be randomized.
        seed (int, optional): The seed value for the random number generator. Default is 42.
        
    Returns:
        list: The randomized list.
    """
    randomized_list = lst.copy()  # Create a copy of the original list to avoid modifying it
    
    random.seed(seed)  # Set the seed for the random number generator
    
    random.shuffle(randomized_list)  # Shuffle the list in-place
    return randomized_list

# Helper functions for filename handling
def generate_unique_filename(base_name, extension = ".txt"):
    '''
    Generate unique filename, if filename existed then add a count
    
    @Params:
        base_name: input model name
        extension: format of the file, default to .txt

    @Return:
        filename: a filename (base_name + extension)
    '''
    count = 0
    while True:
        if count == 0:  #if filename not existed then dont add count
            filename = f"{base_name}{extension}"
            filename = filename.replace("/", "_")
            if not os.path.exists(filename):
                return filename
        else:
            filename = f"{base_name}_{count}{extension}"
            filename = filename.replace("/", "_")
            if not os.path.exists(filename):
                return filename
        count += 1

def create_file(filename):
    '''
    Create filename if not existed
    '''
    if not os.path.exists(filename):
        with open(filename, "w") as file:
            file.write("")  # Create an empty file

def get_random_prompts(dataset, num_examples=100):
    '''
    Get n random prompts from dataset

    @Params:
        dataset: dataset
        num_examples: n number of examples to draw from dataset. Default to 100

    @Return:
        dataset[picks]: a subset of n examples from dataset
    '''
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    return(dataset[picks])

def store_result(result, result_agg, model=None, filename=None):
    '''
    Store regard results to filename in txt format

    @Params:
        result (dict): regard differenced scores (cat1 - cat2)
        result_agg (dict): regard scores for cat1 and cat2
        model (str): name of the model, default to None
        filename (str): default to None
    '''
    # Handle filename creation or checking
    if filename is None:
        # Generate a unique filename if none provided
        filename = generate_unique_filename(f"{model}_bias", ".txt")
    else:
        # Check if the provided filename exists
        filename = str(model) + " " + filename
        
    filename = filename.replace("/", "_")
    if not os.path.exists(filename):
        # Create the file if it doesn't exist
        create_file(filename)

    print('store to this file: ', filename)
    with open(filename, "a") as file:
        file.write(f"{model} Bias Evaluation:")
        print(f"{model} Bias Evaluation:")

        file.write('\nCategory 1:\n')
        print('Category 1:\n')
        for i in ['positive', 'neutral', 'other', 'negative']:
            file.write(i + ": "+ str(result_agg['average_data_regard'][i]) + "\n")
            print(i + ": "+ str(result_agg['average_data_regard'][i]))

        file.write('\nCategory 2:\n')
        print('\nCategory 2:\n')
        for i in ['positive', 'neutral', 'other', 'negative']:
            file.write(i + ": "+ str(result_agg['average_references_regard'][i]) + "\n")
            print(i + ": "+ str(result_agg['average_references_regard'][i]))

        file.write('\nDifference:\n')
        print('\nDifference:\n')
        for i in ['positive', 'neutral', 'other', 'negative']:
            file.write(i + ": "+ str(result['regard_difference'][i]) + "\n")
            print(i + ": "+ str(result['regard_difference'][i]))
    
# MAIN ===========================================================================

def main():
    # Check CUDA ===============================================================
    check_cuda()

    # ARGS ===============================================================
    parser = argparse.ArgumentParser(description='Prompting Bias on LLMs')
    parser.add_argument('--dataset_info_path', type=str, help='Dataset File path')
    parser.add_argument('--embeding_vector_path', type=str, help='Dataset File path')
    parser.add_argument('--batch_size', type=int, help='batch_size', default=2)
    parser.add_argument("--model_path", type=str, default=None, help="Path of the Model")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path of the Model")
    
    parser.add_argument("--cat_1", type=str, default='creative_occ', help="Category 1")
    parser.add_argument("--cat_2", type=str, default='techinical_occ', help="Category 2")
    args = parser.parse_args()

    # ENV ==========================================================================
    load_dotenv()
    access_token = os.getenv('HF_KEY')

    # PIPELINE ==========================================================================
    # Model Pipeline
    text_generation = pipeline("text-generation", model=args.model_path, 
                                torch_dtype=torch.float16,
                                token = access_token)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, token = access_token)
    print("✅ Done creating pipeline")

    #padding token id
    try:
        pad_token_id = tokenizer.eos_token_id
    except:
        pad_token_id = 50256
    
    # REGARD METRIC
    regard = evaluate.load('regard', 'compare')
    print("✅ Done loading metric module")

    # DATASET
    bold = load_dataset("AlexaAI/bold", split="train")  # API call from HuggingFace

    # Get all entries
    if args.cat_1 == 'creative_occ' and args.cat_2 == 'techinical_occ':
        cat1_category_ls = ['dance_occupations', 'film_and_television_occupations', 'entertainer_occupations', 'sewing_occupations', 'artistic_occupations', 'theatre_personnel', 'writing_occupations']
        cat2_category_ls = ['engineering_branches',  'industrial_occupations', 'professional_driver_types', 'corporate_titles',
                            'metalworking_occupations', 'railway_industry_occupations', 'nursing_specialties',  'scientific_occupations', 'computer_occupations', 'healthcare_occupations', 'mental_health_occupations']
        cat1_bold = [p for p in bold if p['category'] in cat1_category_ls]
        cat2_bold = [p for p in bold if p['category'] in cat2_category_ls]
    
    # Get all prompts in randomized order
    cat1_prompts = randomize_list([prompts for p in cat1_bold for prompts in p['prompts']])
    cat2_prompts = randomize_list([prompts for p in cat2_bold for prompts in p['prompts']])
    print("✅ Done loading Data. \nStart prompting models...")
    
    # Save all the prompts====================================
    prompt1_savename = "cat1_prompts.txt"
    prompt2_savename = "cat2_prompts.txt"
    # Create the file if it doesn't exist
    if not os.path.exists(prompt1_savename):
        create_file(prompt1_savename)
    if not os.path.exists(prompt2_savename):
        create_file(prompt2_savename)
        with open(prompt1_savename, "a") as file:
            for p in cat1_prompts[:3830]:
                file.write(p)
                file.write("\n")
        with open(prompt2_savename, "a") as file:
            for p in cat2_prompts[:3830]:
                file.write(p)
                file.write("\n")

    # PROMPTING MODEL ========================================================================
    filename_cat1 = generate_unique_filename(f"{args.model_path}_cat1_cont", ".txt")
    filename_cat1 = filename_cat1.replace("/", "_")
    if not os.path.exists(filename_cat1):
        # Create the file if it doesn't exist
        create_file(filename_cat1)

    filename_cat2 = generate_unique_filename(f"{args.model_path}_cat2_cont", ".txt")
    filename_cat2 = filename_cat2.replace("/", "_")
    # Create the file if it doesn't exist
    if not os.path.exists(filename_cat2):
        create_file(filename_cat2)

    with open(filename_cat1, "a") as file:
        cat1_continuations=[]
        for prompt in tqdm(cat1_prompts[:3830]):    #ensure both lists have same length 3830
            generation = text_generation(prompt, max_length=50, truncation=True, do_sample=False, pad_token_id=pad_token_id) #
            continuation = generation[0]['generated_text'].replace(prompt,'').replace('\n', ' ')
            cat1_continuations.append(continuation)
            file.write(continuation)
            file.write("\n")
        print("Finish cat1 generations\n")

    with open(filename_cat2, "a") as file:
        cat2_continuations=[]
        for prompt in tqdm(cat2_prompts[:3830]):
            generation = text_generation(prompt, max_length=50, truncation=True, do_sample=False, pad_token_id=pad_token_id)
            continuation = generation[0]['generated_text'].replace(prompt,'').replace('\n', ' ')
            cat2_continuations.append(continuation)
            file.write(continuation)
            file.write("\n")
        print("Finish cat2 generations\n")

    

    print("✅ Finish Prompting")
    print("Spot check one example:\n", cat1_prompts[2], "\n", cat1_continuations[2])
    print("\nStart evaluating...")

    # BIAS EVALUATION ========================================================================
    result = regard.compute(data = cat1_continuations, references= cat2_continuations)
    result_agg = regard.compute(data = cat1_continuations, references= cat2_continuations, aggregation = 'average')
    
    print("✅ Finish evaluating, storing results...")

    store_result(result=result, result_agg=result_agg, model=args.model_path)


if __name__ == "__main__":
    main()
