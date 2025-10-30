import re
from ollama import chat
from ollama import ChatResponse
import pandas as pd

table_ordinance = pd.read_csv(r"bills.csv")

def ask_deepseek(input_content, system_prompt, deep_think = True, print_log = True):
    response: ChatResponse = chat(model='deepseek-r1', messages=[
        {'role' : 'system', 'content' : system_prompt},
        {'role': 'user','content': input_content}
    ])
    response_text = response['message']['content']
    if print_log: print(response_text)
    # Extract everything inside <think>...</think> - this is the Deep Think
    think_texts = re.findall(r'<think>(.*?)</think>', response_text, flags=re.DOTALL)
    # Join extracted sections (optional, if multiple <think> sections exist)
    think_texts = "\n\n".join(think_texts).strip()
    # Exclude the Deep Think, and return the response
    clean_response= re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()

    # Return either the context, or a tuple with the context and deep think
    return clean_response if not deep_think else (clean_response, think_texts)

def assign_type(input_table):
    system_prompt_sentiment = ''' 
    You will be given a description of a ordinance proposed for the county of San Francisco.  
    Based on the content and context of the ordinance, catagorize it as Public Safety, Education or Economic Development exclusively.
    If the ordinance fits into multiple, pick the one that is closest.
    If it does not fit into any of the catagories, return N/A for that ordinance.
    Standardize the output to match how the input is written (Proper Nouns, no additional spaces or characters )
    '''
    
    input_table['type'] = input_table['summary'].apply(lambda comment : ask_deepseek(comment, system_prompt_sentiment)[0])
    return input_table

def assign_assessment(input_table):
    system_prompt_sentiment = ''' 
    You will be given a description and the catagorization of a ordinance proposed for the county of San Francisco.  
    In the context of the type and the summary, from a range of -1.0 to 1.0, determine if the passing of the ordinance would be a positive or negative impact on the respective field.
    For example, an ordinance that would slash public funding for schools, a yes would have a negative number while a no would have a positive.
    Return only the value of the score.
    If the type of the ordinance is N/A, return N/A as well
    '''
    
    input_table['assessment'] = input_table.apply(lambda row: ask_deepseek(
        f"Summary: {row['summary']}\nType: {row['type']}", 
        system_prompt_sentiment
    )[0], axis=1)

    return input_table

assign_type(table_ordinance).to_csv("bills.csv")
assign_assessment(table_ordinance).to_csv("bills.csv")
