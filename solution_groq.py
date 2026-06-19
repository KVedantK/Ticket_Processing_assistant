import csv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
import json
from dotenv import load_dotenv
import os

load_dotenv()
## Constants;

class Ticket(BaseModel):
    category: str = Field(default="")
    subcategory: str = Field(default="")
    priority: str = Field(default="")

class Reccurence(BaseModel):
    followup_relation: str = Field(
        description="Example: TKT-013 is followup to TKT-002. If none, return ALL TICKETS ARE DISCONNECTED."
    )

MODEL = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("GROQ_KEY")
)

PROMPT_MISSING_DATA = """
You are an expert in ticket analysis. Following are the fields and their description:
[0] ticket_id - Unique identifier for each siupport ticket
[1] date_submitted - Date of ticket raised.
[2] customer_id - Customer Identifier
[3] category - High-level topic of the issue (eg. Model Output, Billing, Integration)
[4] subcategory - More specific issue type within category (e.g. Hallucination, Latency, API Timeout)
[5] prioroty - Urgency level assigned to the ticket. Low, Medium, High, Critical based on the description.
[6] status - Current state of the ticket.
[7] agent assigned - Name of the support agent assigned to the ticket.
[8] resolution_time - Time taken to resolve the ticket in hours.
[9] satisfaction_score - Customer satisfaction rating for the ticket resolution (1-5).
[10] description - Detailed description of the issue reported in the ticket.
[11] resolution_summary - Summary of how the issue was resolved.

If the ticket has no missing values, return the ticket as is. Do not change anything.
Among these there are fields that may be inclomplete in the dataset. Based on the other fields assign values to the missing fields.
Do not fill values for deterministic fields like ticket_id, date_submitted, customer_id, description, resolution_summary. Only fill the missing values for category, subcategory, priority.

TICKET DATA: {}
"""

PROMPT_SEMANTIC_RECCURENCE = """
You are a ticket analyst. 
When provided with list of tickets your job is to identify if the tickets are reccurence of any previous tickets in that list. 
If there are any, list in the response which 'TKT' is followup of which other in the list. 
Response in format = `TKT-003 is followup to TKT002` no other format no other gibberish needed.
DO NOT PROVIDE ANY ADDITIONAL INFORMATION respond only with the followup or not. 
If there are none respond with 'ALL TICKETS ARE DISCONNECTED'.  

TICKETS = {}
"""

PROMPT_SUMMARY = """
You are an assistant to a human support manager, Provided ticket generate a human readable and easy to understand summary of the ticket mainly explaining the exact problem, priority and other elaborations.

Ticket = {}
summary : 
"""
def load_csv_to_dict(file_path):
    data = []
    data_dict = {}
    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)
        for r in range(1, len(data)):
            if data[r][2] in data_dict.keys():
                data_dict[data[r][2]].append(data[r])
            else:
                data_dict[data[r][2]] = [data[r]]
            
        return data_dict

def run_pipe_clean_data():
    data_dict = load_csv_to_dict("support_tickets.csv")
    for customer_id, tickets in data_dict.items():
        for i in range(0, len(tickets)):
            
            if tickets[i][3] == "" or tickets[i][4] == "" or tickets[i][5] == "":
                prompt = PROMPT_MISSING_DATA.format(tickets[i])
                response = MODEL.with_structured_output(Ticket).invoke(prompt)
                tickets[i][3] = response.category
                tickets[i][4] = response.subcategory
                tickets[i][5] = response.priority
        
    return data_dict

def run_pipe_check_semantic_reccurence(data_dict):
    recuurence_information = {}
    for cust_id in data_dict.keys():
        tickets = data_dict[cust_id]
        prompt = PROMPT_SEMANTIC_RECCURENCE.format(tickets)
        response = MODEL.with_structured_output(Reccurence).invoke(prompt)
        recuurence_information[cust_id] = response.followup_relation

    return recuurence_information

def run_pipe_generate_summary(data_dict):
    summary_information = {}
    for cust_id in data_dict.keys():
        tickets = data_dict[cust_id]
        for ticket in tickets:
            prompt = PROMPT_SUMMARY.format(ticket)
            response = MODEL.invoke(prompt)
            summary_information[ticket[0]] = response.content
    return summary_information

if __name__ == "__main__":
    data_dict = run_pipe_clean_data()
    print("Cleaned Data: ", data_dict)
    reccurence_information = run_pipe_check_semantic_reccurence(data_dict)
    print("Reccurence Information: ", reccurence_information)
    summary_information = run_pipe_generate_summary(data_dict)
    print("Summary Information: ", summary_information)

    with open('cleaned_data.json', 'w') as f:
        json.dump(data_dict, f, indent=4)

    with open('reccurence_information.json', 'w') as f:
        json.dump(reccurence_information, f, indent=4)

    with open('summary_information.json', 'w') as f:
        json.dump(summary_information, f, indent=4) 