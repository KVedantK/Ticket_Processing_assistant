# Support Ticket Processing Pipeline

This project implements an automated pipeline for cleaning, enriching, analyzing, and summarizing customer support tickets using an LLM (Groq LLaMA 3.1 via LangChain). I have used GROQ currently as it is free, the model parameter being an isolated variable can be changed easily according to the requirement.

It performs three main tasks:
1. Data Cleaning & Missing Field Completion
2. Semantic Recurrence Detection
3. Human-readable Ticket Summarization

---

## Features

- Loads and groups support tickets from a CSV file by customer ID  
- Uses an LLM to infer missing ticket fields (category, subcategory, priority)  
- Detects whether tickets are follow-ups of previous tickets (semantic recurrence)  
- Generates concise human-readable summaries for each ticket  
- Saves processed outputs into JSON files  

---

## Architecture Overview

I) CSV File (support_tickets.csv) 

II) Grouping by Customer ID

III) LLM-based Data Cleaning (fill missing fields)

IV) Semantic Recurrence Detection (ticket relationships)

V) Ticket Summarization

VI) JSON Outputs

---

## Installation

1. Clone the repository.
2. use `pip install requirements.txt` to install the dependencies.
3. add a `.env` file in the folder where the repo is cloned and a groq key which is free and can be taken from `[GROQ API KEYS](https://console.groq.com/keys)`
4. ru the solution.py file to run the outcome

---

## CURRENT FILES

1. The files -- `cleaned_data.json`, `reccurence_information.json`, `summary_information.json` in the dir are the results of the sample run on the support_tickets provided.
