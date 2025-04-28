"""Evaluation module for testing the data analysis agent.

This module provides functionality to evaluate the performance of the data analysis agent
by comparing its responses against a ground truth dataset. It uses Phoenix for experiment
tracking and provides automated testing across a wide range of query types.

The evaluation approach uses substring matching to handle cases where the agent correctly
includes the answer but may format it differently or add additional context.
"""

#==============================================================================
# IMPORTS
#==============================================================================
import asyncio
import re
import sys
from pathlib import Path

import pandas as pd
from phoenix.experiments import run_experiment
from phoenix.session.client import Client

# -------------------------------------------------------------------
# Ensure prototype4 root (parent of Evaluations) is on sys.path
# -------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent        # .../prototype4/Evaluations
PROJECT_ROOT = SCRIPT_DIR.parent                    # .../prototype4
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# -------------------------------------------------------------------

from main import main

#==============================================================================
# CONSTANTS & CONFIGURATION
#==============================================================================
# Prepare test data
agent_ground_truth = {
    # Basic Retrieval
    "What is the amount of men in Prague at the end of Q3 2024?": "676069",
    # "What is the amount of women in Prague at the end of Q3 2024?": "716056",
    # "What is the amount of women in Zlin region at the end of Q3 2024?": "294996",
    # "What was the total population of Czech Republic at the start of Q1-Q3 2024?": "10900555",
	# "Show me the female population in South Moravia at mid-2024?": "625214",
	# "What was the middle-period population count for Zlín Region?": "579562",
	#
	# # Comparisons
	# "Did Prague have more residents than Central Bohemia at the start of 2024?": "No (1384732 < 1455940)",
	# "Which region had higher female population - Liberec or Karlovy Vary at period end?": "Liberec (228648 > 150270)",
    # "List regions where mid-period male population exceeded 300,000": "Česko, Prague, Central Bohemia, Ústí, South Moravia, Moravian-Silesia",
    # "Was there any quarter where Plzeň's women outnumbered men by >15,000?": "No (max difference 11235)",
    #
    # # Aggregations
    # "What was the average male population across all regions at period end?": "≈326000",
    # "Which region had the lowest total population at the end of Q3?": "Karlovy Vary (293218)",
    # "Count how many regions had <500k total residents mid-period": "5 (Karlovy Vary, Liberec, Vysočina, Olomouc, Zlín)",
    # "What was the maximum female population recorded in any region?": "716056 (Prague)",
    #
    # # Temporal/Deltas
    # "How much did Prague's population grow from start to end of Q3?": "+7393",
    # "Calculate percentage change in Ústí's male population (start vs end)": "-0.37%",
    # "Which region had the steepest population decline?": "Moravian-Silesia (-5730)",
    # "What was South Bohemia's population change rate per month?": "-386/month",
    #
    # # Demographic Ratios
    # "What percentage of Pardubice region was female at mid-period?": "50.57%",
    # "What was the male:female ratio in Vysočina at period start?": "0.985",
    # "Which region had the most balanced gender ratio at period end?": "Vysočina (50.7% female)",
    # "How many women per 100 men in Olomouc at mid-year?": "104.4",
    #
    # # Complex Joins/Logic
    # "Compare Prague's start vs end population for men and women": "Men: +5389, Women: +2004",
    # "Rank regions by population growth rate from start to end": "1. Prague (+0.53%), 2. Central Bohemia (+0.46%)...",
    # "List regions where women population decreased >1000": "Ústí (-1466), Moravian-Silesia (-3299)",
    # "Show regions where mid-period population was < start but > end": "None (no such cases)",
    #
    # # Edge Cases
    # "What if Karlovy Vary had no data for mid-period women count?": "NULL",
    # "Show regions with exactly 500,000 residents at any point": "None",
    # "Find records where male count equals female count": "None",
    # "List regions with unreported start-period data": "None (all reported)",
    # "What was Brno's population?": "[Brno not listed as separate region]",
    # "Current male population in undefined 'North Region'": "[Region doesn't exist]",
    #
    # # Negative/Inverse Queries
    # "Which regions did NOT experience population decline?": "Prague, Central Bohemia, South Moravia",
    # "Were there any quarters where no region had growing population?": "No",
    # "List regions where women count never exceeded men count": "None",
    #
    # # Imprecise Phrasings
    # "Tell me approximate folks count in Plzeň around mid-year": "≈612025",
    # "What's the rough male population in Bohemia regions combined?": "≈2.1M",
    # "How many people in that big eastern wine region?": "South Moravia ≈1.23M",
    #
    # # Boundary Testing
    # "Show regions with population between 500k-600k at end": "Plzeň (613109), Vysočina (517664), Olomouc (631688)",
    # "Which region's population is closest to 400k?": "Liberec (449261)",
    # "First 3 regions alphabetically with >700k population": "Central Bohemia, Moravian-Silesia, South Moravia",
    #
    # # Calculation Heavy
    # "What was the national population density assuming 78,866 km² area?": "≈138/km²",
    # "Calculate gender imbalance index (|M-F|/total) for Zlín at end": "0.0186",
    # "Sum all regional end-period populations - does it match Česko?": "10897237 vs 10897237 (matches)",
    #
    # # Time Logic
    # "Project annual growth if Q1-Q3 trend continues for Prague": "≈+9857/year",
    # "When would Karlovy Vary reach 250k residents at current rate?": "Never (declining)",
    # "Compare Q3 2024 growth to same period in hypothetical 2023": "[No 2023 data]",
    #
    # # Exception Testing
    # "Query population for non-existent 'West Bohemia' region": "[Region doesn't exist]",
    # "Show data for time period not in dataset (Q4 2024)": "[No data]",
    # "What was population before recorded history?": "[No data]",
    #
    # # Multi-Metric
    # "For Vysočina, show start/mid/end counts for both genders": "Start: 257012M/260948F, Mid: 256873M/260373F, End: 257229M/260435F",
    # "Compare Prague and Brno demographics": "[Brno data not separate from South Moravia]",
}

#==============================================================================
# HELPER FUNCTIONS
#==============================================================================
def get_or_create_dataset(client, name: str, data_df: pd.DataFrame):
    """Get existing dataset or create new one if it doesn't exist.
    
    This function handles the complexity of dataset management by either retrieving
    an existing dataset or creating a new one. It also handles incremental updates
    when new test cases are added, preventing duplication while allowing the test
    suite to grow over time.
    
    Args:
        client: Phoenix client instance
        name: Name to identify the dataset
        data_df: DataFrame containing the test questions and expected answers
        
    Returns:
        The retrieved or newly created dataset
    """
    try:
        # Try to get existing dataset - this prevents recreating datasets
        # on each run, which would lose historical experiment data
        dataset = client.get_dataset(name=name)
        print("Found existing dataset")
        
        # Check for new records to append - this allows incrementally adding
        # test cases without creating duplicates
        existing_df = dataset.as_dataframe()
        current_questions = [row['input']['question'] for _, row in existing_df.iterrows()]

        # Find new records that aren't in the existing dataset
        new_records = data_df[~data_df['question'].isin(current_questions)]

        if len(new_records) > 0:
            print(f"Appending {len(new_records)} new records")
            dataset = client.append_to_dataset(
                dataset_name=name,
                dataframe=new_records,
                input_keys=["question"],
                output_keys=["expected_answer"]
            )
        else:
            print("No new records to append")
            
        return dataset
    except (ValueError, Exception):
        # If any error (including dataset not found), create a new dataset
        # This handles first-time setup or recovery from corrupted datasets
        print("Creating new dataset")
        return client.upload_dataset(
            dataset_name=name,
            dataframe=data_df,
            input_keys=["question"],
            output_keys=["expected_answer"]
        )

async def task(input_data):
    """Phoenix async task function that runs the agent on a single test question.
    
    This function is designed for concurrent execution within Phoenix experiments.
    It uses asyncio to run the potentially blocking main() function in a separate thread,
    preventing it from blocking other concurrent tasks. This is essential for achieving
    high throughput during batch evaluation.
    
    Args:
        input_data: Dictionary containing the test question
        
    Returns:
        A dictionary with the original query, results, and any error information
    """
    question = input_data["question"]
    # Run the blocking main() in a thread to allow concurrent execution
    # without blocking the event loop - crucial for performance
    result = await asyncio.to_thread(main, prompt=question)
    return {
        "query": question,
        "results": result["result"],
        "error": None  # Capture any errors during execution
    }

def result_matches_expected(output, expected):
    """Check if the agent's output contains the expected answer.
    
    This function uses regex with word boundaries to find exact matches of the expected 
    answer within the agent's output. The word boundary approach provides a balance between
    strict equality (which fails on formatting differences) and loose substring matching
    (which might give false positives).
    
    Args:
        output: Dictionary containing the agent's response
        expected: Dictionary containing the expected answer
        
    Returns:
        Boolean indicating if the expected answer was found in the output
    """
    # Extract and normalize the strings for comparison
    results_str = str(output.get("results", "")).strip()
    expected_str = str(expected["expected_answer"]).strip()
    
    # Use regex with word boundaries for more precise matching
    # This prevents partial word matches while allowing for varied formatting
    return re.search(rf"\b{re.escape(expected_str)}\b", results_str) is not None

#==============================================================================
# MAIN EXECUTION CODE
#==============================================================================
# Initialize Phoenix client
px_client = Client(warn_if_server_not_running=True)

# Convert to DataFrame
new_data_df = pd.DataFrame(agent_ground_truth.items(), columns=["question", "expected_answer"])
dataset_name = "agent_output_evaluation_langchain8"

# Get or create dataset
dataset = get_or_create_dataset(px_client, dataset_name, new_data_df)

# Show final state
final_df = dataset.as_dataframe()
print(f"\nFinal dataset has {len(final_df)} records")

# Test evaluator manually
test_output = {"query": "Q", "results": "The answer is 12345", "error": None}
test_expected = {"expected_answer": "12345"}
print(result_matches_expected(test_output, test_expected))  # Should be True

# Run the experiment
test_experiment = run_experiment(
        dataset, 
        task=task, 
        evaluators=[result_matches_expected],
        concurrency=20
    )









