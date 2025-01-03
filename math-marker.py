from openai import OpenAI
import csv
import json
import ast
import logging
from datetime import datetime

class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;21m"
    blue = "\x1b[34;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = "%(asctime)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: blue + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler with custom formatter
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)

# Create file handler
fh = logging.FileHandler('app.log')
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(fh)

# Load configuration from config.json
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    logger.error("Config file not found.")
    exit(1)

# Initialize the OpenAI client
try:
    client = OpenAI(base_url=config['base_url'], api_key=config['api_key'])
except KeyError as e:
    logger.error(f"Missing key in config.json: {e}")
    exit(1)

def print_conversation(question, response, is_sub_question=False):
    prefix = "  ├─ " if is_sub_question else "►"
    logger.info(f"\n{prefix} Question: {question}")
    logger.info(f"  └─ Response: {response}\n")

def analyze_question(question, correct_answer, generated_answer, client):
    proof_methods = [
        {
            "name": "Numerical Analysis",
            "prompt": "Using numerical calculation and examples, is the generated answer correct? Explain your reasoning and conclude with CORRECT or INCORRECT."
        },
        {
            "name": "Logical Reasoning", 
            "prompt": "Using logical reasoning and step-by-step deduction, is the generated answer correct? Explain your reasoning and conclude with CORRECT or INCORRECT."
        },
        {
            "name": "Axiomatic Proof",
            "prompt": "Using mathematical axioms and definitions, is the generated answer correct? Explain your reasoning and conclude with CORRECT or INCORRECT."
        },
        {
            "name": "Contradiction Proof",
            "prompt": "Using proof by contradiction, assuming the opposite of the generated answer, is the generated answer correct? Explain your reasoning and conclude with CORRECT or INCORRECT."
        },
        {
            "name": "Contrapositive Proof",
            "prompt": "Using contrapositive logic, is the generated answer correct? Explain your reasoning and conclude with CORRECT or INCORRECT."
        }
    ]

    evaluation_prompt = f"""
    Question: {question}
    Generated Answer: {generated_answer}
    Correct Answer: {correct_answer}
    
    Please evaluate if the generated answer is correct using the following methods:
    1. Numerical Analysis
    2. Logical Reasoning
    3. Axiomatic Proof
    4. Contradiction Proof
    5. Contrapositive Proof

    For each method, provide your reasoning and conclude with CORRECT or INCORRECT.
    """

    messages = [
        {"role": "system", "content": "You are a mathematical evaluation expert focusing on specific proof methods."},
        {"role": "user", "content": evaluation_prompt}
    ]

    try:
        response = client.chat.completions.create(
            model=config["model"],
            messages=messages
        )
        
        reply = response.choices[0].message.content.strip()
        
        # Count CORRECT/INCORRECT mentions
        correct_count = reply.upper().count("CORRECT")
        incorrect_count = reply.upper().count("INCORRECT")
        
        final_result = correct_count > incorrect_count
        confidence = max(correct_count, incorrect_count) / (correct_count + incorrect_count)
        
        return {
            "result": final_result,
            "confidence": confidence,
            "explanation": reply
        }
        
    except Exception as e:
        logger.error(f"Error in evaluation: {e}")
        return {
            "result": False,
            "confidence": 0,
            "explanation": f"Error: {str(e)}"
        }

def evaluate_answer(question, correct_answer, generated_answer, client):
    # First, try direct numerical comparison if possible
    try:
        gen_ans = ast.literal_eval(str(generated_answer))
        correct_ans = ast.literal_eval(str(correct_answer))
        if isinstance(gen_ans, (int, float)) and isinstance(correct_ans, (int, float)):
            comparison = f"{gen_ans} == {correct_ans}"
            result = abs(gen_ans - correct_ans) < 1e-10  # Using small epsilon for float comparison
            return {"comparison": comparison, "result": result, "confidence": 1.0}
    except (ValueError, SyntaxError):
        pass

    # If numerical comparison isn't possible, use consolidated analysis
    return analyze_question(question, correct_answer, generated_answer, client)

# Initialize arrays to store results
all_results = []
question_answers = []

# Create output files with empty arrays
with open(config["question_answers_output"], 'w', encoding='utf-8') as f:
    json.dump([], f)

with open(config["evaluation_output"], 'w', encoding='utf-8') as f:
    json.dump([], f)

try:
    with open('math_problems_with_solutions.csv', mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                problem_index = row['problem_index']
                question = row['Question']
                answer = row['Answer']
                generated_answer = row['generated_answer']

                logger.info(f"\n{'#'*50}\nProcessing Problem {problem_index}\n{'#'*50}")
                logger.info(f"Question: {question}")
                logger.info(f"Correct Answer: {answer}")
                logger.info(f"Generated Answer: {generated_answer}")

                evaluation = evaluate_answer(question, answer, generated_answer, client)
                
                result_data = {
                    "problem_index": str(problem_index),
                    "question": str(question),
                    "correct_answer": str(answer),
                    "generated_answer": str(generated_answer),
                    "evaluation": evaluation,
                    "timestamp": datetime.now().isoformat()
                }

                question_answers.append(result_data)
                all_results.append({
                    "problem_index": str(problem_index),
                    "evaluation": evaluation
                })
                
                # Write to files after each evaluation
                with open(config["question_answers_output"], 'w', encoding='utf-8') as f:
                    json.dump(question_answers, f, indent=4, ensure_ascii=False)
                
                with open(config["evaluation_output"], 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=4, ensure_ascii=False)
                
                logger.info(f"Completed processing problem {problem_index}")
                
            except KeyError as e:
                logger.error(f"Missing field in CSV row: {e}")
                continue
            except Exception as e:
                logger.error(f"Error processing row: {e}")
                continue

except FileNotFoundError:
    logger.error("CSV file not found.")
    exit(1)