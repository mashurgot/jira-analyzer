"""
JIRA Ticket Analyzer
-------------------
A tool for analyzing JIRA tickets to estimate development complexity and categorize work types.

Features:
- Loads and processes JIRA ticket data from CSV files
- Uses BERT model for ticket categorization (bug/feature/maintenance)
- Uses Mistral LLM for complexity estimation (low/medium/high)
- Calculates effort estimates and cost projections
- Generates detailed analysis reports

Dependencies:
- pandas: Data processing and analysis
- torch: BERT model operations
- transformers: BERT model and tokenizer
- requests: Mistral API communication

Usage:
python jira_analyzer.py [--sample-size N]
  --sample-size: Optional. Number of tickets to analyze (default: all tickets)

Author: AI Assistant
"""

import pandas as pd
from datetime import datetime
import logging
import sys
import traceback
import json
import os
import glob
import argparse
import requests
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.nn.functional as F

# Configuration Constants
HOURLY_RATE = 180  # Engineering blended rate in USD
EFFORT_HOURS = {
    'low': 8,      # 1 day
    'medium': 24,  # 3 days
    'high': 60     # 7.5 days
}

# Model Configuration
BERT_MODEL = "bert-base-uncased"
MISTRAL_MODEL = "mistral"
MISTRAL_API_BASE = "http://localhost:11434/api/generate"
MISTRAL_TEMPERATURE = 0.1
MISTRAL_NUM_PREDICT = 32

# File and Directory Configuration
DATA_DIR = 'cleaned-data'
LOGS_DIR = 'logs'
REPORT_FILE = 'jira_analysis_report.txt'

# Logging Configuration
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
CONSOLE_FORMAT = '%(message)s'

# Analysis Configuration
COLUMN_WEIGHTS = {
    'description': 0.9,  # Contains detailed technical requirements and scope
    'summary': 0.8,      # Provides high-level overview of complexity
    'component/s': 0.7,  # Indicates technical domain and potential dependencies
    'issue type': 0.6,   # Helps determine if it's a bug, feature, or maintenance
}

# Text Processing
BUG_TYPES = {'bug', 'defect', 'problem', 'incident'}
REQUIRED_COLUMNS = {
    'key': ['key', 'issue key'],
    'issue type': ['issue type', 'issuetype', 'type'],
    'status': ['status'],
    'created': ['created', 'creation date'],
    'summary': ['summary', 'title']
}
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
FILE_ENCODINGS = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']

# Maximum lengths
MAX_DESCRIPTION_LENGTH = 500    # Maximum length for description field in LLM prompts
MAX_TICKET_DESCRIPTION = 1000   # Maximum length for description when storing ticket info

# Set up logging with both file and console handlers
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Create logs directory if it doesn't exist
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
    
    # File handler - detailed logging
    file_handler = logging.FileHandler(f'{LOGS_DIR}/jira_analyzer_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(LOG_FORMAT)
    file_handler.setFormatter(file_formatter)
    
    # Console handler - minimal logging
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(CONSOLE_FORMAT)
    console_handler.setFormatter(console_formatter)
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Add both handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# Initialize logging
setup_logging()

class JiraAnalyzer:
    def __init__(self):
        self.data = None
        self.error_files = []
        self.processed_files = []
        self.column_weights = {}  # Initialize empty column weights
        
        logging.info("Initializing JiraAnalyzer...")
        
        # Initialize BERT for categorization
        try:
            logging.info("Initializing BERT model...")
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logging.info(f"Using device: {self.device}")
            
            logging.info(f"Loading BERT model: {BERT_MODEL}")
            
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
                logging.info("Successfully loaded BERT tokenizer")
            except Exception as e:
                logging.error(f"Error loading BERT tokenizer: {str(e)}")
                raise
            
            try:
                self.bert_model = AutoModelForMaskedLM.from_pretrained(BERT_MODEL)
                self.bert_model.to(self.device)
                self.bert_model.eval()
                logging.info("Successfully loaded and configured BERT model")
            except Exception as e:
                logging.error(f"Error loading BERT model: {str(e)}")
                raise
            
            # Get the mask token
            self.mask_token = self.tokenizer.mask_token
            self.mask_token_id = self.tokenizer.mask_token_id
            logging.info(f"BERT mask token: {self.mask_token}")
            
            logging.info(f"Successfully initialized BERT model on {self.device}")
        except Exception as e:
            logging.error(f"Error initializing BERT model: {str(e)}")
            raise
            
        # Initialize Mistral settings
        logging.info("Initializing Mistral settings...")
        self.mistral_model = MISTRAL_MODEL
        self.api_base = MISTRAL_API_BASE
        
        # Define effort mapping for complexity levels
        self.effort_mapping = EFFORT_HOURS
        
        logging.info("Successfully initialized JiraAnalyzer with BERT and Mistral models")

    def predict_masked_token(self, text, candidates):
        """Use BERT to predict the most likely token from candidates for a masked position"""
        try:
            # Replace [MASK] with the actual mask token
            text = text.replace('[MASK]', self.mask_token)
            
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Find position of mask token
            mask_positions = (inputs['input_ids'] == self.mask_token_id).nonzero(as_tuple=True)[1]
            if len(mask_positions) == 0:
                raise ValueError("No mask token found in input text")
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                predictions = outputs.logits[0, mask_positions[0]]
            
            # Get token IDs for candidates
            candidate_token_ids = []
            for candidate in candidates:
                tokens = self.tokenizer.encode(candidate, add_special_tokens=False)
                if tokens:
                    candidate_token_ids.append(tokens[0])
            
            # Get probabilities for candidate tokens
            candidate_probs = F.softmax(predictions[candidate_token_ids], dim=0)
            
            # Get the most likely candidate
            max_prob_idx = torch.argmax(candidate_probs).item()
            predicted_token = candidates[max_prob_idx]
            confidence = candidate_probs[max_prob_idx].item()
            
            return predicted_token, confidence
            
        except Exception as e:
            logging.error(f"Error in masked token prediction: {str(e)}")
            raise

    def get_llm_response(self, prompt):
        """Get response from Ollama API"""
        try:
            response = requests.post(
                self.api_base,
                json={
                    "model": self.mistral_model,
                    "prompt": prompt,
                    "temperature": MISTRAL_TEMPERATURE,
                    "num_predict": MISTRAL_NUM_PREDICT
                },
                timeout=30
            )
            response.raise_for_status()
            
            # Extract the response text
            response_text = ""
            for line in response.text.strip().split('\n'):
                if line:
                    data = json.loads(line)
                    response_text += data.get('response', '')
            
            return response_text.strip().lower()
            
        except Exception as e:
            logging.error(f"Error getting LLM response: {str(e)}")
            raise

    def estimate_complexity_with_llm(self, ticket_info: dict) -> str:
        """Estimate complexity using Mistral"""
        try:
            # Create a prompt using weighted fields
            prompt_parts = []
            
            # Add fields based on their weights
            for field, value in ticket_info.items():
                field_lower = field.lower()
                # Only include fields that we've defined as important in COLUMN_WEIGHTS
                if field_lower in COLUMN_WEIGHTS:
                    # Truncate description to prevent large prompts
                    if field_lower == 'description':
                        value = str(value)[:MAX_DESCRIPTION_LENGTH]
                    prompt_parts.append(f"{field.title()}: {str(value)}")
            
            # Create classification prompt with emphasis on starting with low complexity
            prompt = f"""
You are a JIRA ticket complexity classifier. 
Your goal: read the ticket details and decide if complexity is "low", "medium", or "high". 

**Important**: Always start by assuming **LOW** complexity. Only pick **medium** or **high** if there is clear, explicit evidence that meets or exceeds the guidelines for those categories.

You MUST respond with **only one word**: "low", "medium", or "high".

Analyze the following ticket details and classify the development complexity based on these guidelines:

LOW Complexity (Default)
- Simple changes (1 file/component)
- No data model changes
- No external dependencies
- Minimal testing
Examples: text edits, simple UI tweaks, minor bug fixes

MEDIUM Complexity
- Changes to 2-3 files/components
- Possibly minor data model updates
- 1-2 external dependencies
Examples: new API endpoint, new UI component, moderate bug fixes

HIGH Complexity
- Changes to 4+ files/components
- Complex/unclear requirements
- 3+ external dependencies or major data model changes
Examples: new feature, architecture changes, complex integrations

Ticket Details:
{chr(10).join(prompt_parts)}

Repeat the final decision on complexity below (exactly one word: "low", "medium", or "high"), 
remembering to default to **LOW** if in doubt:
"""


            # Log prompt without problematic Unicode characters
            logging.debug(f"Mistral Prompt for ticket {ticket_info['key']}")
            
            # Get model's classification
            response = self.get_llm_response(prompt).strip().lower()
            
            # Map verbose responses to simple ones
            response_mapping = {
                'medium complexity': 'medium',
                'low complexity': 'low',
                'high complexity': 'high',
                'the complexity is medium': 'medium',
                'the complexity is low': 'low',
                'the complexity is high': 'high'
            }
            
            # Clean up response and extract just the complexity level
            cleaned_response = response_mapping.get(response, response)
            
            # Extract single word if response is verbose
            if cleaned_response not in ['low', 'medium', 'high']:
                words = cleaned_response.split()
                for word in words:
                    if word in ['low', 'medium', 'high']:
                        cleaned_response = word
                        break
            
            logging.info(f"Ticket {ticket_info['key']}: Complexity = {cleaned_response}")
            
            # Validate response
            if cleaned_response not in ['low', 'medium', 'high']:
                logging.warning(f"Invalid complexity from Mistral: '{response}' -> '{cleaned_response}', analyzing ticket content for default")
                # Instead of defaulting to medium, try to make an educated guess based on ticket content
                if any(high_signal in str(ticket_info.get('description', '')).lower() + str(ticket_info.get('summary', '')).lower()
                      for high_signal in ['complex', 'major', 'redesign', 'architecture', 'migration']):
                    return 'high'
                elif any(low_signal in str(ticket_info.get('description', '')).lower() + str(ticket_info.get('summary', '')).lower()
                        for low_signal in ['simple', 'minor', 'text', 'typo', 'label']):
                    return 'low'
                return 'medium'  # Only default to medium if no clear signals
            
            return cleaned_response
                
        except Exception as e:
            logging.error(f"Error in complexity estimation: {str(e)}")
            # Instead of defaulting to medium, try to make an educated guess based on ticket content
            try:
                if any(high_signal in str(ticket_info.get('description', '')).lower() + str(ticket_info.get('summary', '')).lower()
                      for high_signal in ['complex', 'major', 'redesign', 'architecture', 'migration']):
                    return 'high'
                elif any(low_signal in str(ticket_info.get('description', '')).lower() + str(ticket_info.get('summary', '')).lower()
                        for low_signal in ['simple', 'minor', 'text', 'typo', 'label']):
                    return 'low'
            except:
                pass
            return 'medium'  # Only default to medium if analysis fails

    def categorize_with_llm(self, ticket_info):
        """Categorize ticket using BERT"""
        try:
            # First check if it's explicitly a bug based on Issue Type
            issue_type = str(ticket_info.get('issue_type', '')).lower().strip()
            
            # If it's explicitly a bug by issue type, return immediately
            if any(bug_type in issue_type for bug_type in BUG_TYPES):
                logging.debug(f"Category determined as 'bug' from Issue Type: '{issue_type}'")
                return 'bug'
            
            # For non-bug issue types, let BERT analyze the content
            logging.debug(f"Analyzing content to determine category...")
            
            # Create a prompt that helps BERT understand the context better
            prompt = f"""Based on this JIRA ticket's content, it should be classified as a [MASK] ticket.

Summary: {ticket_info.get('summary', '')}
Issue Type: {issue_type}

Context:
- A 'bug' involves fixing errors, defects, or unexpected behavior in existing functionality
- A 'feature' involves new functionality or significant improvements
- 'maintenance' involves updates like OS/version upgrades, library updates, compliance changes, legal requirements, text updates, technical debt, or minor improvements"""
            
            logging.debug(f"BERT Prompt:\n{prompt}")
            
            # Define possible categories (including bug since BERT can still identify content-based bugs)
            categories = ['bug', 'feature', 'maintenance']
            
            # Get prediction
            predicted_category, confidence = self.predict_masked_token(prompt, categories)
            logging.info(f"Ticket {ticket_info['key']}: Category = {predicted_category} (confidence: {confidence:.3f})")
            
            if predicted_category not in categories:
                logging.warning(f"Invalid category from BERT: '{predicted_category}', defaulting to maintenance")
                return 'maintenance'
            
            return predicted_category
                
        except Exception as e:
            logging.error(f"Error in BERT categorization for ticket {ticket_info['key']}: {str(e)}")
            return 'maintenance'  # Default to maintenance on error

    def load_data(self, directory=DATA_DIR):
        """Load cleaned CSV files from the cleaned-data directory"""
        self.error_files = []
        self.processed_files = []
        all_files = []
        
        logging.info(f"\nLooking for CSV files in {directory}...")
        
        if not os.path.exists(directory):
            logging.error(f"Directory '{directory}' not found. Please run the data cleansing script first.")
            raise Exception(f"Directory '{directory}' not found. Please run the data cleansing script first.")
        
        # Get list of files
        csv_files = glob.glob(os.path.join(directory, "*.csv"))
        logging.info(f"Found {len(csv_files)} CSV files")
        
        if not csv_files:
            logging.error(f"No CSV files found in the {directory} directory. Please run the data cleansing script first.")
            raise Exception(f"No CSV files found in the {directory} directory. Please run the data cleansing script first.")
        
        # Process each file
        for file_path in csv_files:
            logging.info(f"\nProcessing file: {os.path.basename(file_path)}")
            try:
                # Try reading with different encodings
                for encoding in FILE_ENCODINGS:
                    try:
                        logging.info(f"Trying encoding: {encoding}")
                        df = pd.read_csv(file_path, encoding=encoding)
                        logging.info(f"Successfully read file with {encoding} encoding")
                        logging.info(f"Found {len(df.columns)} columns: {', '.join(df.columns[:5])}...")
                        
                        # Convert column names to lowercase for case-insensitive matching
                        df.columns = df.columns.str.lower()
                        
                        # Check for required columns using mappings
                        missing_columns = []
                        column_renames = {}
                        
                        for required_col, alternatives in REQUIRED_COLUMNS.items():
                            found = False
                            for alt in alternatives:
                                if alt in df.columns:
                                    if alt != required_col:
                                        column_renames[alt] = required_col
                                    found = True
                                    break
                            if not found:
                                missing_columns.append(required_col)
                        
                        if missing_columns:
                            logging.warning(f"Missing required columns: {missing_columns}")
                            self.error_files.append((file_path, f"Missing required columns: {missing_columns}"))
                            continue
                        
                        # Rename columns to standard names if needed
                        if column_renames:
                            df = df.rename(columns=column_renames)
                            logging.info(f"Renamed columns: {column_renames}")
                        
                        # Convert date columns with specific format
                        date_columns = ['created', 'updated', 'resolved', 'last viewed']
                        for col in date_columns:
                            if col in df.columns:
                                try:
                                    df[col] = pd.to_datetime(df[col], format=DATE_FORMAT, errors='coerce')
                                except ValueError:
                                    # If specific format fails, try flexible parsing
                                    df[col] = pd.to_datetime(df[col], errors='coerce')
                        
                        logging.info(f"Successfully processed file with {len(df)} rows")
                        all_files.append(df)
                        self.processed_files.append(file_path)
                        break
                        
                    except UnicodeDecodeError:
                        if encoding == 'cp1252':  # Last encoding attempt
                            logging.error(f"Failed to decode file with any supported encoding")
                            self.error_files.append((file_path, "Failed to decode file with any supported encoding"))
                    except Exception as e:
                        logging.error(f"Error reading file with {encoding}: {str(e)}")
                        self.error_files.append((file_path, str(e)))
                        break  # Break on non-encoding errors
                        
            except Exception as e:
                logging.error(f"Failed to process file: {str(e)}")
                self.error_files.append((file_path, str(e)))
        
        if not all_files:
            error_details = "\n".join([f"- {path}: {error}" for path, error in self.error_files])
            logging.error(f"No valid files could be processed from cleaned data. Errors:\n{error_details}")
            raise Exception(f"No valid files could be processed from cleaned data. Errors:\n{error_details}")
            
        # Combine all dataframes
        logging.info("\nCombining all processed files...")
        self.data = pd.concat(all_files, ignore_index=True)
        logging.info(f"Combined dataset has {len(self.data)} rows and {len(self.data.columns)} columns")
        
        # Log processing summary
        logging.info(f"\nProcessing summary:")
        logging.info(f"- Successfully processed {len(self.processed_files)} files")
        logging.info(f"- Failed to process {len(self.error_files)} files")
        
        if self.error_files:
            logging.warning(f"Failed to process {len(self.error_files)} files")
            
        # Load column weights from analysis results
        logging.info("\nLoading column weights...")
        self.load_column_weights()
        
        return self.data

    def load_column_weights(self):
        """Initialize column weights from constants"""
        try:
            self.column_weights = COLUMN_WEIGHTS.copy()
            logging.info(f"Using {len(self.column_weights)} weighted columns for analysis")
            
            # Log the weighted columns being used
            print("\nWeighted columns for analysis:")
            for col, weight in sorted(self.column_weights.items(), key=lambda x: x[1], reverse=True):
                print(f"  {col}: {weight:.1f}")
        except Exception as e:
            logging.error(f"Error initializing column weights: {str(e)}")
            self.column_weights = {}

    def analyze_tickets(self):
        """Analyze all tickets and generate statistics"""
        try:
            if self.data is None or len(self.data) == 0:
                raise ValueError("No data loaded. Please load data before analysis.")
            
            # Process each ticket for categorization and complexity
            total_rows = len(self.data)
            logging.info(f"\nProcessing {total_rows} tickets...")
            
            self.data['category'] = None
            self.data['complexity'] = None
            categorization_failures = 0
            complexity_failures = 0
            
            # Initialize counters for distribution tracking
            category_counts = {'bug': 0, 'feature': 0, 'maintenance': 0}
            complexity_counts = {'low': 0, 'medium': 0, 'high': 0}
            
            for idx in range(total_rows):
                row = self.data.iloc[idx]
                try:
                    # Create ticket info with all available fields
                    ticket_info = {
                        'key': row['key'],
                        'summary': str(row['summary']),
                        'issue_type': str(row['issue type'])
                    }
                    
                    # Add description if available
                    if 'description' in row:
                        ticket_info['description'] = str(row['description'])[:MAX_TICKET_DESCRIPTION]
                    
                    # Add all available fields
                    for col in row.index:
                        if col not in ['key', 'summary', 'description'] and pd.notna(row[col]):
                            ticket_info[col.lower()] = str(row[col])
                    
                    # Get category using BERT logic
                    category = self.categorize_with_llm(ticket_info)
                    if category:
                        self.data.at[idx, 'category'] = category
                        category_counts[category] = category_counts.get(category, 0) + 1
                    else:
                        categorization_failures += 1
                    
                    # Get complexity using Mistral
                    complexity = self.estimate_complexity_with_llm(ticket_info)
                    if complexity:
                        self.data.at[idx, 'complexity'] = complexity
                        complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
                    else:
                        complexity_failures += 1
                    
                    # Log progress every 5 tickets
                    if (idx + 1) % 5 == 0 or idx == total_rows - 1:
                        logging.info(f"Processed {idx + 1}/{total_rows} tickets")
                        logging.info(f"Categories: {category_counts}")
                        logging.info(f"Complexity: {complexity_counts}")
                    
                except Exception as e:
                    logging.error(f"Error processing ticket: {str(e)}")
                    if 'category' not in self.data.iloc[idx] or pd.isna(self.data.iloc[idx]['category']):
                        categorization_failures += 1
                    if 'complexity' not in self.data.iloc[idx] or pd.isna(self.data.iloc[idx]['complexity']):
                        complexity_failures += 1
            
            # Convert complexity to numeric effort (hours)
            self.data['estimated_hours'] = self.data['complexity'].map(self.effort_mapping)
            
            # Calculate statistics
            total_tickets = len(self.data)
            category_counts = self.data['category'].value_counts()
            complexity_counts = self.data['complexity'].value_counts()
            effort_by_category = self.data.groupby('category')['estimated_hours'].sum()
            total_effort = float(self.data['estimated_hours'].sum())
            
            stats = {
                'total_tickets': total_tickets,
                'by_category': category_counts.to_dict(),
                'by_complexity': complexity_counts.to_dict(),
                'total_effort_hours': total_effort,
                'effort_by_category': effort_by_category.to_dict(),
                'processed_files': len(self.processed_files),
                'error_files': len(self.error_files),
                'failures': {
                    'categorization': categorization_failures,
                    'complexity': complexity_failures
                }
            }
            
            return stats
            
        except Exception as e:
            logging.error(f"Error analyzing tickets: {str(e)}")
            raise

    def test_llm_responses(self):
        """Test LLM responses using sample data from our dataset"""
        print("\nTesting LLM responses with sample tickets from dataset...")
        print("=" * 50)
        
        if self.data is None or len(self.data) == 0:
            print("No data loaded. Please load data before running tests.")
            return
        
        # Get 4 random tickets from different issue types if possible
        sample_tickets = []
        for issue_type in ['Bug', 'Feature', 'Task', 'Story']:
            ticket = self.data[self.data['issue type'].str.lower() == issue_type.lower()].sample(n=1) if len(self.data[self.data['issue type'].str.lower() == issue_type.lower()]) > 0 else None
            if ticket is not None and not ticket.empty:
                sample_tickets.append(ticket.iloc[0])
        
        # If we don't have enough tickets, just get random ones
        while len(sample_tickets) < 4:
            ticket = self.data.sample(n=1).iloc[0]
            if ticket not in sample_tickets:
                sample_tickets.append(ticket)
        
        for ticket in sample_tickets:
            try:
                print(f"\nTicket: {ticket['key']}")
                print(f"Summary: {ticket['summary']}")
                print(f"Issue Type: {ticket['issue type']}")
                
                # Create ticket info dictionary
                ticket_info = {
                    'key': ticket['key'],
                    'summary': str(ticket['summary']),
                    'issue_type': str(ticket['issue type'])
                }
                
                # Add any high-weight columns
                for col, weight in self.column_weights.items():
                    if weight > 0.4 and col in ticket.index:
                        ticket_info[col] = str(ticket[col])
                
                # Get complexity
                complexity = self.estimate_complexity_with_llm(ticket_info)
                print(f"Complexity assigned by Mistral: {complexity}")
                
                # Get category
                category = self.categorize_with_llm(ticket_info)
                print(f"Category assigned by Mistral: {category}")
                
            except Exception as e:
                print(f"Error processing test ticket: {str(e)}")
            
            print("-" * 50)

    def generate_report(self, stats):
        """Generate a detailed report with error information"""
        try:
            report = []
            report.append("JIRA Ticket Analysis Report")
            report.append("=" * 30)
            report.append(f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Add methodology note
            report.append("\nMethodology Note:")
            report.append("-" * 20)
            report.append("Complexity estimates are based on ticket content analysis using Mistral.")
            report.append("Estimates consider technical requirements, dependencies, and potential risks.")
            report.append("Time estimates are guidelines and may need adjustment based on team velocity.")
            report.append(f"Labor costs are calculated using a blended engineering rate of ${HOURLY_RATE}/hour.")
            
            # Processing Summary
            report.append("\nProcessing Summary:")
            report.append("-" * 20)
            report.append(f"Successfully processed files: {stats['processed_files']}")
            report.append(f"Files with errors: {stats['error_files']}")
            
            if stats['failures']['categorization'] > 0 or stats['failures']['complexity'] > 0:
                report.append("\nProcessing Failures:")
                if stats['failures']['categorization'] > 0:
                    report.append(f"- Failed categorizations: {stats['failures']['categorization']}")
                if stats['failures']['complexity'] > 0:
                    report.append(f"- Failed complexity estimations: {stats['failures']['complexity']}")
            
            if self.error_files:
                report.append("\nFiles with Errors:")
                for file_path, error in self.error_files:
                    report.append(f"- {file_path}: {error}")
            
            # Ticket Distribution
            total_tickets = stats['total_tickets']
            report.append("\nTicket Distribution:")
            report.append("-" * 20)
            report.append(f"Total Tickets Analyzed: {total_tickets}")
            
            # Category Distribution
            report.append("\nBy Category:")
            total_categorized = sum(stats['by_category'].values())
            for category, count in sorted(stats['by_category'].items()):
                percentage = (count / total_tickets) * 100
                report.append(f"{category.capitalize()}: {count} ({percentage:.1f}%)")
            
            # Complexity Distribution
            report.append("\nBy Complexity:")
            report.append("Note: Complexity levels indicate relative development effort and risk")
            total_complexity = sum(stats['by_complexity'].values())
            complexity_order = ['low', 'medium', 'high']  # Ensure consistent ordering
            for complexity in complexity_order:
                count = stats['by_complexity'].get(complexity, 0)
                percentage = (count / total_tickets) * 100
                report.append(f"{complexity.capitalize()}: {count} ({percentage:.1f}%)")
            
            # Effort and Cost Analysis
            total_effort = stats['total_effort_hours']
            total_cost = total_effort * HOURLY_RATE
            report.append("\nEffort and Cost Analysis:")
            report.append("-" * 20)
            report.append("Note: These are baseline estimates and should be calibrated to team velocity")
            report.append(f"Total Estimated Hours: {total_effort:.1f}")
            report.append(f"Total Estimated Cost: ${total_cost:,.2f} (at ${HOURLY_RATE}/hour)")
            
            report.append("\nEffort and Cost by Category:")
            for category, hours in sorted(stats['effort_by_category'].items()):
                percentage = (hours / total_effort) * 100
                cost = hours * HOURLY_RATE
                ticket_count = stats['by_category'].get(category, 0)
                avg_hours_per_ticket = hours / ticket_count if ticket_count > 0 else 0
                report.append(f"{category.capitalize()}:")
                report.append(f"  - Tickets: {ticket_count}")
                report.append(f"  - Hours: {hours:.1f} ({percentage:.1f}%)")
                report.append(f"  - Average Hours/Ticket: {avg_hours_per_ticket:.1f}")
                report.append(f"  - Cost: ${cost:,.2f}")
            
            return "\n".join(report)
        except Exception as e:
            logging.error(f"Error generating report: {str(e)}\n{traceback.format_exc()}")
            return f"Error generating report: {str(e)}"

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze JIRA tickets')
    parser.add_argument('--sample-size', type=int, help='Number of tickets to analyze (default: all tickets)')
    args = parser.parse_args()

    try:
        analyzer = JiraAnalyzer()
        
        # Run test cases first
        logging.info("Running Mistral response tests...")
        analyzer.test_llm_responses()
        
        logging.info("\nProceeding with main analysis...")
        logging.info("Loading JIRA ticket data...")
        analyzer.load_data()
        
        # Take a sample if specified, otherwise process all tickets
        if args.sample_size:
            logging.info(f"\nAnalyzing sample of {args.sample_size} tickets...")
            analyzer.data = analyzer.data.sample(n=args.sample_size, random_state=42).copy()
        else:
            logging.info(f"\nAnalyzing all {len(analyzer.data)} tickets...")
        
        logging.info("Analyzing tickets...")
        stats = analyzer.analyze_tickets()
        
        logging.info("\nGenerating report...")
        report = analyzer.generate_report(stats)
        
        # Save report to file
        try:
            with open('jira_analysis_report.txt', 'w', encoding='utf-8') as f:
                f.write(report)
            logging.info("\nAnalysis complete! Report saved to 'jira_analysis_report.txt'")
        except Exception as e:
            logging.error(f"Error saving report to file: {str(e)}")
            logging.info("\nError saving report to file. Printing report to console instead:")
        
        logging.info("\nSummary of findings:")
        logging.info(report)
        
    except Exception as e:
        logging.error(f"Critical error: {str(e)}\n{traceback.format_exc()}")
        logging.error(f"\nCritical error occurred. Please check jira_analysis.log for details.")
        logging.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 