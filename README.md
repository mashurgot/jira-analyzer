# JIRA Ticket Analyzer

A sophisticated Python application that analyzes JIRA tickets to estimate development complexity and categorize work types using AI models (BERT and Mistral).

## Features

- **AI-Powered Analysis**:

  - Uses BERT for intelligent ticket categorization
  - Uses Mistral LLM for complexity estimation
  - Smart content analysis for accurate categorization

- **Ticket Categorization**:

  - Bugs (defects, issues, incidents)
  - Features (new functionality)
  - Maintenance (updates, compliance, technical debt)

- **Complexity Assessment**:

  - Low: Simple changes (1 file/component)
  - Medium: Moderate changes (2-3 files/components)
  - High: Complex changes (4+ files/components)

- **Comprehensive Analysis**:
  - Effort estimation based on complexity
  - Cost projections using configurable hourly rates
  - Detailed distribution analysis
  - Processing of multiple CSV files

## Requirements

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- Pandas
- Required packages listed in `requirements.txt`
- Ollama with Mistral model installed

## Installation

1. Clone this repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Install Ollama and download the Mistral model:
   ```bash
   # Install Ollama from https://ollama.ai/
   ollama pull mistral
   ```

## Usage

Basic usage:

```bash
python jira_analyzer.py
```

Analyze a sample of tickets:

```bash
python jira_analyzer.py --sample-size 20
```

## Input Requirements

Place your JIRA CSV exports in the `cleaned-data` directory. Required columns:

- `key`: Ticket identifier
- `issue type`: Type of ticket
- `status`: Current status
- `created`: Creation date
- `summary`: Ticket title

Additional fields that improve analysis accuracy:

- `description`: Detailed ticket description
- `component/s`: Affected components
- Other fields will be considered based on their analysis weight

## Configuration

The following constants can be configured in `jira_analyzer.py`:

```python
# Cost Configuration
HOURLY_RATE = 180  # Engineering blended rate in USD

# Model Configuration
BERT_MODEL = "bert-base-uncased"
MISTRAL_MODEL = "mistral"
MISTRAL_API_BASE = "http://localhost:11434/api/generate"

# Analysis Configuration
COLUMN_WEIGHTS = {
    'description': 0.9,  # Contains detailed technical requirements
    'summary': 0.8,      # Provides high-level overview
    'component/s': 0.7,  # Indicates technical domain
    'issue type': 0.6,   # Helps determine category
}
```

## Output

The analysis generates a comprehensive report (`jira_analysis_report.txt`) containing:

- Processing summary and any errors
- Ticket distribution by category
- Complexity distribution
- Effort estimates and cost projections
- Detailed breakdowns by category

Example complexity distribution:

```
By Complexity:
Low: X (XX.X%)
Medium: X (XX.X%)
High: X (XX.X%)
```

## Logging

- Detailed logs are saved in the `logs` directory
- Console output shows key progress information
- Each run creates a timestamped log file

## Error Handling

The analyzer includes robust error handling:

- Multiple file encoding support
- Graceful handling of missing columns
- Fallback categorization for edge cases
- Detailed error logging

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
