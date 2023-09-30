# Consensus Search

Note: Consensus is the precursor for Lynk. This is a depracated version of the platform. For the most up-to-date version, visit lynksearch.com

Consensus Search is a powerful search engine that leverages the wisdom of crowds to provide human, unbiased, and comprehensive answers to open-ended questions. By consolidating results from various discussion-based platforms such as Reddit, Quora, Stack Overflow, and Twitter, Consensus Search ensures users receive genuine insights from real users, instead of biased or sponsored content.

## Features

- **Intelligent Search**: Aggregates and analyzes data from numerous discussion boards using engagement, relevance, and feedback data (upvotes, likes, views) available through APIs.
- **OpenAI GPT-3 Integration**: Performs sentiment analysis, summarization, and opinion grouping on the collected data, providing users with concise, relevant answers.
- **Opinion and Consensus Metrics**: Displays prevailing consensus using proprietary relevance percentages and groups opinions based on similar concepts.
- **Wide Range of Topics**: Covers a diverse array of topics such as medical, travel, general knowledge, tech, and reviews.

# Repository Contents

## Directories:
- **enhanced-summary**: Adds functionality for enhanced summary with larger context sizes and more verbose responses
- **gpt-search**: Implements Vector search functionality. Generates Key claims. 
- **gpt-summary**: Generates Intelligent Summary based on Discussion Data. 
- **raw-search**: Ingests data from Reddit PRAW API 

## Key Files:
- **Dockerfile**: Instructions to build a Docker container for the Consensus application.
- **app.py**: Main application file.
- **app_enhanced_summary.py**: Variant of the main app focusing on enhanced summary generation.
- **prev_version.py**: Legacy version of primary functionality.
- **raw_search_lambda.py**: Lambda function dedicated to raw search operations.
- **raw_search_lambda_pushshift.py**: Raw search lambda function, with integration for pushshift (Reddit data platform).
- **requirements.txt**: Lists the Python dependencies required.




## Acknowledgements

- [OpenAI](https://www.openai.com/) for their GPT API
- [PRAW (Python Reddit API Wrapper)](https://praw.readthedocs.io/)
- All contributors and users of Consensus Search, helping us build a better search experience for everyone
- All contributors and users of Consensus Search, helping us build a better search experience for everyone
