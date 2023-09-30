import json
import openai
import os
import pinecone
import datetime
import boto3
from boto3.dynamodb.conditions import Key
import tiktoken
from tqdm.auto import tqdm
import time
import uuid
import anthropic
from langchain.embeddings.openai import OpenAIEmbeddings
import xml.etree.ElementTree as ET
from datetime import datetime
import random
import re
import pandas as pd
 

client = boto3.client('kendra')
lambda_client = boto3.client('lambda')
dynamodb = boto3.resource('dynamodb')

questions_table = dynamodb.Table('lynk-questions')
comments_table = dynamodb.Table('lynk-comments')
cache_table = dynamodb.Table('lynk-search-cache')
search_table = dynamodb.Table('consensus-searches')
topics_table = dynamodb.Table('consensus-topics')
user_search_table = dynamodb.Table('consensus-searches-daily')



def lambda_handler(event, context):
    print(event)
    #query = json.loads(event['body'])["query"]
    # sessionID = json.loads(event['body'])["sessionID"]
    query = event["query"]
    sessionID = event["sessionID"]
    userID = event["userID"]
    #query = event["queryStringParameters"]["q"]
    #sessionID = "1232321"
    #TIME
    start_time = time.time()

    rand = random.randint(0, 1)
    
    if rand % 2 == 0:
        openai.api_key = "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"
        os.environ["OPENAI_API_KEY"] = "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"
    else:
        openai.api_key = "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"
        os.environ["OPENAI_API_KEY"] = "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"
        

    PINECONE_CACHE_KEY='5d2e9d8d-af18-465e-a3de-1a25c0a8ea93'
    PINECONE_CACHE_ENV='us-west1-gcp-free'

    pinecone.init(
        api_key=PINECONE_CACHE_KEY,
        environment=PINECONE_CACHE_ENV
    )


    index_cache = pinecone.Index("lynk-search-cache")


    questions = []
    PINECONE_API_KEY = '974f9758-d34f-4083-b82d-a05e3b1742ae'
    PINECONE_ENV = 'us-central1-gcp'


    openai_embed = OpenAIEmbeddings(openai_api_key=openai.api_key)
    xq = openai_embed.embed_query(query)
    

    xc = index_cache.query(xq, top_k=10, include_metadata=True)
    written_dynamo_db = False
    try: 
        for result in xc['matches']:
            if result['score'] < .97:
                print('THIS IS RUNNING')
                str_id = f'{uuid.uuid4()}'
                record = (str_id, xq, {'text': query})
                index_cache.upsert(vectors=[record])
                written_dynamo_db = True
                break

            response = cache_table.query(
                IndexName='question-index',
                KeyConditionExpression=Key('question').eq(result['metadata']['text'])
            )

            #print(result['metadata']['text'])
            answer = response['Items'][0]['response']


            print("CACHE USED BYEEEE")
            end_time = time.time()
            time_diff = end_time - start_time
            print("GET CACHE PINECONE: {:.3f} seconds".format(time_diff))
    
            return {
                'statusCode': 200,
                'headers': {
                                   "Content-Type" : "application/json",
                "Access-Control-Allow-Headers" : "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods" : "OPTIONS,POST",
                "Access-Control-Allow-Credentials" : 'true',
                "Access-Control-Allow-Origin" : "*",
                "X-Requested-With" : "*"
                },
                'body':  answer
            }
        if len(xc['matches']) == 0:
            str_id = f'{uuid.uuid4()}'
            record = (str_id, xq, {'text': query})
            index_cache.upsert(vectors=[record])
            written_dynamo_db = True

            print("UPDATED PINECONE")

    except Exception as e:
        print('ERROR SINCE NO INDEX')
        print(e)
        str_id = f'{uuid.uuid4()}'
        record = (str_id, xq, {'text': query})
        index_cache.upsert(vectors=[record])
        written_dynamo_db = True

        
    end_time = time.time()
    time_diff = end_time - start_time
    # print("GET CACHE PINECONE: {:.3f} seconds".format(time_diff))
    
    # query dynamo db to get all comments
    start_time = time.time()

    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )

    index = pinecone.Index("lynk-questions")
    openai_embed = OpenAIEmbeddings(openai_api_key=openai.api_key)


    # query Pinecone to get all Questions
    all_comment_dfs = []
    questions = []
    question_text = {}
    xc = index.query(xq, top_k=10, include_metadata=True)
    total_tokens = 0

    for result in xc['matches']:
        if result['score'] < .9:
            break

        response = questions_table.query(
            IndexName='title-index',
            KeyConditionExpression=Key('title').eq(result['metadata']['text'])
        )

        #print(result['metadata']['text'])
        for item in response['Items']:
            #sub_id = str(item['id'] + "_" item['sub'])
            question_text[item['questionID']] = (item['title'], item['body'])
            questions.append(item['questionID'])

            row = {}
            row['id'] = uuid.uuid1().hex
            row['topics'] = item['sub']

            with topics_table.batch_writer() as batch:
                print('WRITING TO DYNAMODB')
                batch.put_item(Item=row)


    print("Questions")    
    #print(questions)
    #print(question_text)    

    #print(question_text)
    #TIME
    end_time = time.time()
    time_diff = end_time - start_time
    print("GET QUESTIONS PINECONE: {:.3f} seconds".format(time_diff))
    
    # Initialization
    id_mapping = {}
    max_tokens = 13000
    total_tokens = len(query)  # Add tokens from the user_question to total_tokens
    
    # Generate a shuffled list of IDs for all comments
    id_list = [hex(i)[2:] for i in range(1, 801)]
    random.shuffle(id_list)
    

    def transform_first_id(row):
        sub_len = -2 * len(row['sub']) - 1 
        first_id = row['firstId'][:sub_len]
        first_id = first_id.split("_")[1]
        first_id = first_id + "_" + row['sub']

        return first_id

    matches = []
    
    for question_id in questions:
        print(total_tokens)
        if total_tokens >= max_tokens:
            break

        question_id_t3 = "t3_" + question_id

        question_comments = comments_table.query(
            IndexName='parentId-index',
            KeyConditionExpression=Key('parentId').eq(question_id_t3)
        )

        question_comments_df = pd.DataFrame(question_comments['Items'])
        question_comments_df['parent_id'] = question_comments_df.apply(transform_first_id, axis=1)
        all_comment_dfs.append(question_comments_df)

        # Add tokens from question_title and question_further_description to total_tokens
        total_tokens += len(question_text.get(question_id)[0]) + len(question_text.get(question_id)[1])

        question_comments_df['parent_id'] = question_comments_df.apply(transform_first_id, axis=1)
        all_comment_dfs.append(question_comments_df)

        threads = {}
        indexed_df = question_comments_df.set_index('commentId')

        for idx, row in question_comments_df.iterrows():
            comment_id = row['commentId']
            parent_id = row['parent_id']
            body_tokens = num_tokens_from_string(row['body'], "cl100k_base")

            if total_tokens + body_tokens > max_tokens:
                # Stop adding new comments once we hit the token limit
                break

            total_tokens += body_tokens

            # Generate a new ID
            new_id = str(id_list.pop())
            id_mapping[new_id] = row['body']

            # Only consider comments which don't have a parent
            if parent_id not in indexed_df.index:
                threads[new_id] = {
                    'comment_id': str(new_id),
                    'body': row['body']
                }

        match = {
            'question_id': question_id,
            'question_title': question_text.get(question_id)[0],
            'question_further_description': question_text.get(question_id)[1],
            'comments': list(threads.values())  # Convert dictionary values to a list
        }
        matches.append(match)
        
        
    
    input_json = {
        'user_question': query,
        'matching_reddit_posts': matches
    }
    
    input_json = json.dumps(input_json).replace("{", "{{").replace("}", "}}")  # Replace single curly braces with double

    print("FORMATTED COMMENTS")
    print(input_json)
    
    print("TOTAL TOKEN LENGTH: ")
    input_json_str = f'''{input_json}'''
    print(num_tokens_from_string(input_json_str, "cl100k_base"))
    
    #TIME
    end_time = time.time()
    time_diff = end_time - start_time
    print("GET COMMENTS DYNAMODB: {:.3f} seconds".format(time_diff))    

    # GPT KEY CLAIMS, SUMMARY, SUPPORTING COMMENTS -------------------------------------------------------------------------
    start_time = time.time()
    

    prompt = f'''
    As an AI search engine, your task is to analyze Reddit discussion threads and generate answers to a user's question based on common responses from Reddit users. The user's question is: '{query}'.
    
    Your analysis should involve the following:
    
    1. Group the comments into four to six distinct clusters. Each cluster should contain multiple comments that share a unique common theme. To do this, you'll need to analyze the content of each comment, identify keywords and phrases that indicate its theme, and then categorize the comments accordingly. 
    
    2. For each cluster, identify an overarching key claim that unites the comments. Phrase it in a way that directly answers the user's question '{query}'.
    
    3. Summarize and contextualize the main points from the comments within each cluster in 1-2 sentences.
    
    4. List the comment ids belonging to each cluster. Note that for a comment to belong in a cluster it must make the overarching key claim of that cluster. Ensure every comment in a certain cluster directly makes the cluster's overarching key claim.  
    
    5. Assign a relevance score to each cluster. This should be a percentage (1-100) indicating how many comments are in the cluster. 
    
    Make sure to ignore irrelevant or troll-like comments, focusing on substantial and meaningful responses. Format your analysis into an XML block with the following structure:
    
    <llm_resp_array>
        <summary_and_analysis>
            <clusters>
                <cluster>
                    <cluster_number>1</cluster_number>
                    <claim>Claim for cluster 1</claim>
                    <summary>Summary for cluster 1</summary>
                    <comments_in_cluster>Comment IDs in cluster 1</comments_in_cluster>
                    <relevance_score>Relevance score for cluster 1</relevance_score>
                </cluster>
                <!-- More clusters follow the same structure -->
            </clusters>
        </summary_and_analysis>
    </llm_resp_array>
    
    Remember, your task is to return this XML block and nothing else. Your response will be displayed directly to the user and must be less than 1500 tokens. 
    
    Here's the user's question along with the most relevant Reddit threads:
    
    {input_json}
    '''

    
    print("PROMPT LENGTH")
    print(num_tokens_from_string(prompt, "cl100k_base"))
    
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo-16k",
      messages=[
            {"role": "system", "content": "You are an AI search engine."},
            {"role": "user", "content": prompt},
        ],
      temperature=0.3,
    )
    
    output = response['choices'][0]['message']['content']
    print(output)

    try:
        combined_questions_comments_df = pd.concat(all_comment_dfs)
        comments_list = combined_questions_comments_df['body'].tolist()
    except:
        pass

    
    # Parse XML output
    root = ET.fromstring(output)
    
    # Initialize a dictionary for JSON output
    json_output = {"llm_resp_array": {"summary_and_analysis": {"clusters": []}}}
    
    # Extract data and replace IDs with their respective comments
    for cluster in root.find('summary_and_analysis').find('clusters'):
        cluster_number = cluster.find('cluster_number').text
        claim = cluster.find('claim').text
        summary = cluster.find('summary').text
        relevance_score = cluster.find('relevance_score').text

        comment_ids = re.sub(r"[^a-f0-9,]", "", cluster.find('comments_in_cluster').text).split(',')
        comment_ids = [id.strip() for id in comment_ids if id.strip()]
        comment_texts = [id_mapping[id] for id in comment_ids if id in id_mapping]
    
        # Append the cluster to the JSON output
        json_output["llm_resp_array"]["summary_and_analysis"]["clusters"].append({
            "cluster_number": cluster_number,
            "claim": claim,
            "summary": summary,
            "comments_in_cluster": comment_texts,
            "relevance_score": relevance_score
        })
    
    
    end_time = time.time()
    time_diff = end_time - start_time

    print("LLM: {:.3f} seconds".format(time_diff))

    try: 
        finalResponse = {
            "comments": len(comments_list),
            "llm_resp_array" : json_output["llm_resp_array"]
        }
    except Exception:
        finalResponse = {
            "comments": 0,
            "llm_resp_array" : json_output["llm_resp_array"]
        }


    print("FINAL RESPONSE")
    print(finalResponse["llm_resp_array"])
    

    # write to pinecone cache in dynamoDB if not exists
    row = {}
    row['id'] = uuid.uuid1().hex
    row['question'] = query
    row['response'] = json.dumps(finalResponse)

    print(row)

    if written_dynamo_db:
        with cache_table.batch_writer() as batch:
            print('WRITING TO DYNAMODB')
            batch.put_item(Item=row)



    row['sessionID'] = sessionID
    with search_table.batch_writer() as batch:
            print('WRITING TO DYNAMODB')
            batch.put_item(Item=row)


    if userID != "null":
        row['userID'] = userID
        # Returns the current local date
        current_date = datetime.now()

        date_string = current_date.strftime("%Y-%m-%d")
        row['date'] = date_string

        with user_search_table.batch_writer() as batch:
                print('WRITING TO DYNAMODB')
                batch.put_item(Item=row)
    # return search results and headers
    return {
        'statusCode': 200,
        'headers': {
               "Content-Type" : "application/json",
                "Access-Control-Allow-Headers" : "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods" : "OPTIONS,POST",
                "Access-Control-Allow-Credentials" : 'true',
                "Access-Control-Allow-Origin" : "*",
                "X-Requested-With" : "*"
        },
        'body':  json.dumps(finalResponse)
    }


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens