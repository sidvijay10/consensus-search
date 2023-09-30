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


client = boto3.client('kendra')
lambda_client = boto3.client('lambda')
dynamodb = boto3.resource('dynamodb')

questions_table = dynamodb.Table('6998-questions-db')
comments_table = dynamodb.Table('6998-comments-db')
cache_table = dynamodb.Table('consensus-cache')
search_table = dynamodb.Table('consensus-searches')

openai.api_key = "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"
os.environ["OPENAI_API_KEY"] = "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"

def do_nothing():
    pass


def lambda_handler(event, context):
    print(event)
    #query = json.loads(event['body'])["query"]
    # sessionID = json.loads(event['body'])["sessionID"]
    #query = event["query"]
    #sessionID = event["sessionID"]
    query = event["queryStringParameters"]["q"]
    sessionID = "1232321"
    #TIME
    start_time = time.time()

    PINECONE_CACHE_KEY='5d2e9d8d-af18-465e-a3de-1a25c0a8ea93'
    PINECONE_CACHE_ENV='us-west1-gcp-free'

    pinecone.init(
        api_key=PINECONE_CACHE_KEY,
        environment=PINECONE_CACHE_ENV
    )


    index_cache = pinecone.Index("index-cache")


    questions = [] 
    PINECONE_API_KEY = '974f9758-d34f-4083-b82d-a05e3b1742ae'
    PINECONE_ENV = 'us-central1-gcp'


    openai_embed = OpenAIEmbeddings(openai_api_key=openai.api_key)
    xq = openai_embed.embed_query(query)
    

    xc = index_cache.query(xq, top_k=10, include_metadata=True)
    written_dynamo_db = False
    try: 
        for result in xc['matches']:
            if result['score'] < .96:
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

            row = {}
            row['id'] = uuid.uuid1().hex
            row['question'] = query
            row['response'] = answer
            row['sessionID'] = sessionID
            with search_table.batch_writer() as batch:
                        print('WRITING TO DYNAMODB')
                        batch.put_item(Item=row)

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

    index = pinecone.Index("semantic-search-6998")
    openai_embed = OpenAIEmbeddings(openai_api_key=openai.api_key)


    # query Pinecone to get all Questions
    #print("TEST")
    questions = []
    question_text = {}
    xc = index.query(xq, top_k=10, include_metadata=True)
    total_tokens = 0

    for result in xc['matches']:
        if result['score'] < .8:
            break

        response = questions_table.query(
            IndexName='title-index',
            KeyConditionExpression=Key('title').eq(result['metadata']['text'])
        )

        #print(result['metadata']['text'])
        for item in response['Items']:
            print(item)
            question_text[item['id']] = (item['title'], item['selftext'])
            questions.append(item['id'])

    print(question_text)
    print(sd)
    
    #TIME
    end_time = time.time()
    time_diff = end_time - start_time
    print("GET QUESTIONS PINECONE: {:.3f} seconds".format(time_diff))
    
    # query dynamo db to get all comments
    start_time = time.time()

    
    from collections import defaultdict
    
    def recursive_comments(query_result):
        if 'Items' not in query_result or len(query_result['Items']) == 0:
            return []
    
        comments = []
        for item in query_result['Items']:
            comment = {
                'link_id': item['link_id'],
                'parent_id': item['parent_id'],
                'subreddit': item['subreddit'],
                'body': item['body'],
            }
            child_comments = comments_table.query(
                IndexName='parent_id-index',
                KeyConditionExpression=Key('parent_id').eq(item['link_id'])
            )
            comment['replies'] = recursive_comments(child_comments)
            comments.append(comment)
        return comments
    
    total_tokens = 0
    totalTokensOver = False
    question_text = {} # assuming this is a mapping from question_id to question text
    vector_similarity_scores = {} # assuming this is a mapping from question_id to similarity score with user's query
    user_query = "What benefits does cardio bring to muscle building?" # the user's query
    
    matches = []
    for question_id in questions:
        if totalTokensOver:
            break
    
        question_comments = comments_table.query(
            IndexName='parent_id-index',
            KeyConditionExpression=Key('parent_id').eq(question_id)
        )
    
        comments = recursive_comments(question_comments)
    
        match = {
            'question_id': question_id,
            'question': question_text[question_id],
            'similarity_score': vector_similarity_scores[question_id],
            'comments': comments
        }
        matches.append(match)
    
    final_json = {
        'user_query': user_query,
        'matches': matches
    }









comments = []
    questions_to_comments = {}  # Our target dictionary
    totalTokensOver = False
    
    for result in questions:
        response = comments_table.query(
            IndexName='parent_id-index',
            KeyConditionExpression=Key('parent_id').eq(result)
        )
        comments = []  # Reset the comments list for each question
        for item in response['Items']:
            num_tokens = num_tokens_from_string(item['body'],"cl100k_base")
            total_tokens +=  num_tokens
    
            if total_tokens > 3600:
                totalTokensOver = True
                break
    
            comments.append(item['body'])
            
        questions_to_comments[result] = comments  # Map question to its comments
        if totalTokensOver:
            break
        
        
        
        
        
    
    
    
    

    #TIME
    end_time = time.time()
    time_diff = end_time - start_time
    print("GET COMMENTS DYNAMODB: {:.3f} seconds".format(time_diff))    

    # CLAUDE KEY CLAIMS, SUMMARY, SUPPORTING COMMENTS -------------------------------------------------------------------------
    start_time = time.time()

    
    
    
    end_time = time.time()
    time_diff = end_time - start_time

    print("CLAUDE EXTRACT SUMMARY: {:.3f} seconds".format(time_diff))

    finalResponse = {
        "comments": comments[:],
        "llm_resp_array" : claims_dict
    }
    print(finalResponse)
    

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