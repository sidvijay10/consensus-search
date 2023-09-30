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
 

lambda_client = boto3.client('lambda')
dynamodb = boto3.resource('dynamodb')

questions_table = dynamodb.Table('lynk-questions')
comments_table = dynamodb.Table('lynk-comments')


def lambda_handler(event, context):
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


    index_cache = pinecone.Index("index-cache")


    questions = []
    PINECONE_API_KEY = '974f9758-d34f-4083-b82d-a05e3b1742ae'
    PINECONE_ENV = 'us-central1-gcp'


    openai_embed = OpenAIEmbeddings(openai_api_key=openai.api_key)
    xq = openai_embed.embed_query(query)
        
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
    questions = []
    question_text = {}
    xc = index.query(xq, top_k=25, include_metadata=True)
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
            #sub_id = str(item['id'] + "_" item['sub'])
            q = {}
            if len(item['body']) > 0:
                q['body'] = item['body']
                q['title'] = item['title']
                q['link'] = item['url']
                q['num'] = str(item['num_comments'])
                q['source'] = "reddit.com"
                questions.append(q)

    # return search results and headers
    print(questions)
    questions_json = {
        "questions": questions
    }
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
        'body':  json.dumps(questions_json)
    }


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens