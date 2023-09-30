import json
import pinecone
import boto3
from boto3.dynamodb.conditions import Key
import time

client = boto3.client('kendra')
lambda_client = boto3.client('lambda')
dynamodb = boto3.resource('dynamodb')

questions_table = dynamodb.Table('6998-questions-db')
comments_table = dynamodb.Table('6998-comments-db')
search_table = dynamodb.Table('consensus-searches')

def get_comments(comment_id, all_comments):
    comments = []
    for comment in all_comments:
        if comment['parent_id'] == comment_id:
            child_comments = get_comments(comment['id'], all_comments)
            comment['comments'] = child_comments
            comments.append(comment)
    return comments

def lambda_handler(event, context):
    query = event["queryStringParameters"]["q"]
    sessionID = "1232321"

    # Perform semantic search using Pinecone
    PINECONE_API_KEY = '974f9758-d34f-4083-b82d-a05e3b1742ae'
    PINECONE_ENV = 'us-central1-gcp'

    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )

    index = pinecone.Index("semantic-search-6998")
    xc = index.query(query, top_k=10, include_metadata=True)
    question_ids = [result['metadata']['text'] for result in xc['matches']]

    # Retrieve relevant posts from DynamoDB
    questions = []
    for question_id in question_ids:
        response = questions_table.query(
            IndexName='title-index',
            KeyConditionExpression=Key('title').eq(question_id)
        )
        questions.extend(response['Items'])

    # Retrieve comments for each post from DynamoDB
    comments = []
    for question in questions:
        response = comments_table.query(
            IndexName='parent_id-index',
            KeyConditionExpression=Key('parent_id').eq(question['id'])
        )
        comments.extend(response['Items'])

    # Format the response to include posts and comments
    formatted_questions = []
    for question in questions:
        question['comments'] = get_comments(question['id'], comments)
        formatted_questions.append(question)

    response = {
        'posts': formatted_questions
    }

    # Return the response
    return {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'OPTIONS,GET',
        },
        'body': json.dumps(response)
    }