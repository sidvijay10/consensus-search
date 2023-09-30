#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 17:07:16 2023

@author: sidvijay
"""

import praw
import json

reddit = praw.Reddit(
    client_id="AAY8EB_VYRfMhiAqyHIMtw",
    client_secret="jUpxFW-ZYgF0ewYNl8nLAQDgc4-LOg",
    user_agent="scraper by aaronrodgers10",
)

def get_comments(comment, comments_list):
    comments_list.append({
        'id': comment.id,
        'parent_id': comment.parent_id.split('_')[1],
        'body': comment.body,
        'author': str(comment.author)
    })
    
    if comment.replies:
        for reply in comment.replies:
            get_comments(reply, comments_list)

def search_reddit(query):
    posts_list = []
    for submission in reddit.subreddit("all").search(query, limit=3):
        comments_list = []
        for comment in submission.comments:
            get_comments(comment, comments_list)

        posts_list.append({
            'title': submission.title,
            'url': submission.url,
            'id': submission.id,
            'comments': comments_list,
        })

    return posts_list

def lambda_handler(event, context):
    query = event["queryStringParameters"]["q"]
    posts = search_reddit(query)
    print(json.dumps(posts))
    return {
        'statusCode': 200,
        'body': json.dumps(posts),
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
    }