#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Trigger api gateway 
try:
    import boto3
    from boto3 import resource
    from boto3.dynamodb.conditions import Key
except:
    boto3 = None
    pass
 
PROFILES_TABLE_NAME = 'Metrics'
# The boto3 dynamoDB resource
dynamodb_resource = resource('dynamodb')

def lambda_handler(event, context):
    result = scan_table_allpages(PROFILES_TABLE_NAME)
    return result

def scan_table_allpages(table_name, filter_key=None, filter_value=None):
    """
    Perform a scan operation on table. 
    Can specify filter_key (col name) and its value to be filtered. 
    This gets all pages of results. Returns list of items.
    """
    table = dynamodb_resource.Table(table_name)

    if filter_key and filter_value:
        filtering_exp = Key(filter_key).eq(filter_value)
        response = table.scan(FilterExpression=filtering_exp)
    else:
        response = table.scan()

    items = response['Items']
    while True:
        print len(response['Items'])
        if response.get('LastEvaluatedKey'):
            response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
            items += response['Items']
        else:
            break

    return items
    
# def lambda_handler(event, context):
#     document = load_document('id') or {}
#     return document

def load_document(key):
    print (key)
    dynamodb = boto3.resource('dynamodb')
    if boto3 is not None:
        try:
            response = dynamodb.Table(PROFILES_TABLE_NAME).query(
                KeyConditionExpression=Key('id').eq(key)
             )
            print ("response from DB", response)
            if 'Items' in response:
                return (response['Items'][0]['visit_count'],response['Items'][0]['created_time'])
            else:
                return None
        except Exception as e:
            print("Exception" + str(e))
    return None

