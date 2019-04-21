# Argument Spider

## what is it?
Argument Spider is a web search engine to retrieve individual argumentative sentences from web data for 
controversial topics.

[Argument Spider is running here](http://ec2-52-13-61-56.us-west-2.compute.amazonaws.com:5000/)  
video  
detailed blogs  
[Google Slides for introduction of Argument Spider](https://docs.google.com/presentation/d/165WDYcDfVdoiy8gR36nTCXvPcLT7KKlicLLaZxge7R0/edit#slide=id.g566ab6e222_0_81)
##Table of Contents
##Introduction
Argumentation is a domain of knowledge that studies debate and reasoning processes.The typical argument structure \
is composed by  two parts:  
  * Topic: a short, usually controversial statement
  * Argument: a span of text expressing evidence or reasoning that can be used to either support or oppose a \
given topic.  

A topic would be part of a (major) claim expressing a positive or negative stance, and our arguments would be premises with supporting/
 attacking consequence relations to the claim.
 In this peoject, topics are restricted  to be expressed through keywords, and arguments that consist of individual sentences.
 
 Besides,deep learning are rapidly evolving recently and show a  huge  potential for automated reasoning. therefor,this project 
 implenment a deep learning model to identify individual argumentative sentences  and 
 classify them as "pros" or "con" for controversial topics.

 
##How does it work?
![alt text](pig/pipline.png "pipline")
This system’s architecture
could be  split into two parts:offine and online processing parts.   
* The offine processing consists of collecting data from common crawl which stored
in S3,batched cleaning the raw data to plain-text with spark and redis,and  document-
indexed into Elasticsearch.  
* the online processing depend on users' query which could be a sentence or phrase of a topic.
elasticsearch will pull out the top-N relevant web pages  based on the query, Then the deep learning
 models which pre-trained implement in Keras/tensorFlow identify the argumentative sentence and 
 determined if the  are support or oppose that queried topic.

## data source 

All the data are from [Common Crawl](http://commoncrawl.org/) 
which contains petabytes of data and lives on Amazon S3. 
It is archived as the Web ARChive (WARC) format which  is the raw data from the crawl, 
providing a direct mapping to the crawl process that Not only does the format store the HTTP response 
from the websites it contacts (WARC-Type: response), it also stores information about how that
 information was requested (WARC-Type: request) and metadata on the crawl process itself (WARC-Type: metadata).


This project utilized a small portion of the data from [2018 March](http://commoncrawl.org/2018/03/march-2018-crawl-archive-now-available/)  which approximates 20T.

