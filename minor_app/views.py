from django.shortcuts import render
from minor_app.models import contact
from datetime import datetime
import tweepy
import pandas
from dotenv import load_dotenv
import os

# Create your views here.
import pickle

def index(request):
    return render(request, 'new.html')


def contacts(request):
    if request.method == "POST":

        name = request.POST.get('name')

        email = request.POST.get('email')

        phone = request.POST.get('phone')

        desc = request.POST.get('desc')
        Contact = contact(name=name, email=email, phone=phone,
                          desc=desc, date=datetime.today())
        Contact.save()
    return render(request, 'contact.html')


def about(request):
    return render(request, 'about.html')


BEARER_TOKEN=os.getenv('BEARER_TOKEN')
def results(request):
    data=[]
    
    client = tweepy.Client(bearer_token=BEARER_TOKEN)
    
    query=request.POST.get('search','')
    
    tweets = client.get_user(username=query,user_fields=["description","verified","location","created_at","id","url","public_metrics","profile_image_url"])
 
    image=tweets.data.profile_image_url[:-11]+'.jpg'
   
    
    name=query
    query='from:'+query
    print(query)
    tweets_by_user=client.search_recent_tweets(query=query,max_results=100)
    
    return render(request,'results.html',{'name':name,'links':image,'link':tweets_by_user.data,'followers':tweets.data.public_metrics['followers_count'],'following':tweets.data.public_metrics['following_count'],'desc':tweets.data.description})
    

def new(request):
    return render(request,'new.html')




    
def potential(request):
    data=[]
    client = tweepy.Client(bearer_token=BEARER_TOKEN)
    
    query=request.POST.get('search','')
    print(query)
    tweets = client.get_user(username=query,user_fields=["description","verified","location","created_at","id","url","public_metrics","profile_image_url"])
    
    bag_of_words_bot = ['bot','b0t','cannabis','tweet me','mishear','follow me','updates every','gorilla','yes_ofc','forget',
                    'expos','kill','clit','bb','butt','fuck','XXX','sex','truthe','fake','anony','free','virus','funky','RNA','kuck','jargon' ,
                    'nerd','swag','jack','bang','bonsai','chick','prison','paper','pokem','xx','freak','ffd','dunia','clone','genie'
                    'ffd','onlyman','emoji','joke','troll','droop','free','every','wow','cheese']
    #data.append(query)
    # data.append(tweets.data.name)
    # data.append(tweets.data.description)
    # data.append(tweets.data.id)
    # data.append(tweets.data.verified)
    # data.append(tweets.data.public_metrics['followers_count'])
    # data.append(tweets.data.public_metrics['listed_count'])
    # data.append(tweets.data.public_metrics['listed_count'])
    # data.append(tweets.data.public_metrics['listed_count'])
    # data.append(tweets.data.location)
    # data.append(tweets.data.created_at)
    # data.append(tweets.data.description)
    # data.append(tweets.data.profile_image_url)
   

    ok=0
    for i in bag_of_words_bot:
        if(i in query):
            data.append(True)
            ok=1
            break

    if(ok==0):
        data.append(False)

    ok=0    
    for i in bag_of_words_bot:
        if(i in tweets.data.name):
            data.append(True)
            ok=1
            break

    if(ok==0):
        data.append(False)


    ok=0 
    for i in bag_of_words_bot:
        if(i in tweets.data.description):
            data.append(True)
            ok=1
            break

    if(ok==0):
        data.append(False)


    ok=0

    for i in bag_of_words_bot:
        if(i in query):
            data.append(True)
            ok=1
            break

    if(ok==0):
        data.append(False)



    data.append(tweets.data.verified)
    data.append(tweets.data.public_metrics['followers_count'])
    data.append(tweets.data.public_metrics['following_count'])

    tweets_by_user=client.search_recent_tweets(query=query,max_results=100)

    data.append(len(tweets_by_user))
    
    
    if(tweets.data.public_metrics['listed_count']>0):
        data.append(True)
    else:
        data.append(False)
    

    image=tweets.data.profile_image_url[:-11]+'.jpg'
    
    
    model = pickle.load(open("forest.pkl", "rb"))
    prediction = model.predict([data])
    
     
    
    return render(request,'new.html',{'prediction':prediction})

features = ['screen_name_binary', 'name_binary', 'description_binary', 'status_binary', 'verified', 'followers_count', 'friends_count', 'statuses_count', 'listed_count_binary', 'bot']



