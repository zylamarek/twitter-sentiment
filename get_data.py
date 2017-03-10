from __future__ import print_function
import tweepy
from tweepy import OAuthHandler
import csv
import os
import cPickle as pickle
import HTMLParser
import data_wrapper
import shutil
import random
import ConfigParser

# Load the twitter credentials from get_data.cfg file
config = ConfigParser.ConfigParser()
config.readfp(open('get_data.cfg'))
consumer_key = config.get('twitter_credentials', 'consumer_key')
consumer_secret = config.get('twitter_credentials', 'consumer_secret')
access_token = config.get('twitter_credentials', 'access_token')
access_secret = config.get('twitter_credentials', 'access_secret')

# Path to extracted data
path_in = os.path.join('data', 'extracted')

# Output folder for downloaded tweets and loaded data
path_out_fresh = os.path.join('data', 'fresh')

# Output folder for production data
path_out = os.path.join('data', 'production')

# http://alt.qcri.org/semeval2017/task4/data/uploads/download.zip
path_in_2016 = os.path.join(path_in, 'DOWNLOAD', 'Subtask_A')

# http://alt.qcri.org/semeval2017/task4/data/uploads/codalab/4a-english.zip
path_in_dev2017 = os.path.join(path_in, '4A-English', 'SemEval2017-task4-dev.subtask-A.english.INPUT.txt')

# http://alt.qcri.org/semeval2017/task4/data/uploads/semeval2016_task4_submissions_and_scores.zip
path_in_test2017_gold = os.path.join(path_in, 'SemEval2016_task4_submissions_and_scores', '_scripts',
                                     'SemEval2017_task4_subtaskA_test_english_gold.txt')

# http://alt.qcri.org/semeval2017/task4/data/uploads/semeval2017-task4-test-input-phase1-v3.0.zip
path_in_test2017_txt = os.path.join(path_in, 'SemEval2017-task4-test-input-phase1-v3.0',
                                    'SemEval2017-task4-test.subtask-A.english.txt')

# http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
path_in_stanford = os.path.join(path_in, 'training.1600000.processed.noemoticon.csv')

file_name_out_dev2017 = 'twitter-2017dev-A.p'
file_name_out_test2017 = 'twitter-2017test-A.p'
file_name_stanford = 'stanford1600000.p'

if not os.path.exists(path_out_fresh):
    os.makedirs(path_out_fresh)
if not os.path.exists(path_out):
    os.makedirs(path_out)

tweets_not_existing = 0
files_in = os.listdir(path_in_2016)
files_in = [f for f in files_in if not f.startswith('.')]
all_tweets = 0
all_tweets_loaded = 0

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

# Download the tweets
print('Downloading the tweets...')
for file_name_in in files_in:
    print(file_name_in)
    tweets = []
    with open(os.path.join(path_in_2016, file_name_in), 'rb') as file_read:
        tweet_nums = csv.reader(file_read, delimiter='\t')
        row_count = sum(1 for row in tweet_nums)
        file_read.seek(0)
        all_tweets += row_count
        for idx, tweet_entry in enumerate(tweet_nums):
            tweet_id = tweet_entry[0]
            tweet_label = tweet_entry[1]
            if tweet_label == 'positive':
                tweet_sentiment = 2
            elif tweet_label == 'neutral':
                tweet_sentiment = 1
            else:
                tweet_sentiment = 0
            try:
                text = api.get_status(int(tweet_id)).text
                new_row = [tweet_id, text, tweet_sentiment]
                tweets.append(new_row)
                print('{}/{} '.format(idx + 1, row_count) + str(new_row))
                all_tweets_loaded += 1
            except tweepy.error.TweepError as err:
                tweets_not_existing += 1
                print('{}/{} '.format(idx + 1, row_count))
                print(err)

    with open(os.path.join(path_out_fresh, file_name_in[:-3] + 'p'), 'wb') as pfile:
        pickle.dump(tweets, pfile)

print('Tweets not existing: {}'.format(tweets_not_existing))
print('Tweets downloaded: {}'.format(all_tweets_loaded))
print('Tweets all: {}'.format(all_tweets))

# Load dev2017 dataset
print('Loading dev2017 dataset...')
tweets = []
with open(path_in_dev2017, 'rb') as file_read:
    tweet_nums = csv.reader(file_read, delimiter='\t')
    row_count = sum(1 for row in tweet_nums)
    file_read.seek(0)
    for idx, tweet_entry in enumerate(tweet_nums):
        tweet_id = tweet_entry[0]
        tweet_label = tweet_entry[1]
        text = tweet_entry[2]
        if tweet_label == 'positive':
            tweet_sentiment = 2
        elif tweet_label == 'neutral':
            tweet_sentiment = 1
        else:
            tweet_sentiment = 0

        new_row = [tweet_id, text, tweet_sentiment]
        tweets.append(new_row)
        print('{}/{} '.format(idx + 1, row_count) + str(new_row))
        all_tweets_loaded += 1

with open(os.path.join(path_out_fresh, file_name_out_dev2017), 'wb') as pfile:
    pickle.dump(tweets, pfile)

# Load test2017 dataset
print('Loading test2017 dataset...')
tweets_txt = []
with open(path_in_test2017_txt, 'rb') as file_read:
    tweet_nums = csv.reader(file_read, delimiter='\t', quoting=csv.QUOTE_NONE)
    for idx, tweet_entry in enumerate(tweet_nums):
        tweet_id = tweet_entry[0]
        text = tweet_entry[2]
        new_row = [tweet_id, text]
        tweets_txt.append(new_row)
print('Tweets in test2017 txt: {:d}'.format(len(tweets_txt)))

tweets_gold = []
with open(path_in_test2017_gold, 'rb') as file_read:
    tweet_nums = csv.reader(file_read, delimiter='\t')
    for idx, tweet_entry in enumerate(tweet_nums):
        tweet_id = tweet_entry[0]
        tweet_label = tweet_entry[1]
        if tweet_label == 'positive':
            tweet_sentiment = 2
        elif tweet_label == 'neutral':
            tweet_sentiment = 1
        else:
            tweet_sentiment = 0
        new_row = [tweet_id, tweet_sentiment]
        tweets_gold.append(new_row)
print('Tweets in test2017 gold: {:d}'.format(len(tweets_gold)))

print('The same amount of tweets in test2017txt and test2017gold: ' + str(len(tweets_txt) == len(tweets_gold)))
tweets = [[t_g[0], t_t[1], t_g[1]] for (t_g, t_t) in zip(tweets_gold, tweets_txt) if t_g[0] == t_t[0]]
print('Tweets loaded from test2017: {:d}'.format(len(tweets)))
all_tweets_loaded += len(tweets)

with open(os.path.join(path_out_fresh, file_name_out_test2017), 'wb') as pfile:
    pickle.dump(tweets, pfile)

print('done.')
print('Tweets loaded: {}'.format(all_tweets_loaded))

# Load Stanford 1.6M dataset
tweets = []
print('Loading Stanford 1.6M dataset...')
with open(path_in_stanford, 'rb') as file_s:
    entries = csv.reader(file_s, skipinitialspace=True)
    for e, entry in enumerate(entries):
        if not e % 160000:
            print('.', end='')
        new_row = [entry[1], entry[5], int(int(entry[0])/2.)]
        tweets.append(new_row)
print('')

all_tweets_loaded += len(tweets)
with open(os.path.join(path_out_fresh, file_name_stanford), 'wb') as pfile:
    pickle.dump(tweets, pfile)

print('done.')
print('Tweets loaded: {}'.format(all_tweets_loaded))

# Unescape HTML
files_in = os.listdir(path_out_fresh)
counter = 0
html_parser = HTMLParser.HTMLParser()
print('Unescaping HTML...')
for file_name_in in files_in:
    print(file_name_in)
    with open(os.path.join(path_out_fresh, file_name_in), 'rb') as pfile:
        tweets = pickle.load(pfile)

    new_tweets = []
    for tweet in tweets:
        try:
            tweet[1] = html_parser.unescape(tweet[1].decode('utf-8'))
            new_tweets.append(tweet)
            if len(tweet[1]) > 140:
                print('{:d}: '.format(len(tweet[1])) + tweet[1])
                counter += 1
        except UnicodeDecodeError:
            print('UnicodeDecodeError: ' + tweet[1])
        except UnicodeEncodeError:
            print('UnicodeEncodeError: ' + tweet[1])
    print('{:d}->{:d}'.format(len(tweets), len(new_tweets)))
    with open(os.path.join(path_out, file_name_in), 'wb') as pfile:
        pickle.dump(new_tweets, pfile)

print('done.')
print('tweets with len > 140: {:d}'.format(counter))

# Get all the characters and symbols used in all the datasets except Stanford
# Filter Stanford dataset so only tweets with all chars in this charset are left
print('Loading charset...')
datasets = os.listdir(path_out)
datasets.remove(file_name_stanford)
datasets = [os.path.join(path_out, file_name) for file_name in datasets]
data = data_wrapper.DataWrapper(datasets, 0, 1, temp_dir='tmp_get_data')
char_list = list(data.charset_map)
shutil.rmtree('tmp_get_data', ignore_errors=True)

print('Filtering Stanford dataset...')
with open(os.path.join(path_out, file_name_stanford), 'rb') as pfile:
    tweets = pickle.load(pfile)

new_tweets = []
for i_t, tweet in enumerate(tweets):
    if not i_t % 160000:
        print('.', end='')
    f_pass = True
    for s in tweet[1]:
        if s not in char_list:
            f_pass = False
            break
    if f_pass:
        new_tweets.append(tweet)

print('')
print('{:d}->{:d}'.format(len(tweets), len(new_tweets)))
with open(os.path.join(path_out, file_name_stanford), 'wb') as pfile:
    pickle.dump(new_tweets, pfile)
print('done.')

# Remove repetitions
files_in = os.listdir(path_out)
# Skip stanford, dev2017 and test2017 datasets
files_in.remove(file_name_stanford)
files_in.remove(file_name_out_dev2017)
files_in.remove(file_name_out_test2017)

print('Loading dev2017 dataset...')
with open(os.path.join(path_out, file_name_out_dev2017), 'rb') as pfile:
    orig_dev2017 = pickle.load(pfile)
dev2017 = []
for tweet in orig_dev2017:
    if tweet not in dev2017:
        dev2017.append(tweet)
print('Original dev2017 set size: {:d}'.format(len(orig_dev2017)))
print('  No-rep dev2017 set size: {:d}'.format(len(dev2017)))
with open(os.path.join(path_out, file_name_out_dev2017), 'wb') as pfile:
    pickle.dump(dev2017, pfile)

print('Loading test2017 dataset...')
with open(os.path.join(path_out, file_name_out_test2017), 'rb') as pfile:
    orig_test2017 = pickle.load(pfile)
test2017 = []
counter_dev_test = 0
for tweet in orig_test2017:
    if tweet not in test2017:
        test2017.append(tweet)
        if tweet in dev2017:
            counter_dev_test += 1
print('Original dev2017 set size: {:d}'.format(len(orig_test2017)))
print('  No-rep dev2017 set size: {:d}'.format(len(test2017)))
with open(os.path.join(path_out, file_name_out_test2017), 'wb') as pfile:
    pickle.dump(test2017, pfile)
print('dev and test 2017 intersection: {:d} (not removed)'.format(counter_dev_test))

test2017_ids = [t[0] for t in test2017]
dev2017_ids = [t[0] for t in dev2017]
all_tweet_ids = []
print('Processing the files...')
for file_name_in in files_in:
    print(file_name_in)
    with open(os.path.join(path_out, file_name_in), 'rb') as pfile:
        tweets = pickle.load(pfile)

    new_tweets = []
    c_all, c_test, c_dev = 0, 0, 0
    for tweet in tweets:
        if tweet[0] in all_tweet_ids:
            c_all += 1
        if tweet[0] in test2017_ids:
            c_test += 1
        if tweet[0] in dev2017_ids:
            c_dev += 1
        if tweet[0] not in test2017_ids and tweet[0] not in dev2017_ids and tweet[0] not in all_tweet_ids:
            new_tweets.append(tweet)
            all_tweet_ids.append(tweet[0])

    print('{:d} -> {:d} ({:d} removed, {:d} in test, {:d} in dev, {:d} in all)'.format(len(tweets), len(new_tweets),
                                                                                       len(tweets) - len(new_tweets),
                                                                                       c_test, c_dev, c_all))
    with open(os.path.join(path_out, file_name_in), 'wb') as pfile:
        pickle.dump(new_tweets, pfile)

print('done.')

# Concatenate and rename files into pretraining, training, validation and test datasets
print('Rename stanford dataset...')
os.rename(os.path.join(path_out, file_name_stanford), os.path.join(path_out, 'a_pretrain.p'))

print('Rename dev2017 dataset...')
os.rename(os.path.join(path_out, file_name_out_dev2017), os.path.join(path_out, 'c_valid.p'))

print('Rename test2017 dataset...')
os.rename(os.path.join(path_out, file_name_out_test2017), os.path.join(path_out, 'd_test.p'))

print('Loading... (the rest of the files will make the default 2017 train dataset)')
tweets = []
for file_name_in in files_in:
    print(file_name_in)
    with open(os.path.join(path_out, file_name_in), 'rb') as pfile:
        tweets += pickle.load(pfile)
print('Total train tweets: {:d}'.format(len(tweets)))

print('Shuffling...')
random.shuffle(tweets)

print('Saving...')
with open(os.path.join(path_out, 'b_train.p'), 'wb') as pfile:
    pickle.dump(tweets, pfile)

print('Removing leftover files...')
for file_name in files_in:
    os.remove(os.path.join(path_out, file_name))

print('done.')
