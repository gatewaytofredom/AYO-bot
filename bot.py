import discord
import os
from dotenv import load_dotenv

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm 
import joblib
import datetime

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
SUS_EMOJI = 'AYO'
SUS_EMOJI_ID = 992835477709271090
MODEL_PATH = './sus_meter.pkl'
CV_PATH = './cv.pkl'
SUS_GROUND_ZERO = datetime.datetime(2022, 11, 8, tzinfo=datetime.timezone.utc)

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

sus_meter = None
feat_trans = None

def train_and_save(messages, classifiers):
    x_train, x_test, y_train, y_test = train_test_split(messages, classifiers, test_size=.2)

    cv = CountVectorizer()
    features = cv.fit_transform(x_train)
    joblib.dump(cv, CV_PATH)
    model = svm.SVC()
    model.fit(features, y_train)
    features_test = cv.transform(x_test)
    print(f"Accuracy for test: {model.score(features_test, y_test)}")
    joblib.dump(model, MODEL_PATH)
    return model

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    if not os.path.exists(CV_PATH):
        return None, None
    return joblib.load(MODEL_PATH), joblib.load(CV_PATH)

def check_sus(message):
    if len(message.reactions) == 0:
        return False
    else:
        for reaction in message.reactions:
            if (type(reaction.emoji) is discord.Emoji or type(reaction.emoji) is discord.PartialEmoji) and reaction.emoji.name == SUS_EMOJI:
                return True
    return False

@client.event
async def on_ready():
    global sus_meter
    global feat_trans
    print(f'{client.user} is connected to:', end='')
    for guild in client.guilds:
        if guild.name != 'Worse Company':
            continue
        print(f' {guild}', end='')
    print()
    if sus_meter is None:
        sus_meter, feat_trans = load_model()
    if sus_meter is not None:
        return
    for guild in client.guilds:
        if guild.name != 'Worse Company':
            continue
        messages = []
        nums = []
        for channel in guild.text_channels:
            permissions = channel.permissions_for(guild.self_role)
            print(f'channel {channel.name}')
            print(permissions)
            if not permissions.view_channel:
                continue
            async for message in channel.history(limit=None):
                if message.created_at < SUS_GROUND_ZERO:
                    break 
                if message.content[0:6] == 'https:':
                    continue
                if (len(message.content) > 0 and len(message.content) < 100):
                    messages.append(message.content)
                    if check_sus(message):
                        nums.append(1)
                    else:
                        nums.append(0)
        print(1 in nums)
        train_and_save(messages, nums)
        print('done!')

@client.event
async def on_message(message: discord.Message):
    global feat_trans
    if feat_trans is None or sus_meter is None:
        return
    channel = message.channel
    permissions = channel.permissions_for(message.guild)
    if channel.name == 'test' or (not permissions.view_channel):
        return
    content = message.content
    if sus_meter.predict(feat_trans.transform([content]))[0] == 1:
        print(f'found sus message: {content}')
        await message.add_reaction(client.get_emoji(SUS_EMOJI_ID))
    

client.run(TOKEN)