# -*- coding: utf-8 -*-
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
from hashlib import sha1
import json
import pathlib
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os

from config import *
cache_dir += '/acquire/'

fs = 16000  # Sample rate
seconds = 2  # Duration of recording
# keywords = ["edison", "on", "off", "livingroomlight", "kitchenlight",
#             "bedroomlight", "cinema"]
keywords = ["livingroom", "kitchen", "bedroom", "office", "off", "on"]
out_dir = cache_dir+"/noah/"
pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
max_cold_word_length = 4 # maximum number of words in a cold word sequence
heading = 1600 # frames before threshold of cutting
tailing = 3200 # frames after threshold of cutting
percent_of_average_volume = 0.2

stats_per_keywords = [0]*(1+len(keywords))

for k in keywords:
    pathlib.Path(out_dir+k).mkdir(parents=True, exist_ok=True)
    stats_per_keywords[keywords.index(k)] = len([name for name in os.listdir(out_dir+k) if os.path.isfile(out_dir+k+'/'+name)])

# dictionary that contains all input strings and actions to which they lead
actions = {"r": "cancel"}

for i in range(1, len(keywords) + 1):
    actions[str(i)] = keywords[i - 1]

# make empty dictionaries and lists for later
files = {}
cold_words = []

# Read cold words from text file
with open('edison/acquire/cold_words.txt','r') as f:
    for line in f:
        for word in line.split():
            if len(word) < 2: # if word is shorter than 2 letters, wo don't want it
                continue
            else:
                cold_words.append(word)

## Program
while True:
    # Define random cold word(s)
    random_cold_word = np.random.randint(0, len(cold_words)) # random position of word in text file
    random_cold_word_length = np.random.randint(0, max_cold_word_length) # random amount of words in cold word sequence
    cold_word = ""
    for i in range(random_cold_word_length):
        cold_word = cold_word + cold_words[random_cold_word + i] + " " # Create cold word sequence

    # Add cold word to keywords and actions
    keywords.append(cold_word)
    actions[str(len(keywords))] = cold_word

    # Print possibilities
    print("Possible actions:")
    for action_number in actions:
        if actions[action_number] in keywords:
            print(action_number, ':', actions[action_number], "(%d)" % (stats_per_keywords[keywords.index(actions[action_number])]))
        else:
            print(action_number, ':', actions[action_number])
    # Define random action in this round
    random_action = np.random.randint(0, len(keywords))
    random_word = keywords[random_action]
    random_word_is_cold = random_word == cold_word
    print(" ", ":", random_word, "(%d)" % (stats_per_keywords[keywords.index(random_word)]))

    # Get input
    task = input("What do you like to do?")

    # Cancel the loop
    if task == "r":
        break


    else:
        # Recording
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        print("I am listening")
        sd.wait()  # Wait until recording is finished
        myrecording = myrecording[:,0]

        # Define RMS of myrecording
        overall_rms = np.sqrt(np.mean(myrecording ** 2))
        min_volume = percent_of_average_volume * overall_rms

        # Find first and last entry in myrecording, higher than min_volume
        start_index = np.where(myrecording > min_volume)[0][0] - heading
        end_index = np.where(myrecording > min_volume)[0][-1] + heading
        print("Start index:", start_index)
        print("End index:", end_index)

        # Cut recording (remove before and after treshold)
        myrecording = myrecording[start_index: end_index]
        print("length myrecording:", len(myrecording))


        # Define direcotry depending on task if necessary
        if task == "":
            if random_word_is_cold:
                directory = out_dir + "_cold_word"
            else:
                directory = out_dir + random_word
                stats_per_keywords[keywords.index(random_word)] += 1

        else:
            directory = out_dir + keywords[task]
            stats_per_keywords[keywords.index(keywords[task])] += 1

        file_name = directory + '/' + sha1(myrecording).hexdigest()[:8] + '.wav'

        # Save file
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True) # Make directory, if not already present
        write(file_name, fs, myrecording)  # Save as WAV file

        # Write to json file
        # try:
        #    with open('digest.json', 'r') as fd:
        #        jfile = json.load(fd)

        # except:
        #     jfile = []

        # if task == "":
        #     jfile.append({'file_path': file_name,
        #                   'keyword': "cold"})
        # else:
        #     jfile.append({'file_path': file_name,
        #                   'keyword': keywords[int(task)]})

        # with open('digest.json', 'w') as fd:
        #     json.dump(jfile, fd, indent = 2)

        keywords = keywords[:-1]