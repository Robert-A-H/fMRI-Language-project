from __future__ import division
import os
import csv
import numpy as np
import soundfile as sf
import queue
from psychopy import event, visual, core, gui, sound, clock
import time, random, pylab, scipy.io, math
from scipy import signal, stats
import matplotlib.pyplot as plt
import pandas as pd
from numpy.random import RandomState
from psychopy.hardware import keyboard
import sounddevice as sd

import psychtoolbox.audio as audio
from psychopy.sound import microphone
from psychopy.core import getTime

# Start the ioHub process
keb = keyboard.getKeyboards()
print('keyboard:', keb)

# Function to save detailed timestamps to a CSV file
def write_detailed_timestamps_to_csv(timestamps, file_path):
    """
    Converts a list of dictionaries containing detailed audio start and end timestamps
    along with the respective audio filenames into a CSV file.

    Parameters:
    - timestamps: List of dicts, each containing 'task_name', 'audio_start_time', and 'audio_end_time'.
    - file_path: Path to the CSV file.
    """
    header = ["Task Name", "Audio Start Time", "Audio End Time"]

    with open(file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        for timestamp in timestamps:
            row = {
                "Task Name": timestamp['task_name'],
                "Audio Start Time": timestamp['audio_start_time'],
                "Audio End Time": timestamp['audio_end_time']
            }
            writer.writerow(row)

def add_timestamps_to_list(task_name, audio_start_time, audio_end_time, timestamps_list):
    """
    Creates a dictionary from given data and appends it to the timestamps list.
    """
    timestamps_list.append({
        "task_name": task_name,
        "audio_start_time": audio_start_time,
        "audio_end_time": audio_end_time
    })

# Create an empty list for timestamps
timestamps = []

# Specify desired audio output device
desired_device_name = 'Aux'.lower()
devices = sd.query_devices()
print("Available audio devices:", devices)
selected_device_id = None

for index, device in enumerate(devices):
    device_name = device['name'].lower()
    if desired_device_name in device_name:
        selected_device_id = index
        selected_device_name = device['name']
        break

if selected_device_id is not None:
    print(f"Selected audio device: {selected_device_name}")
    sd.default.device = selected_device_id
else:
    print("Desired audio device not found. Using default output device.")

# Set sample rate
sd.default.samplerate = 44100

# Predefined order of audio playback
audiofiles_list = [
    '1_int', '2_degr', '4_degr', '3_int', '6_degr', '5_int', '8_degr', '7_int',
    '9_int', '10_degr', '11_int', '12_degr', '14_degr', '13_int', '15_int', '16_degr',
    '17_int', '18_degr', '20_degr', '19_int', '22_degr', '21_int', '24_degr', '23_int',
    '25_int', '26_degr', '27_int', '28_degr', '30_degr', '29_int', '31_int', '32_degr'
]

# Define delay before playback (in seconds)
delay_before_playback = 1.0

def play_audio_file(audiofile_name, base_path):
    """
    Plays the specified audio file after the defined delay.

    Parameters:
    - audiofile_name: Filename of the audio file without ".wav".
    - base_path: Directory where audio files are located.
    """
    task_name = audiofile_name
    audio_file_path = os.path.join(base_path, f"{audiofile_name}.wav")

    # Display scanner start message
    scanner_gestartet.draw()
    mywin.flip()

    # Load audio file
    data, fs = sf.read(audio_file_path)

    # Wait before playback
    core.wait(delay_before_playback)

    # Display fixation symbol
    fixation.draw()
    mywin.flip()

    # Record start time
    audio_start_time = scanner_time.getTime()
    
    # Play audio file
    sd.play(data, fs)
    print("Audio start time:", audio_start_time)

    # Wait until playback finishes
    sd.wait()

    # Record end time
    audio_end_time = scanner_time.getTime()
    print("Audio end time:", audio_end_time)

    # Clear fixation symbol
    mywin.flip()

    print("Audio playback finished.")

    # Save timestamps
    add_timestamps_to_list(task_name, audio_start_time, audio_end_time, timestamps)

# Initialize GUI for subject info
myDlg = gui.Dlg(title="Subject Information")
myDlg.addField('SubjID:')
myDlg.addField('Run:')
ok_data = myDlg.show()

if myDlg.OK and ok_data is not None:
    subjID = myDlg.data[0]
    Nrun = float(myDlg.data[1])
    print('SubjID:', subjID, 'Run:', Nrun)

    # Define visual elements
    mywin = visual.Window([1280, 720], screen=0, color='black', fullscr=False)
    fixation = visual.TextStim(win=mywin, text='+', height=0.5)
    scanner_wait = visual.TextStim(win=mywin, height=0.1, bold=True, color='white', text='Waiting for scanner...\n\n')
    scanner_gestartet = visual.TextStim(win=mywin, height=0.1, bold=True, color='white', text='Scanner started, playback pending...\n\n')

    # Display waiting message
    scanner_wait.draw(mywin)
    mywin.flip()

    kb = keyboard.Keyboard()
    keys = kb.waitKeys(keyList=['s'], waitRelease=False)

    if keys:
        scanner_time = core.Clock()
        scanner_time.reset()

        # Request user to specify the audio file directory
        base_audio_path = gui.fileOpenDlg(tryFilePath=os.getcwd(), prompt="Select Audio Directory", allowed="")

        if base_audio_path:
            base_audio_path = base_audio_path[0]  # Get the selected directory
            for audiofile_name in audiofiles_list:
                play_audio_file(audiofile_name, base_audio_path)

    # Save timestamps to CSV (prompt user for location)
    save_csv_path = gui.fileSaveDlg(tryFilePath=os.getcwd(), prompt="Save Timestamps CSV", allowed=".csv")
    if save_csv_path:
        write_detailed_timestamps_to_csv(timestamps, save_csv_path)

    # Display experiment completion message
    finish_txt = visual.TextStim(win=mywin, height=0.1, bold=True, color='white', text='Experiment finished. Thank you!')
    finish_txt.draw()
    mywin.flip()
    core.wait(2)

core.quit()
