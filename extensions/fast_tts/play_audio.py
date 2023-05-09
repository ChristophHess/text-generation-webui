import pyaudio
import wave
import logging
from time import sleep

def play(filename):
    CHUNK = 1024

    wf = wave.open(filename, 'rb')

    p = pyaudio.PyAudio()

    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    data = wf.readframes(CHUNK)

    while data != '':
        logging.info("Playing audio")
        print(len(data))
        stream.write(data)
        data = wf.readframes(CHUNK)
        sleep(0.5)

    stream.stop_stream()
    stream.close()

    p.terminate()