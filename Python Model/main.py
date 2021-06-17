# load libraries
import speech_recognition as sr


# set parameters
recognizer = sr.Recognizer() # to recognie speech
recognizer.energy_threshold = 300


# load audio file
audio_file = sr.AudioFile("G:/rauf/STEPBYSTEP/Projects/SPEECH/Speech Recognition/Python Model/demo_example.wav")
#_>print(type(audio_file))


# lets convert audio data to text 
with audio_file as source:
  audio_file = recognizer.record(source)
  result = recognizer.recognize_google(audio_data=audio_file)
#_>print(result)

# here we go i saved my own voice as saying "Hello my name is Rauf Odilov, I am learning Programming
# and model predicted like "hello how are you my name is Ralph order love I am learning programming" here problem is may be my pronunciation