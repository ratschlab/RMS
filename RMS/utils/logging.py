import sys
import datetime

def log(value, stream=sys.stderr):
    ''' Logs to a specified stream'''
    stream.write("LOG: "
                 + datetime.datetime.now().time().isoformat() + " "
                 + sys.argv[0] + " " + str(value) + "\n")
    stream.flush()


