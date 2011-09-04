
DEBUG = ARGUMENTS.get('debug', 'yes')
if DEBUG == 'yes' or DEBUG == 'true' or DEBUG == '1':
    DEBUG = True
else:
    DEBUG = False

env = Environment()
if DEBUG:
   env.Append(CCFLAGS='-g')
else:
   env.Append(CCFLAGS='-O')
env.Program("aa.cpp")

