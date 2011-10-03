
DEBUG = ARGUMENTS.get('debug', 'yes')
if DEBUG == 'yes' or DEBUG == 'true' or DEBUG == '1':
    DEBUG = True
else:
    DEBUG = False

env_cm = Environment()

env_aa = Environment()
env_aa.Append(LIBS=['rt'])

env_ab = Environment()
env_ab.Append(LIBS=['rt'])
env_ab.Tool('nvcc', toolpath = ['/home/kimura/usr/lib/scons/SCons/Tool/'])
env_ab.Append(CPPPATH = ['/home/kimura/dist/cudasdk/C/common/inc'])
env_ab.Append(LIBPATH  = ['/home/kimura/dist/cudasdk/C/lib', '/home/kimura/dist/cudasdk/C/common/lib/linux', '/usr/local/cuda/lib64'])
env_ab.Append(LIBS = ['glut', 'GLEW_x86_64', 'cudart', 'cutil_x86_64'])
if DEBUG:
   env_cm.Append(CCFLAGS='-g')
   env_aa.Append(CCFLAGS='-g')
   env_ab.Append(CCFLAGS='-g')
else:
   env_cm.Append(CCFLAGS='-O')
   env_ab.Append(CCFLAGS='-O')
   env_aa.Append(CCFLAGS='-O')

cm = env_cm.Library(["mx.cpp", "sw.cpp"])

aa = env_aa.Program(["aa.cpp", cm])

ab = env_ab.Program(["ab.cu", cm])

Default([aa, ab])

